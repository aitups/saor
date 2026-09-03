//! `embed_sparse`: embebe bloques FFN dispersos (D16) en una copia del GGUF del
//! profesor, desde un perfil de esparsidad por capa (poda por magnitud) o desde
//! la topología CPPN global (`--genome`, Vía B).
//!
//!   embed_sparse --model <orig.gguf> --out <embedded.gguf>
//!               --weights <dir> --sparsities <file> [--tau 0.42]
//!   embed_sparse --model <orig.gguf> --out <embedded.gguf>
//!               --weights <dir> --genome <genome.bin> --tau <f> [--gpu]
//!
//! `--weights` es el directorio de `dump_weights` (hayai): `w.{layer}.{block}.bin`
//! con `d_out*d_in` f32 en orden i-mayor. `--sparsities` tiene un float por línea
//! (esparsidad del gate por capa; 0 = densa). `--genome` decodifica la topología
//! CPPN global por capa (`y_layer`); `--gpu` ejecuta el decode en la GPU vía el
//! kernel OpenCL `cppn_decode_adj` (necesario para ALIA-40b: 201M conexiones × 48
//! capas ≈ 9.6G evaluaciones, inviable en CPU). Reescritura **streaming** (sin
//! cargar el archivo completo).

use std::path::PathBuf;

use saor_opencl::compute::ClEngine;
use saor_streamer::gguf_embed::{rewrite_embedded, BlockReplacement};
use saor_streamer::gguf_sparse::SparseBlock;

fn read_f32_bin(path: &std::path::Path) -> Result<Vec<f32>, String> {
    let raw = std::fs::read(path).map_err(|e| format!("leer {}: {e}", path.display()))?;
    if raw.len() % 4 != 0 {
        return Err(format!("{}: tamaño no múltiplo de 4", path.display()));
    }
    let n = raw.len() / 4;
    let mut v = Vec::with_capacity(n);
    for c in raw.chunks_exact(4) {
        v.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    Ok(v)
}

/// Poda por magnitud del gate del profesor: conserva la fracción (1-sp) de mayor
/// |w|; devuelve el bit-tensor (i-mayor) y los pesos activos en orden de escaneo.
///
/// Con `order_cache` (índices ya ordenados por |w| desc, escritos por
/// `write_order_cache`) el sort se omite: el orden de magnitud es invariante a
/// la esparsidad, así un barrido de frontera reusa el mismo orden en cada punto.
fn magnitude_prune(
    w: &[f32],
    d_in: usize,
    d_out: usize,
    sp: f32,
    order_cache: Option<&[u32]>,
) -> (Vec<u8>, Vec<f32>) {
    let total = d_in * d_out;
    let keep = (((1.0 - sp) * total as f32) as usize).min(total);
    let mut order: Vec<usize> = match order_cache {
        Some(c) if c.len() == total => c.iter().map(|&i| i as usize).collect(),
        _ => {
            let mut o: Vec<usize> = (0..total).collect();
            use rayon::prelude::*;
            o.par_sort_unstable_by(|&a, &b| w[b].abs().total_cmp(&w[a].abs()));
            o
        }
    };
    order.truncate(keep);
    let mut active = vec![false; total];
    for &idx in &order {
        active[idx] = true;
    }
    let mut bits = vec![0u8; total.div_ceil(8)];
    let mut weights = Vec::with_capacity(keep);
    for i in 0..d_in {
        for j in 0..d_out {
            let conn = i * d_out + j;
            if active[conn] {
                bits[conn / 8] |= 1 << (conn % 8);
                weights.push(w[j * d_in + i]);
            }
        }
    }
    (bits, weights)
}

/// Guarda el orden por magnitud (índices desc por |w|) como u32 LE.
fn write_order_cache(w: &[f32], path: &std::path::Path) {
    let mut order: Vec<usize> = (0..w.len()).collect();
    use rayon::prelude::*;
    order.par_sort_unstable_by(|&a, &b| w[b].abs().total_cmp(&w[a].abs()));
    let mut buf = Vec::with_capacity(order.len() * 4);
    for i in &order {
        buf.extend_from_slice(&(*i as u32).to_le_bytes());
    }
    let _ = std::fs::write(path, buf);
}


fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut weights_dir: Option<PathBuf> = None;
    let mut sparsities: Option<PathBuf> = None;
    let mut genome: Option<PathBuf> = None;
    let mut tau = 0.42f32;
    let mut all_blocks = false;
    let mut use_gpu = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model = args.get(i).map(PathBuf::from);
            }
            "--out" => {
                i += 1;
                out = args.get(i).map(PathBuf::from);
            }
            "--weights" => {
                i += 1;
                weights_dir = args.get(i).map(PathBuf::from);
            }
            "--sparsities" => {
                i += 1;
                sparsities = args.get(i).map(PathBuf::from);
            }
            "--genome" => {
                i += 1;
                genome = args.get(i).map(PathBuf::from);
            }
            "--tau" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    tau = v;
                }
            }
            "--gpu" => use_gpu = true,
            "--all-blocks" => all_blocks = true,
            other => {
                return Err(format!("argumento desconocido '{other}'"));
            }
        }
        i += 1;
    }
    let model = model.ok_or("falta --model <gguf>")?;
    let out = out.ok_or("falta --out <gguf>")?;
    let weights_dir = weights_dir.ok_or("falta --weights <dir>")?;
    if use_gpu && genome.is_none() {
        return Err("--gpu requiere --genome <genome.bin>".into());
    }

    let meta = std::fs::read_to_string(weights_dir.join("meta.json"))
        .map_err(|e| format!("leer meta.json: {e}"))?;
    let meta_json: serde_json::Value =
        serde_json::from_str(&meta).map_err(|e| format!("parse meta.json: {e}"))?;
    let n_layers = meta_json["n_layers"].as_u64().unwrap_or(0) as usize;

    // Modo topología CPPN (Vía B): `--genome <genome.bin> --tau <f>` decodifica la
    // adyacencia de cada capa (sustrato v5 con y_layer) y conserva los pesos del
    // profesor en las posiciones activas. Sin él, `--sparsities` (poda por magnitud).
    let cppn_genome: Option<saor_domain::cppn::CppnGenome> = match &genome {
        Some(p) => {
            let flat = read_f32_bin(p)?;
            Some(saor_domain::cppn::CppnGenome::from_flatten(&flat))
        }
        None => None,
    };

    let sp: Vec<f32> = match &sparsities {
        Some(p) => std::fs::read_to_string(p)
            .map_err(|e| format!("leer sparsities: {e}"))?
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .map(|l| l.split_whitespace().next().and_then(|s| s.parse().ok()).unwrap_or(0.0))
            .collect(),
        None => Vec::new(),
    };

    // Motor OpenCL para el decode por GPU (Vía B). `--gpu` es explícito: sin
    // GPU disponible falla (no hay fallback silencioso — en ALIA-40b el decode
    // por CPU es inviable: 9.6G evaluaciones de CPPN).
    let mut engine: Option<ClEngine> = None;
    if use_gpu {
        let mut e = ClEngine::init()
            .map_err(|err| format!("--gpu solicitado pero no hay dispositivo OpenCL: {err}"))?;
        e.prepare()?;
        eprintln!("embed_sparse: decode por GPU en '{}'", e.device_name());
        engine = Some(e);
    }

    let mut replacements: Vec<BlockReplacement> = Vec::new();
    for layer in 0..n_layers {
        let y_layer = saor_domain::cppn::layer_coord(layer, n_layers);
        for block in ["ffn_gate", "ffn_up", "ffn_down"] {
            let key = format!("blk.{layer}.{block}");
            let d_in = meta_json[&key]["d_in"].as_u64().unwrap_or(0) as usize;
            let d_out = meta_json[&key]["d_out"].as_u64().unwrap_or(0) as usize;
            if d_in == 0 || d_out == 0 {
                continue;
            }
            let wpath = weights_dir.join(format!("w.{layer}.{block}.bin"));
            let w = match read_f32_bin(&wpath) {
                Ok(w) => w,
                Err(_) => continue,
            };
            if w.len() != d_out * d_in {
                return Err(format!(
                    "w.{layer}.{block}: esperaba {} f32, hay {}",
                    d_out * d_in,
                    w.len()
                ));
            }
            // Adyacencia y conteo de activos de esta capa/bloque.
            let (adjacency, active_count) = if let Some(e) = &engine {
                // GPU (Vía B): kernel `cppn_decode_adj` — la coordenada `y_layer`
                // se deriva dentro del kernel a partir de (layer, n_layers),
                // idéntica a `saor_domain::cppn::layer_coord`.
                let g = cppn_genome.as_ref().expect("--gpu requiere --genome");
                let flat = g.flatten();
                let (adj, act) =
                    e.cppn_decode_adjacency(&flat, d_in, d_out, tau, layer, n_layers)?;
                (adj, act as usize)
            } else if let Some(g) = &cppn_genome {
                // Vía B CPU: topología CPPN global (coordenada de capa y_layer).
                let topo = saor_domain::topology::instantiate_layer(g, d_in, d_out, tau, y_layer);
                (topo.adjacency_bits, topo.weights.len())
            } else {
                let layer_sp = sp.get(layer).copied().unwrap_or(0.0);
                let block_sp = if all_blocks || block == "ffn_gate" { layer_sp } else { 0.0 };
                if block_sp <= 0.0 {
                    continue; // no reemplazar: el tensor denso original se conserva
                }
                // Caché del orden por magnitud (invariante a la esparsidad).
                let order_path = weights_dir.join(format!("order.{layer}.{block}.bin"));
                let order_cache: Option<Vec<u32>> = match std::fs::read(&order_path) {
                    Ok(raw) if raw.len() == w.len() * 4 => Some(
                        raw.chunks_exact(4)
                            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect(),
                    ),
                    _ => {
                        write_order_cache(&w, &order_path);
                        None
                    }
                };
                let (adj, _) = magnitude_prune(
                    &w,
                    d_in,
                    d_out,
                    block_sp.min(0.999),
                    order_cache.as_deref(),
                );
                let n = adj.iter().map(|b| b.count_ones() as usize).sum();
                (adj, n)
            };
            // Pesos del profesor en las posiciones activas (escaneo i-mayor).
            let mut weights = Vec::with_capacity(active_count);
            for i in 0..d_in {
                for j in 0..d_out {
                    let conn = i * d_out + j;
                    if adjacency[conn / 8] & (1 << (conn % 8)) != 0 {
                        weights.push(w[j * d_in + i]);
                    }
                }
            }
            // Los pesos activos se cuantizan a **Q4_K** y se escriben a fichero
            // (streaming): para ALIA-40b no caben en RAM y el Q4_K reduce el
            // archivo y la banda de memoria frente al F32 (el dequant lo hace
            // `hayai` al leer el bloque disperso). Se rellena a múltiplo de 256.
            let wfile = weights_dir.join(format!("wdata.{layer}.{block}.bin"));
            let pad = (256 - weights.len() % 256) % 256;
            let mut padded = weights;
            padded.resize(padded.len() + pad, 0.0);
            let q4 = saor_streamer::q4k::quantize_q4_k(&padded, padded.len())?;
            std::fs::write(&wfile, &q4)
                .map_err(|e| format!("escribir pesos q4 {wfile:?}: {e}"))?;
            drop(q4);
            replacements.push(BlockReplacement {
                tensor: format!("blk.{layer}.{block}.weight"),
                block: SparseBlock {
                    d_in,
                    d_out,
                    tau,
                    genome: Vec::new(),
                    adjacency,
                    weights: Vec::new(),
                },
                weights_file: Some(wfile),
                weights_q4: true,
            });
        }
    }

    let report = rewrite_embedded(&model, &out, &replacements)?;
    let device = engine.as_ref().map(|e| e.device_name()).unwrap_or_else(|| "cpu".into());
    println!(
        "{{\"ok\":true,\"out\":{},\"replaced\":{},\"kept_bytes\":{},\"sparse_bytes\":{},\"total_bytes\":{},\"device\":{}}}",
        serde_json::Value::String(out.display().to_string()),
        report.replaced,
        report.kept_bytes,
        report.sparse_bytes,
        report.total_bytes,
        serde_json::Value::String(device)
    );
    Ok(())
}
