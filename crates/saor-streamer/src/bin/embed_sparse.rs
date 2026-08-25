//! `embed_sparse`: embebe bloques FFN dispersos (D16) en una copia del GGUF del
//! profesor, desde un perfil de esparsidad por capa (poda por magnitud).
//!
//!   embed_sparse --model <orig.gguf> --out <embedded.gguf>
//!               --weights <dir> --sparsities <file> [--tau 0.42]
//!
//! `--weights` es el directorio de `dump_weights` (hayai): `w.{layer}.{block}.bin`
//! con `d_out*d_in` f32 en orden i-mayor. `--sparsities` tiene un float por línea
//! (esparsidad del gate por capa; 0 = densa). Reescritura **streaming** (sin
//! cargar el archivo completo). Es el camino de producción para el evaluador de
//! la frontera de Pareto en modelos grandes (ALIA/Qwen) sin depender del OpenCL.

use std::path::PathBuf;

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
fn magnitude_prune(w: &[f32], d_in: usize, d_out: usize, sp: f32) -> (Vec<u8>, Vec<f32>) {
    let total = d_in * d_out;
    let keep = (((1.0 - sp) * total as f32) as usize).min(total);
    let mut order: Vec<usize> = (0..total).collect();
    order.sort_by(|&a, &b| w[b].abs().total_cmp(&w[a].abs()));
    let mut active = vec![false; total];
    for &idx in order.iter().take(keep) {
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


fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut weights_dir: Option<PathBuf> = None;
    let mut sparsities: Option<PathBuf> = None;
    let mut tau = 0.42f32;
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
            "--tau" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    tau = v;
                }
            }
            other => {
                return Err(format!("argumento desconocido '{other}'"));
            }
        }
        i += 1;
    }
    let model = model.ok_or("falta --model <gguf>")?;
    let out = out.ok_or("falta --out <gguf>")?;
    let weights_dir = weights_dir.ok_or("falta --weights <dir>")?;
    let sparsities = sparsities.ok_or("falta --sparsities <file>")?;

    let sp_raw: Vec<String> = std::fs::read_to_string(&sparsities)
        .map_err(|e| format!("leer sparsities: {e}"))?
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    let sp: Vec<f32> = sp_raw
        .iter()
        .map(|l| l.split_whitespace().next().and_then(|s| s.parse().ok()).unwrap_or(0.0))
        .collect();

    let meta = std::fs::read_to_string(weights_dir.join("meta.json"))
        .map_err(|e| format!("leer meta.json: {e}"))?;
    let meta_json: serde_json::Value =
        serde_json::from_str(&meta).map_err(|e| format!("parse meta.json: {e}"))?;
    let n_layers = meta_json["n_layers"].as_u64().unwrap_or(0) as usize;

    let mut replacements: Vec<BlockReplacement> = Vec::new();
    for layer in 0..n_layers {
        let layer_sp = sp.get(layer).copied().unwrap_or(0.0);
        if layer_sp <= 0.0 {
            continue;
        }
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
            // Solo el gate se poda por capa; up/down se mantienen densos (Vía A).
            let block_sp = if block == "ffn_gate" { layer_sp } else { 0.0 };
            let (adjacency, weights) = magnitude_prune(&w, d_in, d_out, block_sp.min(0.999));
            replacements.push(BlockReplacement {
                tensor: format!("blk.{layer}.{block}.weight"),
                block: SparseBlock {
                    d_in,
                    d_out,
                    tau,
                    genome: Vec::new(),
                    adjacency,
                    weights,
                },
            });
        }
    }

    let report = rewrite_embedded(&model, &out, &replacements)?;
    println!(
        "{{\"ok\":true,\"out\":{},\"replaced\":{},\"kept_bytes\":{},\"sparse_bytes\":{},\"total_bytes\":{}}}",
        serde_json::Value::String(out.display().to_string()),
        report.replaced,
        report.kept_bytes,
        report.sparse_bytes,
        report.total_bytes
    );
    Ok(())
}
