//! Subcomando `decode-pop`: decodifica la topología CPPN de **toda la población**
//! en la GPU con el kernel batcheado (Fase 1) y escribe la adyacencia de cada
//! candidato a fichero para que `eval_sparse --adj-dir` (hayai) la consuma sin
//! volver a decodificar en CPU.
//!
//!   saor-engine decode-pop --genomes <dir> --out <dir> --d-in N --d-out N
//!                          --n-layers N --tau F [--blocks gate|gate,up,down]
//!
//! Salida: `out/meta.json` + `out/c<C>.l<L>.<block>.bin` (bit-tensor por
//! (candidato, capa, bloque), n_bytes = ceil(d_in_b*d_out_b/8)).

use std::path::PathBuf;
use std::process::ExitCode;

use saor_opencl::compute::ClEngine;
use serde_json::json;

/// Maneja `saor-engine decode-pop ...`.
pub fn cmd(args: &[String]) -> ExitCode {
    let mut genomes_dir: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut d_in = 0usize;
    let mut d_out = 0usize;
    let mut n_layers = 0usize;
    let mut tau = 0.42f32;
    let mut blocks: Vec<String> = vec!["ffn_gate".into(), "ffn_up".into(), "ffn_down".into()];
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--genomes" => {
                i += 1;
                genomes_dir = args.get(i).map(PathBuf::from);
            }
            "--out" => {
                i += 1;
                out_dir = args.get(i).map(PathBuf::from);
            }
            "--d-in" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    d_in = v;
                }
            }
            "--d-out" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    d_out = v;
                }
            }
            "--n-layers" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    n_layers = v;
                }
            }
            "--tau" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    tau = v;
                }
            }
            "--blocks" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    blocks = v
                        .split(',')
                        .map(|s| {
                            match s.trim() {
                                "gate" => "ffn_gate".to_string(),
                                "up" => "ffn_up".to_string(),
                                "down" => "ffn_down".to_string(),
                                other => other.to_string(),
                            }
                        })
                        .filter(|s| !s.is_empty())
                        .collect();
                }
            }
            other => {
                eprintln!("decode-pop: argumento desconocido '{other}'");
                return ExitCode::from(2);
            }
        }
        i += 1;
    }

    let genomes_dir = match genomes_dir {
        Some(p) => p,
        None => {
            eprintln!("decode-pop: falta --genomes <dir>");
            return ExitCode::from(2);
        }
    };
    let out_dir = match out_dir {
        Some(p) => p,
        None => {
            eprintln!("decode-pop: falta --out <dir>");
            return ExitCode::from(2);
        }
    };
    if d_in == 0 || d_out == 0 || n_layers == 0 {
        eprintln!("decode-pop: falta --d-in/--d-out/--n-layers");
        return ExitCode::from(2);
    }

    // Lee los genomas (466 f32 cada uno).
    let mut files: Vec<_> = match std::fs::read_dir(&genomes_dir) {
        Ok(rd) => rd.filter_map(|e| e.ok()).filter(|e| {
            e.path().extension().map(|x| x == "bin").unwrap_or(false)
        }).collect(),
        Err(e) => {
            eprintln!("decode-pop: leer {genomes_dir:?}: {e}");
            return ExitCode::FAILURE;
        }
    };
    files.sort_by_key(|e| e.file_name());
    const GENOME_BYTES: usize = 466 * 4;
    let mut genomes: Vec<Vec<f32>> = Vec::new();
    for e in files {
        let raw = match std::fs::read(e.path()) {
            Ok(r) => r,
            Err(err) => {
                eprintln!("decode-pop: leer {}: {err}", e.path().display());
                return ExitCode::FAILURE;
            }
        };
        if raw.len() != GENOME_BYTES {
            eprintln!(
                "decode-pop: ignorando {} ({} B ≠ genoma {GENOME_BYTES} B)",
                e.path().display(),
                raw.len()
            );
            continue;
        }
        let mut g = Vec::with_capacity(466);
        for c in raw.chunks_exact(4) {
            g.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
        genomes.push(g);
    }
    if genomes.is_empty() {
        eprintln!("decode-pop: {genomes_dir:?} sin genomas válidos");
        return ExitCode::FAILURE;
    }
    let n_cand = genomes.len();
    let flat: Vec<f32> = genomes.iter().flatten().copied().collect();
    eprintln!(
        "[decode-pop] {n_cand} genomas × {n_layers} capas × {} bloques → {out_dir:?}",
        blocks.len()
    );

    let mut engine = match ClEngine::init().and_then(|mut e| {
        e.prepare()?;
        Ok(e)
    }) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("decode-pop: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        eprintln!("decode-pop: crear {out_dir:?}: {e}");
        return ExitCode::FAILURE;
    }

    // Por (capa, bloque): un dispatch batcheado [conexiones, N].
    let mut total_adj_bytes = 0u64;
    for layer in 0..n_layers {
        for block in &blocks {
            // gate/up: [d_in → d_out]; down: [d_out → d_in] (como embed_sparse).
            let (b_in, b_out) = if block == "ffn_down" {
                (d_out, d_in)
            } else {
                (d_in, d_out)
            };
            let res = engine.cppn_decode_adjacency_batched(
                &flat,
                n_cand,
                b_in,
                b_out,
                tau,
                layer,
                n_layers,
            );
            let (adj_concat, active) = match res {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("decode-pop: decode capa {layer} {block}: {e}");
                    return ExitCode::FAILURE;
                }
            };
            let n_bytes = b_in * b_out / 8;
            for c in 0..n_cand {
                let name = format!("c{c:03}.l{layer:02}.{block}.bin");
                let path = out_dir.join(&name);
                if let Err(e) = std::fs::write(&path, &adj_concat[c * n_bytes..(c + 1) * n_bytes]) {
                    eprintln!("decode-pop: escribir {path:?}: {e}");
                    return ExitCode::FAILURE;
                }
            }
            total_adj_bytes += (n_bytes * n_cand) as u64;
            eprintln!(
                "[decode-pop] capa {layer:3}/{n_layers} {block}: activos {}",
                active.iter().sum::<u32>()
            );
        }
    }

    let meta = json!({
        "n_candidates": n_cand,
        "n_layers": n_layers,
        "d_in": d_in,
        "d_out": d_out,
        "tau": tau,
        "blocks": blocks,
        "genome_dim": 466,
        "adj_bytes": total_adj_bytes,
    });
    let meta_path = out_dir.join("meta.json");
    if let Err(e) = std::fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap()) {
        eprintln!("decode-pop: escribir {meta_path:?}: {e}");
        return ExitCode::FAILURE;
    }
    println!(
        "{{\"ok\":true,\"n_candidates\":{n_cand},\"n_layers\":{n_layers},\"blocks\":{},\"adj_bytes\":{total_adj_bytes}}}",
        blocks.len()
    );
    ExitCode::SUCCESS
}

