//! Subcomando `make-block`: ensambla un `SparseBlock` desde bins crudos.
//!
//! Útil para baselines y checkpoints de escala sin correr la evolución:
//! `saor-engine make-block --d-in 8192 --d-out 24576 --tau 0.0
//!   --adj <bit-tensor.bin> --weights <f32.bin> --genome <f32.bin> --out <gguf>`.
//! La adyacencia identidad (todo activo) se genera con `--identity`.

use std::path::PathBuf;
use std::process::ExitCode;

use saor_streamer::gguf_sparse::{write_sparse_gguf, SparseBlock};

fn read_f32_bin(path: &PathBuf) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("leer {path:?}: {e}"))?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{path:?} no tiene tamaño múltiplo de 4"));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(out)
}

/// Maneja `saor-engine make-block --d-in N --d-out N --tau F --out <gguf>`.
pub fn cmd(args: &[String]) -> ExitCode {
    let mut d_in = 0usize;
    let mut d_out = 0usize;
    let mut tau = 0.0f32;
    let mut adj_path: Option<PathBuf> = None;
    let mut weights_path: Option<PathBuf> = None;
    let mut genome_path: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut identity = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
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
            "--tau" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    tau = v;
                }
            }
            "--identity" => {
                identity = true;
            }
            "--adj" => {
                i += 1;
                adj_path = args.get(i).map(PathBuf::from);
            }
            "--weights" => {
                i += 1;
                weights_path = args.get(i).map(PathBuf::from);
            }
            "--genome" => {
                i += 1;
                genome_path = args.get(i).map(PathBuf::from);
            }
            "--out" => {
                i += 1;
                out = args.get(i).map(PathBuf::from);
            }
            other => {
                eprintln!("make-block: argumento desconocido '{other}'");
                return ExitCode::from(2);
            }
        }
        i += 1;
    }

    if d_in == 0 || d_out == 0 {
        eprintln!("make-block: falta --d-in/--d-out");
        return ExitCode::from(2);
    }
    let out = match out {
        Some(p) => p,
        None => {
            eprintln!("make-block: falta --out <gguf>");
            return ExitCode::from(2);
        }
    };
    let total = d_in * d_out;

    let adjacency = if identity {
        let mut adj = vec![0xffu8; total.div_ceil(8)];
        let rem = total % 8;
        if rem != 0 {
            *adj.last_mut().unwrap() = (1 << rem) - 1; // bits sobrantes a 0
        }
        adj
    } else {
        let path = match &adj_path {
            Some(p) => p,
            None => {
                eprintln!("make-block: falta --identity o --adj <bit-tensor.bin>");
                return ExitCode::from(2);
            }
        };
        match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(e) => {
                eprintln!("make-block: leer {path:?}: {e}");
                return ExitCode::FAILURE;
            }
        }
    };

    let weights_path = match &weights_path {
        Some(p) => p,
        None => {
            eprintln!("make-block: falta --weights <f32.bin>");
            return ExitCode::FAILURE;
        }
    };
    let weights = match read_f32_bin(weights_path) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("make-block: {e}");
            return ExitCode::FAILURE;
        }
    };
    let genome = match &genome_path {
        Some(p) => read_f32_bin(p).unwrap_or_default(),
        None => Vec::new(),
    };

    let weights_len = weights.len();
    let block = SparseBlock {
        d_in,
        d_out,
        tau,
        genome,
        adjacency,
        weights,
    };
    match write_sparse_gguf(&out, &block) {
        Ok(()) => {
            println!(
                "{}",
                serde_json::json!({
                    "ok": true,
                    "gguf": out.display().to_string(),
                    "d_in": d_in,
                    "d_out": d_out,
                    "tau": tau,
                    "active": block.active_connections(),
                    "sparsity": block.sparsity(),
                    "weights": weights_len,
                })
                .to_string()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("make-block: {e}");
            ExitCode::FAILURE
        }
    }
}
