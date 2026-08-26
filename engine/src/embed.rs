//! Subcomando `embed`: embebe bloques dispersos en un GGUF completo (Fase 1).
//!
//! Sustituye tensores densos (`blk.N.<rol>.weight`) por bloques dispersos
//! (`ffn_dag_adjacency` + `ffn_dag_weights` + metadatos `saor.<base>.*`) con el
//! rewriter streaming de `saor-streamer` (nunca carga el archivo completo).

use std::path::PathBuf;
use std::process::ExitCode;

use saor_streamer::gguf_embed::{rewrite_embedded, BlockReplacement};
use saor_streamer::gguf_sparse::{read_sparse_gguf, SparseBlock};
use serde_json::json;

/// Maneja `saor-engine embed --src <gguf> --dst <out> --block <t> --sparse <gguf> ...`.
pub fn cmd(args: &[String]) -> ExitCode {
    let mut src: Option<PathBuf> = None;
    let mut dst: Option<PathBuf> = None;
    let mut pairs: Vec<(String, PathBuf)> = Vec::new();
    let mut pending_tensor: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--src" => {
                i += 1;
                src = args.get(i).map(PathBuf::from);
            }
            "--dst" => {
                i += 1;
                dst = args.get(i).map(PathBuf::from);
            }
            "--block" => {
                i += 1;
                pending_tensor = args.get(i).cloned();
            }
            "--sparse" => {
                i += 1;
                let path = args.get(i).map(PathBuf::from);
                match (pending_tensor.take(), path) {
                    (Some(t), Some(s)) => pairs.push((t, s)),
                    _ => {
                        eprintln!("embed: --sparse requiere un --block previo y una ruta");
                        return ExitCode::from(2);
                    }
                }
            }
            other => {
                eprintln!("embed: argumento desconocido '{other}'");
                return ExitCode::from(2);
            }
        }
        i += 1;
    }

    let src = match src {
        Some(p) => p,
        None => {
            eprintln!("embed: falta --src <gguf original>");
            return ExitCode::from(2);
        }
    };
    let dst = match dst {
        Some(p) => p,
        None => {
            eprintln!("embed: falta --dst <gguf resultado>");
            return ExitCode::from(2);
        }
    };
    if pairs.is_empty() {
        eprintln!("embed: falta al menos un par --block <t> --sparse <gguf>");
        return ExitCode::from(2);
    }

    // Cargar cada bloque disperso (formato de `consolidate`).
    let mut replacements = Vec::with_capacity(pairs.len());
    for (tensor, sparse_path) in &pairs {
        let block: SparseBlock = match read_sparse_gguf(sparse_path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("embed: no se pudo leer {sparse_path:?}: {e}");
                return ExitCode::FAILURE;
            }
        };
        if block.d_in * block.d_out == 0 {
            eprintln!("embed: bloque disperso inválido en {sparse_path:?}");
            return ExitCode::FAILURE;
        }
        replacements.push(BlockReplacement {
            tensor: tensor.clone(),
            block,
            weights_file: None,
        });
    }

    match rewrite_embedded(&src, &dst, &replacements) {
        Ok(report) => {
            // Verificación: re-leer cada bloque embebido y comparar con el original.
            let mut verified = Vec::with_capacity(replacements.len());
            let mut all_ok = true;
            for r in &replacements {
                let back = match saor_streamer::gguf_embed::read_embedded_block(&dst, &r.tensor) {
                    Ok(Some(b)) => b,
                    Ok(None) => {
                        all_ok = false;
                        continue;
                    }
                    Err(e) => {
                        eprintln!("embed: verificación falló para {}: {e}", r.tensor);
                        all_ok = false;
                        continue;
                    }
                };
                let ok = back == r.block;
                if !ok {
                    eprintln!("embed: verificación falló para {}", r.tensor);
                }
                all_ok &= ok;
                verified.push(ok);
            }
            let out = json!({
                "ok": all_ok,
                "src": src.display().to_string(),
                "dst": dst.display().to_string(),
                "tensor_count": report.tensor_count,
                "replaced": report.replaced,
                "kept_bytes": report.kept_bytes,
                "sparse_bytes": report.sparse_bytes,
                "total_bytes": report.total_bytes,
                "verified": verified,
            });
            println!("{}", serde_json::to_string_pretty(&out).expect("json"));
            if all_ok {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(e) => {
            eprintln!("embed: {e}");
            ExitCode::FAILURE
        }
    }
}
