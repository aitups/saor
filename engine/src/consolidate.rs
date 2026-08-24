//! Subcomando `consolidate`: consolida el mejor candidato evolucionado en un
//! **GGUF disperso** (sin densificar, decisión D4 — bit-tensor
//! `ffn_dag_adjacency` + pesos activos) y reporta el contrato de Fase 2
//! (D_arch, fidelidad CKA, no-dormancia estructural).

use std::path::PathBuf;
use std::process::ExitCode;

use saor_streamer::gguf_sparse::{write_sparse_gguf, SparseBlock};
use serde_json::json;

/// Maneja `saor-engine consolidate --out-gguf <ruta> [--gens N] [--seed N]`.
pub fn cmd(args: &[String]) -> ExitCode {
    let mut params = crate::evolve::EvolveParams::default();
    let mut out_gguf: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--gens" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    params.generations = v;
                }
            }
            "--d-in" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    params.d_in = v;
                }
            }
            "--d-out" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    params.d_out = v;
                }
            }
            "--batch" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    params.batch = v;
                }
            }
            "--seed" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    params.seed = v;
                }
            }
            "--tau0" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    params.tau0 = v;
                }
            }
            "--teacher-copy" => {
                params.teacher_copy = true;
            }
            "--warm" => {
                i += 1;
                params.warm_genome = args.get(i).map(PathBuf::from);
            }
            "--teacher-w" => {
                i += 1;
                params.teacher_w = args.get(i).cloned();
            }
            "--teacher-x" => {
                i += 1;
                params.teacher_x = args.get(i).cloned();
            }
            "--out-gguf" => {
                i += 1;
                out_gguf = args.get(i).cloned();
            }
            other => {
                eprintln!("consolidate: argumento desconocido '{other}'");
                return ExitCode::from(2);
            }
        }
        i += 1;
    }

    let path = match out_gguf {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("consolidate: falta --out-gguf <ruta>");
            return ExitCode::from(2);
        }
    };

    match crate::evolve::run(&params) {
        Ok(outcome) => {
            let active_connections = outcome
                .best_adjacency
                .iter()
                .map(|b| b.count_ones() as usize)
                .sum::<usize>();
            let d_arch = 1.0 - active_connections as f32 / (params.d_in * params.d_out) as f32;
            let block = SparseBlock {
                d_in: params.d_in,
                d_out: params.d_out,
                tau: outcome.best_tau,
                genome: outcome.best_flat,
                adjacency: outcome.best_adjacency,
                weights: outcome.best_weights,
            };
            if let Err(e) = write_sparse_gguf(&path, &block) {
                eprintln!("consolidate: no se pudo escribir el GGUF: {e}");
                return ExitCode::FAILURE;
            }
            let file_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

            let report = json!({
                "ok": true,
                "gguf": path.display().to_string(),
                "gguf_bytes": file_bytes,
                "d_in": params.d_in,
                "d_out": params.d_out,
                "tau": outcome.best_tau,
                "active_connections": active_connections,
                "d_arch": d_arch,
                "best_cka": outcome.best.cka,        // fidelidad funcional
                "best_fitness": outcome.best.fitness,
            });
            println!("{}", serde_json::to_string_pretty(&report).expect("json"));
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("consolidate: {e}");
            ExitCode::FAILURE
        }
    }
}
