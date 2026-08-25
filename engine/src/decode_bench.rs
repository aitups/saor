//! Subcomando `decode-bench`: mide el tiempo del decodificador CPPN aislado
//! para un bloque dado (escalado Fase 7.6). Uso:
//! `saor-engine decode-bench --d-in 576 --d-out 1536 --trials 3`.

use std::process::ExitCode;
use std::time::Instant;

use saor_domain::cppn::CppnGenome;
use saor_opencl::compute::ClEngine;
use serde_json::json;

/// Maneja `saor-engine decode-bench [--d-in N] [--d-out N] [--trials N]`.
pub fn cmd(args: &[String]) -> ExitCode {
    let mut d_in = 576usize;
    let mut d_out = 1536usize;
    let mut trials = 3usize;
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
            "--trials" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    trials = v;
                }
            }
            other => {
                eprintln!("decode-bench: argumento desconocido '{other}'");
                return ExitCode::from(2);
            }
        }
        i += 1;
    }

    let mut engine = match ClEngine::init().and_then(|mut e| {
        e.prepare()?;
        Ok(e)
    }) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("decode-bench: {e}");
            return ExitCode::FAILURE;
        }
    };
    let genome = CppnGenome::random_with(42, 0.5).flatten();
    let total = d_in * d_out;

    let mut timings = Vec::with_capacity(trials);
    let mut active = 0u32;
    for t in 0..trials {
        let t0 = Instant::now();
        let res = engine.cppn_decode(&genome, d_in, d_out, 0.42, 0, 1);
        match res {
            Ok((_w, _adj, act)) => {
                active = act;
                let dt = t0.elapsed().as_secs_f32();
                timings.push(dt);
                eprintln!("[decode-bench] trial {t}: {dt:.3}s active={act}");
            }
            Err(e) => {
                eprintln!("decode-bench: trial {t} falló: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    let mean = timings.iter().sum::<f32>() / timings.len() as f32;
    let report = json!({
        "ok": true,
        "d_in": d_in,
        "d_out": d_out,
        "connections": total,
        "active": active,
        "trials": trials,
        "mean_s": mean,
        "connections_per_s": (total as f64 / mean as f64) as u64,
    });
    println!("{}", serde_json::to_string_pretty(&report).expect("json"));
    ExitCode::SUCCESS
}
