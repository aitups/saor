//! Subcomando `decode-bench`: mide el tiempo del decodificador CPPN aislado
//! para un bloque dado (escalado Fase 7.6). Uso:
//! `saor-engine decode-bench --d-in 576 --d-out 1536 --trials 3 [--batched 22]`.
//!
//! Con `--batched N` añade la puerta de aceptación de la **Fase 1** (tensorización
//! de la población): decodifica N genomas de forma individual y en un único
//! dispatch batcheado (`cppn_decode_adj_batched`) y verifica **bit-exactitud**.

use std::process::ExitCode;
use std::time::Instant;

use saor_domain::cppn::CppnGenome;
use saor_opencl::compute::ClEngine;
use serde_json::json;

/// Maneja `saor-engine decode-bench [--d-in N] [--d-out N] [--trials N] [--batched N]`.
pub fn cmd(args: &[String]) -> ExitCode {
    let mut d_in = 576usize;
    let mut d_out = 1536usize;
    let mut trials = 3usize;
    let mut batched = 0usize;
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
            "--batched" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    batched = v;
                }
            }
            other => {
                eprintln!("decode-bench: argumento desconocido '{other}'");
                return ExitCode::from(2);
            }
        }
        i += 1;
    }

    let engine = match ClEngine::init().and_then(|mut e| {
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
    let mut report = json!({
        "ok": true,
        "d_in": d_in,
        "d_out": d_out,
        "connections": total,
        "active": active,
        "trials": trials,
        "mean_s": mean,
        "connections_per_s": (total as f64 / mean as f64) as u64,
    });

    // Puerta de aceptación Fase 1: decode batcheado == decode individual (bit-exacto).
    if batched > 0 {
        let genomes: Vec<Vec<f32>> = (0..batched)
            .map(|c| CppnGenome::random_with(1000 + c as u64, 0.5).flatten())
            .collect();
        let flat: Vec<f32> = genomes.iter().flatten().copied().collect();
        let n_bytes = total.div_ceil(8);

        // Referencia individual.
        let t0 = Instant::now();
        let mut ref_adj = Vec::with_capacity(batched * n_bytes);
        let mut ref_active = Vec::with_capacity(batched);
        for g in &genomes {
            let (adj, act) = engine
                .cppn_decode_adjacency(g, d_in, d_out, 0.42, 0, 1)
                .map_err(|e| {
                    eprintln!("decode-bench: decode individual falló: {e}");
                })
                .unwrap();
            ref_adj.extend_from_slice(&adj);
            ref_active.push(act);
        }
        let ind_s = t0.elapsed().as_secs_f32();

        // Batcheado (un único dispatch por kernel).
        let t0 = Instant::now();
        let (bat_adj, bat_active) = engine
            .cppn_decode_adjacency_batched(&flat, batched, d_in, d_out, 0.42, 0, 1)
            .map_err(|e| {
                eprintln!("decode-bench: decode batcheado falló: {e}");
            })
            .unwrap();
        let bat_s = t0.elapsed().as_secs_f32();

        let adj_ok = bat_adj == ref_adj;
        let act_ok = bat_active == ref_active;
        let exact = adj_ok && act_ok;
        let ratio = bat_s / ind_s;
        eprintln!(
            "[decode-bench] batched N={batched}: {bat_s:.3}s vs individual {ind_s:.3}s \
             ({ratio:.2}×) adj_bit_exact={adj_ok} active_exact={act_ok}"
        );
        report["batched"] = json!({
            "n": batched,
            "individual_s": ind_s,
            "batched_s": bat_s,
            "speedup_vs_individual": bat_s / ind_s,
            "bit_exact": exact,
        });
        if !exact {
            eprintln!("decode-bench: FALLO de bit-exactitud en el decode batcheado");
            return ExitCode::FAILURE;
        }
    }

    println!("{}", serde_json::to_string_pretty(&report).expect("json"));
    ExitCode::SUCCESS
}
