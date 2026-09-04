//! Subcomando `decode-v7`: decodifica un genoma v7 (CPPN de pesos + geometría
//! aprendida) para un bloque en la GPU con `ClEngine::cppn_decode_v7` y vuelca
//! la matriz densa enmascarada + el bit-tensor para el harness de paridad
//! (`python/saor_orchestrator/validate_opencl_v7.py`, referencia escalar f32).
//!
//!   saor-engine decode-v7 --genome <genome.bin> --d-in N --d-out N
//!                         --layer L --n-layers N --mode hi|ih --tau F
//!                         --out <dir>
//!
//! Salida: `w.bin` (d_in*d_out f32 i-mayor), `adj.bin` (bit-tensor u8) y
//! `meta.json`.

use std::path::PathBuf;
use std::process::ExitCode;

use saor_opencl::compute::ClEngine;

fn read_f32_bin(path: &std::path::Path) -> Result<Vec<f32>, String> {
    let raw = std::fs::read(path).map_err(|e| format!("leer {}: {e}", path.display()))?;
    if raw.len() % 4 != 0 {
        return Err(format!("{}: tamaño no múltiplo de 4", path.display()));
    }
    Ok(raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Maneja `saor-engine decode-v7 ...`.
pub fn cmd(args: &[String]) -> ExitCode {
    let mut genome: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut d_in = 0usize;
    let mut d_out = 0usize;
    let mut layer = 0usize;
    let mut n_layers = 30usize;
    let mut mode = 0usize;
    let mut tau = 0.5f32;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--genome" => {
                i += 1;
                genome = args.get(i).map(PathBuf::from);
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
            "--layer" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    layer = v;
                }
            }
            "--n-layers" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    n_layers = v;
                }
            }
            "--mode" => {
                i += 1;
                mode = match args.get(i).map(String::as_str) {
                    Some("ih") => 1,
                    _ => 0, // "hi"
                };
            }
            "--tau" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    tau = v;
                }
            }
            other => {
                eprintln!("decode-v7: argumento desconocido '{other}'");
                return ExitCode::from(2);
            }
        }
        i += 1;
    }
    let genome = match &genome {
        Some(p) => p.clone(),
        None => {
            eprintln!("decode-v7: falta --genome <genome.bin>");
            return ExitCode::FAILURE;
        }
    };
    let out_dir = match &out_dir {
        Some(p) => p.clone(),
        None => {
            eprintln!("decode-v7: falta --out <dir>");
            return ExitCode::FAILURE;
        }
    };
    if d_in == 0 || d_out == 0 {
        eprintln!("decode-v7: falta --d-in/--d-out");
        return ExitCode::FAILURE;
    }

    let flat = match read_f32_bin(&genome) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("decode-v7: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut engine = match ClEngine::init() {
        Ok(mut e) => match e.prepare() {
            Ok(()) => e,
            Err(err) => {
                eprintln!("decode-v7: prepare: {err}");
                return ExitCode::FAILURE;
            }
        },
        Err(err) => {
            eprintln!("decode-v7: no hay dispositivo OpenCL: {err}");
            return ExitCode::FAILURE;
        }
    };

    let t0 = std::time::Instant::now();
    let (w_dense, adj, active) = match engine.cppn_decode_v7(&flat, d_in, d_out, tau, layer, n_layers, mode) {
        Ok(r) => r,
        Err(err) => {
            eprintln!("decode-v7: {err}");
            return ExitCode::FAILURE;
        }
    };
    let decode_ms = t0.elapsed().as_millis();

    let _ = std::fs::create_dir_all(&out_dir);
    let mut wb = Vec::with_capacity(w_dense.len() * 4);
    for w in &w_dense {
        wb.extend_from_slice(&w.to_le_bytes());
    }
    if let Err(e) = std::fs::write(out_dir.join("w.bin"), wb) {
        eprintln!("decode-v7: escribir w.bin: {e}");
        return ExitCode::FAILURE;
    }
    if let Err(e) = std::fs::write(out_dir.join("adj.bin"), &adj) {
        eprintln!("decode-v7: escribir adj.bin: {e}");
        return ExitCode::FAILURE;
    }
    let meta = serde_json::json!({
        "ok": true,
        "genome_len": flat.len(),
        "d_in": d_in,
        "d_out": d_out,
        "layer": layer,
        "n_layers": n_layers,
        "mode": if mode == 0 { "hi" } else { "ih" },
        "tau": tau,
        "active": active,
        "total": d_in * d_out,
        "decode_ms": decode_ms,
        "device": engine.device_name(),
    });
    if let Err(e) = std::fs::write(
        out_dir.join("meta.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    ) {
        eprintln!("decode-v7: escribir meta.json: {e}");
        return ExitCode::FAILURE;
    }
    println!(
        "{{\"ok\":true,\"device\":{},\"active\":{},\"total\":{},\"decode_ms\":{}}}",
        serde_json::Value::String(engine.device_name()),
        active,
        d_in * d_out,
        decode_ms
    );
    ExitCode::SUCCESS
}
