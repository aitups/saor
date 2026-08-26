//! Subcomando `kernels-run`: ejecuta el pipeline OpenCL (CPPN decode, SpMM,
//! Gram/CKA) sobre datos sintéticos deterministas, lo valida contra la
//! referencia de `saor-domain` y escribe un reporte JSON para que
//! `validate_opencl.py` lo cruce con la referencia NumPy.

use std::path::Path;

use nalgebra::DMatrix;
use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};
use saor_opencl::compute::ClEngine;
use serde_json::json;

/// Parámetros del pipeline sintético (por defecto; sobreescribibles vía CLI).
pub struct KernelRunParams {
    pub d_in: usize,
    pub d_out: usize,
    pub batch: usize,
    pub tau: f32,
    pub seed: u64,
}

impl Default for KernelRunParams {
    fn default() -> Self {
        Self {
            d_in: 64,
            d_out: 48,
            batch: 32,
            tau: 0.42,
            seed: 42,
        }
    }
}

fn random_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..len)
        .map(|_| (rng.next_u32() as f32 / u32::MAX as f32 - 0.5) * 2.0)
        .collect()
}

/// `a[m x k] * b^T[n x k]` -> `[m x n]`.
fn matmul_t(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[j * k + kk];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// Matmul CSR en host (misma semántica que el kernel `spmm_csr`).
fn matmul_csr_host(
    x: &[f32],
    row_ptr: &[i32],
    col_idx: &[i32],
    vals: &[f32],
    d_in: usize,
    d_out: usize,
) -> Vec<f32> {
    let batch = x.len() / d_in;
    let mut out = vec![0f32; batch * d_out];
    for b in 0..batch {
        for j in 0..d_out {
            let mut acc = 0.0f32;
            for k in row_ptr[j] as usize..row_ptr[j + 1] as usize {
                acc += x[b * d_in + col_idx[k] as usize] * vals[k];
            }
            out[b * d_out + j] = acc;
        }
    }
    out
}

fn dmatrix(v: &[f32], rows: usize, cols: usize) -> DMatrix<f32> {
    DMatrix::from_row_slice(rows, cols, v)
}

/// Ejecuta el pipeline y devuelve el reporte JSON.
pub fn run(params: &KernelRunParams) -> Result<serde_json::Value, String> {
    let mut engine = ClEngine::init()?;
    let device_name = engine.device_name();
    engine.prepare()?;

    let genome = saor_domain::cppn::CppnGenome::random_with(params.seed, 0.5);
    let flat = genome.flatten();
    let x = random_vec(params.batch * params.d_in, params.seed + 1);
    let w0 = random_vec(params.d_out * params.d_in, params.seed + 2); // profesor denso

    // --- Kernels OpenCL ---
    let (w_dense, adjacency, active) =
        engine.cppn_decode(&flat, params.d_in, params.d_out, params.tau, 0, 1)?;
    let y_dense = engine.spmm_dense(&x, &w_dense, params.d_in, params.d_out)?;

    let topo = saor_domain::topology::instantiate(&genome, params.d_in, params.d_out, params.tau);
    let (row_ptr, col_idx, vals) = topo.to_csr(params.d_in, params.d_out);
    let y_csr = engine.spmm_csr(&x, &row_ptr, &col_idx, &vals, params.d_in, params.d_out)?;

    // --- Referencias en host (saor-domain, espejo de NumPy) ---
    let w_ref = topo.dense_row_major(params.d_in, params.d_out);
    let y_ref = matmul_t(&x, &w_ref, params.batch, params.d_in, params.d_out);
    let y_csr_host = matmul_csr_host(&x, &row_ptr, &col_idx, &vals, params.d_in, params.d_out);

    let err_w = saor_opencl::compute::max_abs_diff(&w_ref, &w_dense);
    let err_y_dense = saor_opencl::compute::max_abs_diff(&y_ref, &y_dense);
    let err_y_csr = saor_opencl::compute::max_abs_diff(&y_ref, &y_csr);
    // Diagnóstico: el CSR en host debe coincidir con la referencia densa, y el
    // kernel CSR debe coincidir con el CSR en host.
    let err_csr_host_vs_ref = saor_opencl::compute::max_abs_diff(&y_ref, &y_csr_host);
    let err_csr_kernel_vs_host = saor_opencl::compute::max_abs_diff(&y_csr_host, &y_csr);

    // Comparación del bit-tensor de adyacencia (debe coincidir exactamente).
    let adj_ref = &topo.adjacency_bits;
    let adj_diff = adjacency
        .iter()
        .zip(adj_ref.iter())
        .filter(|(a, b)| a != b)
        .count();
    let active_ref = topo.active_connections();
    let active_match = active as usize == active_ref;

    // --- Decode multi-capa (Vía B): capa intermedia con y_layer != 0 y tau bajo
    // (el genoma `random_with` sesga l_ij muy por debajo de 0.42, dejando la
    // topología vacía; tau=0.05 la vuelve no vacía). El kernel deriva y_layer
    // internamente desde (layer, n_layers); la referencia usa `layer_coord` +
    // `instantiate_layer` (deben coincidir bit a bit).
    let (n_layer_test, n_layers_test) = (7usize, 30usize);
    let tau_b = 0.30f32;
    let (adj_b, active_b) = engine.cppn_decode_adjacency(
        &flat,
        params.d_in,
        params.d_out,
        tau_b,
        n_layer_test,
        n_layers_test,
    )?;
    let topo_b = saor_domain::topology::instantiate_layer(
        &genome,
        params.d_in,
        params.d_out,
        tau_b,
        saor_domain::cppn::layer_coord(n_layer_test, n_layers_test),
    );
    let adj_diff_b = adj_b
        .iter()
        .zip(topo_b.adjacency_bits.iter())
        .filter(|(a, b)| a != b)
        .count();
    let active_b_match = active_b as usize == topo_b.active_connections();

    // --- Gram/CKA ---
    let h0 = matmul_t(&x, &w0, params.batch, params.d_in, params.d_out); // profesor
    let h1 = y_dense.clone(); // candidato
    let k0 = engine.gram(&h0, params.d_out, params.batch)?;
    let k1 = engine.gram(&h1, params.d_out, params.batch)?;
    let cka = saor_domain::cka::centered_cka(
        &dmatrix(&k0, params.batch, params.batch),
        &dmatrix(&k1, params.batch, params.batch),
    );

    let sparsity = 1.0 - active as f32 / (params.d_in * params.d_out) as f32;
    let ok = err_w < 1e-4
        && err_y_dense < 1e-3
        && err_y_csr < 1e-3
        && adj_diff == 0
        && active_match
        && adj_diff_b == 0
        && active_b_match
        && cka.is_finite();

    Ok(json!({
        "ok": ok,
        "device": device_name,
        "params": {
            "d_in": params.d_in, "d_out": params.d_out,
            "batch": params.batch, "tau": params.tau, "seed": params.seed,
        },
        "metrics": {
            "active": active,
            "active_ref": active_ref,
            "active_match": active_match,
            "adj_bytes_diff": adj_diff,
            "sparsity": sparsity,
            "max_abs_err_w": err_w,
            "max_abs_err_y_dense": err_y_dense,
            "max_abs_err_y_csr": err_y_csr,
            "err_csr_host_vs_ref": err_csr_host_vs_ref,
            "err_csr_kernel_vs_host": err_csr_kernel_vs_host,
            "csr_nnz": vals.len(),
            "active_layer": active_b,
            "active_layer_ref": topo_b.active_connections(),
            "active_layer_match": active_b_match,
            "adj_bytes_diff_layer": adj_diff_b,
            "cka": cka,
        },
        // Datos crudos para la validación cruzada con NumPy.
        "data": {
            "genome": flat,
            "x": x,
            "w_dense": w_dense,
            "adjacency": adjacency,
            "y_dense": y_dense,
            "y_csr": y_csr,
            "gram_h0": k0,
            "gram_h1": k1,
        }
    }))
}

/// Maneja `saor-engine kernels-run [--out <path>]`.
pub fn cmd(args: &[String]) -> std::process::ExitCode {
    let mut out_path: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                i += 1;
                out_path = args.get(i).cloned();
            }
            other => {
                eprintln!("kernels-run: argumento desconocido '{other}'");
                return std::process::ExitCode::from(2);
            }
        }
        i += 1;
    }

    let params = KernelRunParams::default();
    match run(&params) {
        Ok(report) => {
            if let Some(path) = out_path {
                let path = Path::new(&path);
                if let Err(e) = std::fs::write(path, serde_json::to_string_pretty(&report).expect("json")) {
                    eprintln!("kernels-run: no se pudo escribir '{path:?}': {e}");
                    return std::process::ExitCode::FAILURE;
                }
                println!("reporte escrito en {}", path.display());
            }
            println!("{}", serde_json::to_string(&report).expect("json"));
            if report["ok"].as_bool().unwrap_or(false) {
                std::process::ExitCode::SUCCESS
            } else {
                std::process::ExitCode::FAILURE
            }
        }
        Err(e) => {
            eprintln!("kernels-run: {e}");
            println!("{}", json!({ "ok": false, "error": e }).to_string());
            std::process::ExitCode::FAILURE
        }
    }
}
