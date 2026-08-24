//! Subcomando `evolve`: loop evolutivo integrado (Fase 4).
//!
//! Pipeline end-to-end sobre datos sintéticos:
//! `genoma -> topología (CPPN + τ) -> SpMM (OpenCL) -> Gram/CKA -> CMA-ES`.
//!
//! * **Subespacio activo (Fase 1):** solo se evoluciona la cola del genoma
//!   aplanado (capa de salida de la CPPN — el mayor control sobre la topología)
//!   más una coordenada `τ` (vía sigmoide). El resto del genoma se congela.
//! * **Reconstrucción sin estado (QES):** cada generación regenera la población
//!   desde la semilla entera `seed + gen`; no se guardan tensores de
//!   perturbaciones entre generaciones.
//! * **Fitness:** `CKA(H0, H1) + α * Sparsity(A1)` con el profesor `H0` fijo.

use std::process::ExitCode;

use nalgebra::DVector;
use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};
use saor_domain::cka::centered_cka;
use saor_domain::cmaes::{CmaEsParams, CmaEsState};
use saor_domain::cppn::CppnGenome;
use saor_domain::topology::instantiate;
use saor_opencl::compute::ClEngine;
use serde_json::json;

/// Parámetros de la corrida evolutiva sintética.
pub struct EvolveParams {
    /// Dimensión de entrada del bloque.
    pub d_in: usize,
    /// Dimensión de salida del bloque.
    pub d_out: usize,
    /// Tamaño de lote de calibración.
    pub batch: usize,
    /// Semilla determinista de toda la corrida.
    pub seed: u64,
    /// Número de generaciones.
    pub generations: usize,
    /// Dimensión del subespacio activo (incluye τ).
    pub subspace: usize,
    /// Peso de la esparcidad en el fitness.
    pub alpha: f32,
    /// τ inicial.
    pub tau0: f32,
    /// Profesor real: archivo con W0 en f32 plano (d_out*d_in, fila-mayor).
    pub teacher_w: Option<String>,
    /// Profesor real: archivo con X de calibración en f32 plano (batch*d_in).
    pub teacher_x: Option<String>,
}

impl Default for EvolveParams {
    fn default() -> Self {
        Self {
            d_in: 64,
            d_out: 48,
            batch: 32,
            seed: 42,
            generations: 30,
            subspace: 120, // 119 coordenadas del genoma + τ
            alpha: 0.3,
            tau0: 0.42,
            teacher_w: None,
            teacher_x: None,
        }
    }
}

/// Lee un archivo de f32 planos (little-endian).
pub fn read_f32_bin(path: &str) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("leer '{path}': {e}"))?;
    if bytes.len() % 4 != 0 {
        return Err(format!("'{path}': longitud no múltiplo de 4 (f32)"));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn random_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..len)
        .map(|_| (rng.next_u32() as f32 / u32::MAX as f32 - 0.5) * 2.0)
        .collect()
}

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

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_inv(p: f32) -> f32 {
    (p / (1.0 - p)).ln()
}

fn dmatrix(v: &[f32], rows: usize, cols: usize) -> nalgebra::DMatrix<f32> {
    nalgebra::DMatrix::from_row_slice(rows, cols, v)
}

/// Construye una `Topology` compacta a partir de la salida del kernel GPU
/// `cppn_decode` (matriz densa enmascarada + bit-tensor). Reemplaza el
/// `instantiate` CPU (que re-evaluaba la CPPN 885K–201M veces por candidato).
fn topology_from_dense(
    w_dense: &[f32],
    adj: &[u8],
    d_in: usize,
    d_out: usize,
) -> saor_domain::topology::Topology {
    let mut weights = Vec::new();
    for i in 0..d_in {
        for j in 0..d_out {
            let conn = i * d_out + j;
            if adj[conn / 8] & (1 << (conn % 8)) != 0 {
                weights.push(w_dense[j * d_in + i]);
            }
        }
    }
    saor_domain::topology::Topology {
        adjacency_bits: adj.to_vec(),
        total_connections: d_in * d_out,
        weights,
    }
}

/// Un candidato evaluado.
#[derive(Clone, Copy)]
pub struct Scored {
    /// Índice en la población.
    pub col: usize,
    /// Fitness `CKA + α * sparsity`.
    pub fitness: f32,
    /// CKA contra el profesor.
    pub cka: f32,
    /// Esparcidad `D_arch`.
    pub sparsity: f32,
    /// Umbral τ del candidato.
    pub tau: f32,
}

/// Resultado de una corrida evolutiva (reporte + mejor candidato).
pub struct EvolveOutcome {
    /// Reporte JSON completo.
    pub report: serde_json::Value,
    /// Genoma aplanado del mejor candidato.
    pub best_flat: Vec<f32>,
    /// τ del mejor candidato.
    pub best_tau: f32,
    /// Métricas del mejor candidato.
    pub best: Scored,
}

/// Ejecuta el loop evolutivo y devuelve el reporte JSON.
pub fn run(params: &EvolveParams) -> Result<EvolveOutcome, String> {
    let mut engine = ClEngine::init()?;
    let device_name = engine.device_name();
    engine.prepare()?;

    // --- Profesor (H0) y datos de calibración ---
    let g_base = CppnGenome::random_with(params.seed, 0.5);
    let flat = g_base.flatten();
    let genome_len = flat.len();
    // Profesor real (archivos f32) o sintético (aleatorio denso).
    let (x, w0) = match (&params.teacher_w, &params.teacher_x) {
        (Some(tw), Some(tx)) => {
            let w0 = read_f32_bin(tw)?;
            let x = read_f32_bin(tx)?;
            if w0.len() != params.d_out * params.d_in {
                return Err(format!(
                    "teacher_w: esperaba {} f32 (d_out*d_in), hay {}",
                    params.d_out * params.d_in,
                    w0.len()
                ));
            }
            if x.len() != params.batch * params.d_in {
                return Err(format!(
                    "teacher_x: esperaba {} f32 (batch*d_in), hay {}",
                    params.batch * params.d_in,
                    x.len()
                ));
            }
            (x, w0)
        }
        _ => {
            let x = random_vec(params.batch * params.d_in, params.seed + 1);
            let w0 = instantiate(&g_base, params.d_in, params.d_out, 0.0)
                .dense_row_major(params.d_in, params.d_out);
            (x, w0)
        }
    };
    let h0 = matmul_t(&x, &w0, params.batch, params.d_in, params.d_out);
    let k0 = engine.gram(&h0, params.d_out, params.batch)?;
    let k0m = dmatrix(&k0, params.batch, params.batch);

    // --- Subespacio activo (Fase 1): cola del genoma + τ ---
    let dim = params.subspace;
    debug_assert!(dim >= 2 && dim - 1 <= genome_len);
    let tail_start = genome_len - (dim - 1);
    let cma_params = CmaEsParams::new(dim, params.seed);
    let mut mean0 = DVector::<f32>::zeros(dim);
    mean0[dim - 1] = sigmoid_inv(params.tau0);
    let mut state = CmaEsState::init(&cma_params, mean0);

    // --- Loop evolutivo ---
    let mut history: Vec<serde_json::Value> = Vec::new();
    let mut best_so_far = f32::NEG_INFINITY;
    let mut best_scored: Option<Scored> = None;
    let mut best_flat: Option<Vec<f32>> = None;

    for gen in 0..params.generations {
        // Reconstrucción sin estado: población desde la semilla entera.
        let pop = state.spawn_population(&cma_params, params.seed + gen as u64);
        let t_gen = std::time::Instant::now();

        let mut scored: Vec<Scored> = Vec::with_capacity(cma_params.lambda);
        for col in 0..cma_params.lambda {
            let t_cand = std::time::Instant::now();
            let z = pop.candidates.column(col);
            // Genoma candidato: copia del base + perturbación residual en la cola.
            let mut f = flat.clone();
            for k in 0..(dim - 1) {
                f[tail_start + k] += z[k];
            }
            let tau = sigmoid(z[dim - 1]);
            // Decodificación GPU (reemplaza el instantiate CPU por candidato).
            let (w_dense, adj, _active) =
                engine.cppn_decode(&f, params.d_in, params.d_out, tau)?;
            let topo = topology_from_dense(&w_dense, &adj, params.d_in, params.d_out);
            let (rp, ci, vals) = topo.to_csr(params.d_in, params.d_out);
            let h1 = engine.spmm_csr(&x, &rp, &ci, &vals, params.d_in, params.d_out)?;
            let k1 = engine.gram(&h1, params.d_out, params.batch)?;
            let k1m = dmatrix(&k1, params.batch, params.batch);
            let cka = centered_cka(&k0m, &k1m);
            let sparsity = topo.sparsity();
            let fitness = cka + params.alpha * sparsity;
            scored.push(Scored {
                col,
                fitness,
                cka,
                sparsity,
                tau,
            });
            eprintln!(
                "[evolve] gen {gen} cand {col} t={:.2}s cka={cka:.3} sp={sparsity:.3}",
                t_cand.elapsed().as_secs_f32()
            );
        }

        // Selección élite (maximizar fitness) y actualización CMA-ES.
        scored.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
        let elite: Vec<usize> = scored.iter().take(cma_params.mu).map(|s| s.col).collect();
        state.update(&cma_params, &pop, &elite);

        let gen_best = scored[0];
        if gen_best.fitness > best_so_far {
            best_so_far = gen_best.fitness;
            best_scored = Some(gen_best);
            // Guardar el genoma del mejor candidato de la generación.
            let z = pop.candidates.column(gen_best.col);
            let mut f = flat.clone();
            for k in 0..(dim - 1) {
                f[tail_start + k] += z[k];
            }
            best_flat = Some(f);
        }
        let mean_fitness =
            scored.iter().map(|s| s.fitness as f64).sum::<f64>() / scored.len() as f64;
        history.push(json!({
            "gen": gen,
            "best_fitness": gen_best.fitness,
            "best_cka": gen_best.cka,
            "best_sparsity": gen_best.sparsity,
            "best_tau": gen_best.tau,
            "mean_fitness": mean_fitness,
            "best_so_far": best_so_far,
        }));
        eprintln!(
            "[evolve] gen {}/{} best_fit={:.4} cka={:.4} sp={:.4} mean={:.4} (gen {:.1}s)",
            gen + 1,
            params.generations,
            gen_best.fitness,
            gen_best.cka,
            gen_best.sparsity,
            mean_fitness,
            t_gen.elapsed().as_secs_f32()
        );
    }

    let best = best_scored.ok_or("sin candidatos evaluados")?;
    let best_flat = best_flat.ok_or("sin genoma del mejor candidato")?;
    let ok = best.fitness.is_finite() && best.cka.is_finite() && best.cka >= 0.0;

    let report = json!({
        "ok": ok,
        "device": device_name,
        "params": {
            "d_in": params.d_in, "d_out": params.d_out, "batch": params.batch,
            "seed": params.seed, "generations": params.generations,
            "subspace": params.subspace, "alpha": params.alpha, "tau0": params.tau0,
        },
        "result": {
            "best_fitness": best.fitness,
            "best_cka": best.cka,
            "best_sparsity": best.sparsity,
            "best_tau": best.tau,
            "best_so_far": best_so_far,
        },
        "history": history,
    });
    Ok(EvolveOutcome {
        report,
        best_flat,
        best_tau: best.tau,
        best,
    })
}

/// Maneja `saor-engine evolve [--gens N] [--subspace N] [--seed N] [--out <path>]`.
pub fn cmd(args: &[String]) -> ExitCode {
    let mut params = EvolveParams::default();
    let mut out_path: Option<String> = None;
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
            "--subspace" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    params.subspace = v;
                }
            }
            "--seed" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    params.seed = v;
                }
            }
            "--teacher-w" => {
                i += 1;
                params.teacher_w = args.get(i).cloned();
            }
            "--teacher-x" => {
                i += 1;
                params.teacher_x = args.get(i).cloned();
            }
            "--out" => {
                i += 1;
                out_path = args.get(i).cloned();
            }
            other => {
                eprintln!("evolve: argumento desconocido '{other}'");
                return ExitCode::from(2);
            }
        }
        i += 1;
    }

    match run(&params) {
        Ok(outcome) => {
            if let Some(path) = out_path {
                if let Err(e) = std::fs::write(
                    &path,
                    serde_json::to_string_pretty(&outcome.report).expect("json"),
                ) {
                    eprintln!("evolve: no se pudo escribir '{path}': {e}");
                    return ExitCode::FAILURE;
                }
                println!("reporte escrito en {path}");
            }
            println!("{}", serde_json::to_string(&outcome.report).expect("json"));
            if outcome.report["ok"].as_bool().unwrap_or(false) {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(e) => {
            eprintln!("evolve: {e}");
            ExitCode::FAILURE
        }
    }
}



