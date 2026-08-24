//! CMA-ES en subespacio activo con reconstrucción sin estado (QES).
//!
//! Fase 1 del subespacio activo: la diagonal de la Fisher empírica
//! (`Var(H0)`) identifica los canales calientes; CMA-ES congela ~99% de los
//! pesos de la CPPN y evoluciona solo `z ∈ R^d` (`d ≈ 100–500`). La
//! reconstrucción sin estado guarda únicamente la semilla entera y la recompensa
//! por individuo; las perturbaciones se regeneran deterministamente.
//!
//! Implementación canónica de Hansen (rank-one + rank-mu, caminos de evolución
//! `ps`/`pc`) en Rust puro.

use nalgebra::{DMatrix, DVector};
use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};

/// Parámetros del optimizador CMA-ES.
#[derive(Debug, Clone)]
pub struct CmaEsParams {
    /// Dimensión del subespacio activo (`d`).
    pub dim: usize,
    /// Tamaño de población.
    pub lambda: usize,
    /// Número de padres élite.
    pub mu: usize,
    /// Paso inicial (sigma).
    pub sigma0: f32,
    /// Semilla determinista del run.
    pub seed: u64,
}

impl CmaEsParams {
    /// Configuración por defecto siguiendo los valores canónicos de CMA-ES.
    pub fn new(dim: usize, seed: u64) -> Self {
        let lambda = 4 + (3.0 * (dim as f32).ln()) as usize;
        Self {
            dim,
            lambda,
            mu: lambda / 2,
            sigma0: 0.3,
            seed,
        }
    }
}

/// Una población reconstruida a partir de una semilla (stateless).
pub struct Population {
    /// Semilla entera que regenera las perturbaciones al vuelo.
    pub seed: u64,
    /// Perturbaciones `z ~ N(0, I)` de la población `[dim x lambda]`.
    pub perturbations: DMatrix<f32>,
    /// Puntos candidatos `m + sigma * L z`.
    pub candidates: DMatrix<f32>,
}

/// Estado del optimizador (método de Hansen).
#[derive(Debug, Clone)]
pub struct CmaEsState {
    /// Media actual `m`.
    pub mean: DVector<f32>,
    /// Paso global.
    pub sigma: f32,
    /// Covarianza `C`.
    pub covariance: DMatrix<f32>,
    /// Camino de evolución de C.
    pub pc: DVector<f32>,
    /// Camino de evolución de sigma.
    pub ps: DVector<f32>,
    /// Generación actual.
    pub generation: usize,
}

/// Pesos logarítmicos normalizados (recombinación) para los `mu` mejores.
pub fn default_weights(mu: usize) -> Vec<f32> {
    let raw: Vec<f32> = (0..mu)
        .map(|i| ((mu as f32 + 1.0).ln() - (i as f32 + 1.0).ln()).max(1e-12))
        .collect();
    let sum: f32 = raw.iter().sum();
    raw.into_iter().map(|w| w / sum).collect()
}

impl CmaEsState {
    /// Estado inicial: media `mean0`, covarianza identidad, `sigma = sigma0`.
    pub fn init(params: &CmaEsParams, mean0: DVector<f32>) -> Self {
        let dim = mean0.len();
        Self {
            mean: mean0,
            sigma: params.sigma0,
            covariance: DMatrix::<f32>::identity(dim, dim),
            pc: DVector::zeros(dim),
            ps: DVector::zeros(dim),
            generation: 0,
        }
    }

    /// Regenera la población sin estado a partir de una semilla entera.
    pub fn spawn_population(&self, params: &CmaEsParams, seed: u64) -> Population {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let dim = self.mean.len();
        // C = L L^T; muestrear N(m, sigma^2 C) = m + sigma * L z.
        let chol = self
            .covariance
            .clone()
            .cholesky()
            .expect("covarianza debe ser PD");
        let l = chol.l();
        let mut perturbations = DMatrix::<f32>::zeros(dim, params.lambda);
        let mut candidates = DMatrix::<f32>::zeros(dim, params.lambda);
        for col in 0..params.lambda {
            for row in 0..dim {
                perturbations[(row, col)] = gaussian(&mut rng);
            }
            let z = perturbations.column(col);
            let delta = &l * z;
            for row in 0..dim {
                candidates[(row, col)] = self.mean[row] + self.sigma * delta[row];
            }
        }
        Population {
            seed,
            perturbations,
            candidates,
        }
    }
}

/// Normal gaussiana estándar por Box-Muller con rng determinista.
fn gaussian(rng: &mut ChaCha8Rng) -> f32 {
    let u1 = rng.next_u32() as f32 / u32::MAX as f32;
    let u2 = rng.next_u32() as f32 / u32::MAX as f32;
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

impl CmaEsState {
    /// Actualización CMA-ES (rank-one + rank-mu) dados los índices élite
    /// ordenados por fitness ascendente (mejor primero).
    pub fn update(&mut self, params: &CmaEsParams, pop: &Population, elite: &[usize]) {
        let n = self.mean.len();
        let weights = default_weights(params.mu);
        let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f32>();
        let cc = 4.0 / (n as f32 + 4.0);
        let cs = (mu_eff + 2.0) / (n as f32 + mu_eff + 5.0);
        let c1 = 2.0 / ((n as f32 + 1.3).powi(2) + mu_eff);
        let cmu =
            (2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n as f32 + 2.0).powi(2) + mu_eff)).min(1.0 - c1);
        let ds = 1.0 + 2.0 * ((mu_eff - 1.0) / (n as f32 + 1.0)).sqrt().max(0.0) + cs;
        let chi_n =
            (n as f32).sqrt() * (1.0 - 1.0 / (4.0 * n as f32) + 1.0 / (21.0 * n as f32 * n as f32));

        // Pasos y_i = (x_i - m)/sigma con la media ANTIGUA.
        let mut weighted_step = DVector::<f32>::zeros(n);
        let mut ys: Vec<DVector<f32>> = Vec::with_capacity(params.mu);
        for (rank, &idx) in elite.iter().enumerate() {
            let y = (pop.candidates.column(idx).into_owned() - &self.mean) / self.sigma;
            weighted_step += weights[rank] * &y;
            ys.push(y);
        }

        // Camino de sigma y adaptación del paso.
        self.ps = (1.0 - cs) * &self.ps + (cs * (2.0 - cs) * mu_eff).sqrt() * &weighted_step;
        let ps_norm = self.ps.norm();
        self.sigma *= ((cs / ds) * (ps_norm / chi_n - 1.0)).exp();

        // Camino de C (con mitigación h_sigma para la fase de arranque).
        let exp_len =
            (1.0 - (1.0 - cs).powi(2 * (self.generation + 1) as i32)).sqrt();
        let h_sigma = if ps_norm / exp_len < (1.4 + 2.0 / (n as f32 + 1.0)) * chi_n {
            1.0
        } else {
            0.0
        };
        self.pc =
            (1.0 - cc) * &self.pc + h_sigma * (cc * (2.0 - cc) * mu_eff).sqrt() * &weighted_step;

        // Covarianza: rank-one + rank-mu.
        let rank_one = &self.pc * self.pc.transpose();
        let mut rank_mu = DMatrix::<f32>::zeros(n, n);
        for (rank, y) in ys.iter().enumerate() {
            rank_mu += weights[rank] * (y * y.transpose());
        }
        let delta_h = (1.0 - h_sigma) * cc * (2.0 - cc);
        self.covariance = (1.0 - c1 - cmu) * &self.covariance
            + c1 * (rank_one + delta_h * &self.covariance)
            + cmu * rank_mu;

        // Nueva media.
        self.mean += self.sigma * &weighted_step;
        self.generation += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poblacion_determinista_por_semilla() {
        let params = CmaEsParams::new(8, 42);
        let state = CmaEsState::init(&params, DVector::zeros(8));
        let p1 = state.spawn_population(&params, 1234);
        let p2 = state.spawn_population(&params, 1234);
        assert_eq!(
            p1.perturbations, p2.perturbations,
            "misma semilla => misma población"
        );
        let p3 = state.spawn_population(&params, 9999);
        assert_ne!(
            p1.perturbations, p3.perturbations,
            "distinta semilla => distinta población"
        );
    }

    #[test]
    fn minimiza_cuadratica_sintetica() {
        let params = CmaEsParams::new(6, 7);
        let mut state = CmaEsState::init(&params, DVector::from_element(6, 10.0));
        let mut best_ever = f32::INFINITY;
        for gen in 0..80 {
            let pop = state.spawn_population(&params, params.seed + gen as u64);
            let mut scores: Vec<(usize, f32)> = (0..params.lambda)
                .map(|i| {
                    let c = pop.candidates.column(i);
                    let v = (0..6).map(|k| (c[k] - 1.0).powi(2)).sum::<f32>() * 0.5;
                    (i, v)
                })
                .collect();
            scores.sort_by(|a, b| a.1.total_cmp(&b.1));
            best_ever = best_ever.min(scores[0].1);
            let elite: Vec<usize> = scores.iter().take(params.mu).map(|(i, _)| *i).collect();
            state.update(&params, &pop, &elite);
        }
        assert!(
            best_ever < 1e-2,
            "CMA-ES debe converger cerca del mínimo, best={best_ever}"
        );
    }

    #[test]
    fn pesos_logaritmicos_son_positivos_y_normalizados() {
        let w = default_weights(8);
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(w.windows(2).all(|p| p[0] >= p[1]));
    }
}

