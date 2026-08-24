"""CMA-ES en subespacio activo con reconstrucción sin estado.

Referencia NumPy de `saor_domain::cmaes` (método de Hansen: rank-one +
rank-mu, caminos de evolución `ps`/`pc`). La reconstrucción sin estado guarda
solo la semilla entera y la recompensa; la población se regenera al vuelo.
"""

from __future__ import annotations

import numpy as np


class CmaEsParams:
    """Parámetros del optimizador CMA-ES."""

    def __init__(self, dim: int, seed: int, sigma0: float = 0.3) -> None:
        self.dim = dim
        self.lambda_ = 4 + int(3.0 * np.log(dim))
        self.mu = self.lambda_ // 2
        self.sigma0 = float(sigma0)
        self.seed = int(seed)


def default_weights(mu: int) -> np.ndarray:
    """Pesos logarítmicos normalizados para los `mu` mejores."""
    raw = np.array([np.log(mu + 1) - np.log(i + 1) for i in range(mu)], np.float64)
    raw = np.maximum(raw, 1e-12)
    return raw / raw.sum()


class Population:
    """Población reconstruida a partir de una semilla (stateless)."""

    def __init__(self, seed: int, perturbations: np.ndarray, candidates: np.ndarray) -> None:
        self.seed = int(seed)
        self.perturbations = perturbations
        self.candidates = candidates


class CmaEsState:
    """Estado del optimizador (media, paso, covarianza y caminos)."""

    def __init__(self, params: CmaEsParams, mean0: np.ndarray) -> None:
        self.params = params
        dim = params.dim
        self.mean = np.asarray(mean0, np.float64).reshape(-1)
        self.sigma = params.sigma0
        self.covariance = np.eye(dim, dtype=np.float64)
        self.pc = np.zeros(dim, np.float64)
        self.ps = np.zeros(dim, np.float64)
        self.generation = 0

    def spawn_population(self, seed: int) -> Population:
        """Regenera la población N(m, sigma^2 C) de forma determinista."""
        rng = np.random.default_rng(seed)
        dim, lam = self.params.dim, self.params.lambda_
        z = rng.standard_normal((dim, lam))
        l_mat = np.linalg.cholesky(self.covariance)
        candidates = self.mean[:, None] + self.sigma * (l_mat @ z)
        return Population(seed, z, candidates)

    def update(self, pop: Population, elite: list[int]) -> None:
        """Actualización rank-one + rank-mu dados los índices élite (mejor primero)."""
        params = self.params
        n = params.dim
        weights = default_weights(params.mu)
        mu_eff = 1.0 / np.sum(weights**2)
        cc = 4.0 / (n + 4.0)
        cs = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        c1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
        cmu = min(2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff), 1.0 - c1)
        ds = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0))) + cs
        chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n**2))

        # Pasos y_i = (x_i - m)/sigma con la media ANTIGUA.
        old_mean = self.mean
        ys = [(pop.candidates[:, idx] - old_mean) / self.sigma for idx in elite]
        weighted_step = sum(w * y for w, y in zip(weights, ys))

        # Camino de sigma y adaptación del paso.
        self.ps = (1.0 - cs) * self.ps + np.sqrt(cs * (2.0 - cs) * mu_eff) * weighted_step
        ps_norm = np.linalg.norm(self.ps)
        self.sigma *= np.exp((cs / ds) * (ps_norm / chi_n - 1.0))

        # Camino de C (mitigación h_sigma de arranque).
        exp_len = np.sqrt(1.0 - (1.0 - cs) ** (2 * (self.generation + 1)))
        h_sigma = 1.0 if ps_norm / exp_len < (1.4 + 2.0 / (n + 1.0)) * chi_n else 0.0
        self.pc = (1.0 - cc) * self.pc + h_sigma * np.sqrt(cc * (2.0 - cc) * mu_eff) * weighted_step

        # Covarianza: rank-one + rank-mu.
        rank_one = np.outer(self.pc, self.pc)
        rank_mu = sum(w * np.outer(y, y) for w, y in zip(weights, ys))
        delta_h = (1.0 - h_sigma) * cc * (2.0 - cc)
        self.covariance = (
            (1.0 - c1 - cmu) * self.covariance
            + c1 * (rank_one + delta_h * self.covariance)
            + cmu * rank_mu
        )

        # Nueva media.
        self.mean = old_mean + self.sigma * weighted_step
        self.generation += 1
