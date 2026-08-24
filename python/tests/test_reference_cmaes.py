"""Tests de la referencia NumPy: CMA-ES y un slice vertical end-to-end sintético."""

import numpy as np
import pytest

from saor_orchestrator.reference.cka import centered_cka, gram_matrix
from saor_orchestrator.reference.cmaes import CmaEsParams, CmaEsState, default_weights
from saor_orchestrator.reference.cppn import CppnGenome
from saor_orchestrator.reference.topology import instantiate


def test_poblacion_determinista_por_semilla():
    params = CmaEsParams(8, seed=42)
    state = CmaEsState(params, np.zeros(8))
    p1 = state.spawn_population(1234)
    p2 = state.spawn_population(1234)
    np.testing.assert_array_equal(p1.perturbations, p2.perturbations)
    p3 = state.spawn_population(9999)
    assert not np.array_equal(p1.perturbations, p3.perturbations)


def test_cmaes_minimiza_cuadratica_sintetica():
    params = CmaEsParams(6, seed=7)
    state = CmaEsState(params, np.full(6, 10.0))
    best_ever = np.inf
    for gen in range(80):
        pop = state.spawn_population(params.seed + gen)
        scores = sorted(
            (
                (idx, 0.5 * np.sum((pop.candidates[:, idx] - 1.0) ** 2))
                for idx in range(params.lambda_)
            ),
            key=lambda t: t[1],
        )
        best_ever = min(best_ever, scores[0][1])
        elite = [idx for idx, _ in scores[: params.mu]]
        state.update(pop, elite)
    assert best_ever < 1e-2, f"CMA-ES debe converger, best={best_ever}"


def test_pesos_logaritmicos_normalizados():
    w = default_weights(8)
    assert w.sum() == pytest.approx(1.0, abs=1e-5)
    assert all(a >= b for a, b in zip(w, w[1:]))


def test_flujo_cppn_topologia_cka_end_to_end():
    """Slice vertical mínimo: genoma -> DAG -> salida dispersa -> CKA finito."""
    rng = np.random.default_rng(0)
    genome = CppnGenome()
    genome.w0 = rng.normal(0, 0.5, genome.w0.shape).astype(np.float32)
    genome.w1 = rng.normal(0, 0.5, genome.w1.shape).astype(np.float32)
    genome.w2 = rng.normal(0, 0.5, genome.w2.shape).astype(np.float32)

    d_in, d_out, b = 32, 32, 128
    topo = instantiate(genome, d_in, d_out, 0.35)

    # Profesor: bloque denso W0; entrada X aleatoria.
    x = rng.normal(0, 1, (b, d_in)).astype(np.float32)
    w0 = rng.normal(0, 1, (d_out, d_in)).astype(np.float32)
    h0 = x @ w0.T

    # Candidato: reconstrucción dispersa densificada para la comparación.
    dense_w = np.zeros((d_out, d_in), np.float32)
    w_idx = 0
    conn = 0
    for byte in topo.adjacency_bits:
        for bit in range(8):
            if int(byte) & (1 << bit):
                i, j = divmod(conn, d_out)
                dense_w[j, i] = topo.weights[w_idx]
                w_idx += 1
            conn += 1
    assert w_idx == topo.active_connections()
    h1 = x @ dense_w.T

    cka = centered_cka(gram_matrix(h0), gram_matrix(h1))
    assert 0.0 <= cka <= 1.0
    assert 0.0 <= topo.sparsity() <= 1.0
