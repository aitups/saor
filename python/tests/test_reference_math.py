"""Tests de la referencia NumPy: CPPN, topología, CKA, reconciliación y D_arch."""

import numpy as np
import pytest

from saor_orchestrator.reference.arch_distance import (
    active_connections,
    hamming_sparsity,
)
from saor_orchestrator.reference.cka import centered_cka, gram_matrix
from saor_orchestrator.reference.cppn import (
    CPPN_INPUT_DIM,
    HIDDEN,
    CppnGenome,
    input_vector,
)
from saor_orchestrator.reference.reconciler import hot_indices, identity_projection
from saor_orchestrator.reference.topology import instantiate


def test_input_vector_respeta_rangos():
    v = input_vector(4, 4, 0, 3)
    assert v.shape == (CPPN_INPUT_DIM,)
    assert v[0] == -1.0
    assert v[2] == 1.0
    assert -1.0 <= v[1] <= 1.0
    assert -1.0 <= v[3] <= 1.0
    assert v[4] == pytest.approx(2.0)  # dx = x_j - x_i


def test_evaluacion_determinista_y_l_en_rango():
    genome = CppnGenome()
    w1, l1 = genome.evaluate(input_vector(8, 8, 2, 5))
    w2, l2 = genome.evaluate(input_vector(8, 8, 2, 5))
    assert w1 == w2 and l1 == l2
    # Con genoma de ceros: b2[(1,0)] = 0 -> sigmoide(0) = 0.5
    assert l1 == pytest.approx(0.5)
    assert 0.0 < l1 < 1.0


def test_tamano_genoma():
    assert (
        CppnGenome().param_count
        == 8 * HIDDEN + HIDDEN * HIDDEN + HIDDEN * 2 + HIDDEN + HIDDEN + 2
    )


def test_tau_extremo_vacia_o_llena():
    genome = CppnGenome()  # l = 0.5 siempre
    dense = instantiate(genome, 4, 4, 0.0)
    assert dense.active_connections() == 16
    assert dense.sparsity() == pytest.approx(0.0)
    empty = instantiate(genome, 4, 4, 1.0)
    assert empty.active_connections() == 0
    assert empty.sparsity() == pytest.approx(1.0)


def test_sparsity_por_encima_del_umbral_de_contrato():
    genome = CppnGenome()
    t = instantiate(genome, 16, 16, 0.51)  # l=0.5 fijo -> todo inactivo
    assert t.sparsity() >= 0.4


def test_cka_identidad_es_uno():
    h = np.array(
        [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [1.0, 1.0, 0.0], [0.5, 0.5, 1.0]],
        np.float32,
    )
    k = gram_matrix(h)
    cka = centered_cka(k, k)
    assert cka == pytest.approx(1.0, abs=1e-4)


def test_cka_de_matrices_no_relacionadas_es_baja():
    k0 = np.eye(4, dtype=np.float32)
    k1 = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        np.float32,
    )
    cka = centered_cka(k0, k1)
    assert cka < 0.9, f"CKA debe estar bajo el filtro de 0.90, obtuve {cka}"


def test_hot_indices_ordena_por_varianza():
    var = np.array([1.0, 9.0, 4.0, 16.0])
    assert hot_indices(var, 2) == [3, 1]


def test_proyeccion_identidad_preserva_canales():
    p = identity_projection(4, 2, [3, 1])
    assert p[3, 0] == 1.0
    assert p[1, 1] == 1.0
    assert p[0, 0] == 0.0


def test_hamming_sparsity():
    assert hamming_sparsity([0b0000_1111], 4) == 0.0  # denso
    assert hamming_sparsity([0, 0], 16) == 1.0  # vacío
    bits = [0b0101_0101, 0b0000_0011]  # 6 activas de 16
    assert hamming_sparsity(bits, 16) == pytest.approx(0.625)
    assert active_connections([0b1111_0000, 0b0000_0001]) == 5
    assert hamming_sparsity(bits, 16) >= 0.4  # contrato
