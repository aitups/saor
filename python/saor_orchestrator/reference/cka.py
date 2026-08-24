"""Fitness CKA (Centered Kernel Alignment) — equivalencia funcional local.

Referencia NumPy de `saor_domain::cka`. Compara la huella semántica del
candidato (`K1 = H1 H1^T`) contra el profesor (`K0 = H0 H0^T`) con HSIC.
"""

from __future__ import annotations

import numpy as np


def gram_matrix(h: np.ndarray) -> np.ndarray:
    """Matriz de Gram `K = H H^T` para un lote `[B x D]`."""
    h = np.asarray(h, dtype=np.float32)
    return h @ h.T


def centered_cka(k0: np.ndarray, k1: np.ndarray) -> float:
    """CKA centrado entre dos matrices de Gram del mismo lote (en [0, 1]).

    `1` = alineación idéntica (módulo transformación lineal); el filtro
    determinista del Paso 5 exige `CKA >= 0.90`.
    """
    k0 = np.asarray(k0, dtype=np.float32)
    k1 = np.asarray(k1, dtype=np.float32)
    assert k0.shape == k1.shape, "las matrices de Gram deben ser del mismo lote"
    n = k0.shape[0]
    hc = np.eye(n, dtype=np.float32) - np.ones((n, n), dtype=np.float32) / n
    x = hc @ k0 @ hc
    y = hc @ k1 @ hc
    hsic_xy = float(np.sum(x * y))
    hsic_xx = float(np.sum(x * x))
    hsic_yy = float(np.sum(y * y))
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom <= 1e-12:
        return 0.0
    return float(hsic_xy / denom)
