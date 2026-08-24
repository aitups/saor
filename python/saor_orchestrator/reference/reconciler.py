"""Reconciliación dimensional en topologías no dirigidas (sección 5 de v4).

Referencia NumPy de `saor_domain::reconciler`:
1. `d_A > d_B` → subsampling por índices calientes (top-`k` canales por
   varianza de activación del profesor `H0`).
2. `d_A < d_B` → proyección lineal adaptativa que emula identidad.
"""

from __future__ import annotations

import numpy as np


def hot_indices(activation_variance: np.ndarray, k: int) -> list[int]:
    """Índices de los `k` canales de mayor varianza (desempate por índice)."""
    variance = np.asarray(activation_variance, np.float64).reshape(-1)
    assert k <= variance.size
    # Orden descendente por varianza, estable (empates → índice menor primero).
    return np.argsort(-variance, kind="stable")[:k].astype(int).tolist()


def identity_projection(d_in: int, d_out: int, selected: list[int]) -> np.ndarray:
    """`W_proj ∈ R^{d_in x d_out}` con `W_proj[row, col] = 1` para los seleccionados."""
    assert len(selected) <= min(d_in, d_out)
    proj = np.zeros((d_in, d_out), dtype=np.float32)
    for col, row in enumerate(selected):
        proj[row, col] = 1.0
    return proj
