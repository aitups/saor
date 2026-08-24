"""Distancia arquitectónica — Hamming normalizada sobre la matriz de adyacencia.

Referencia NumPy de `saor_domain::arch_distance`. Como el bloque original es
denso (`A0 = 1`), `D_arch` se reduce a la esparcidad del candidato y se calcula
por popcount sobre el bit-tensor `ffn_dag_adjacency`.
"""

from __future__ import annotations

from typing import Sequence


def active_connections(adjacency_bits: Sequence[int]) -> int:
    """Conteo de bits activos (popcount) sobre el tensor de adyacencia."""
    return sum(int(b).bit_count() for b in adjacency_bits)


def hamming_sparsity(adjacency_bits: Sequence[int], total_connections: int) -> float:
    """`D_arch = 1 - activas/total` en [0, 1]. El contrato exige >= 0.4."""
    assert total_connections > 0
    active = min(active_connections(adjacency_bits), total_connections)
    return 1.0 - active / total_connections
