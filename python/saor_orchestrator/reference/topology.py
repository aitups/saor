"""Instanciación de topologías DAG + máscara de esparsidad dinámica τ.

Referencia NumPy de `saor_domain::topology`. La adyacencia se empaqueta como
bits (LSB-first, igual que el bit-tensor `ffn_dag_adjacency` del GGUF disperso)
para que `D_arch` sea un popcount.
"""

from __future__ import annotations

import numpy as np

from .cppn import CppnGenome, input_vector


class Topology:
    """Matriz de adyacencia (bits) + pesos del DAG instanciado."""

    def __init__(
        self,
        adjacency_bits: np.ndarray,
        total_connections: int,
        weights: list[float],
    ) -> None:
        self.adjacency_bits = np.asarray(adjacency_bits, dtype=np.uint8)
        self.total_connections = int(total_connections)
        self.weights = weights

    def active_connections(self) -> int:
        """Conteo por popcount sobre el tensor de bits."""
        n = sum(int(b).bit_count() for b in self.adjacency_bits)
        return min(n, self.total_connections)

    def sparsity(self) -> float:
        """`D_arch(A0, A1) = 1 - activas/total` (esparcidad del candidato)."""
        if self.total_connections == 0:
            raise ValueError("total_connections debe ser > 0")
        return 1.0 - self.active_connections() / self.total_connections


def instantiate(genome: CppnGenome, d_in: int, d_out: int, tau: float) -> Topology:
    """Evalúa la CPPN para todo el sustrato `d_in x d_out` y filtra por `τ`.

    `A_ij = 1 si l_ij > tau`, con `W_ij = w_ij` para las conexiones activas.
    """
    total = d_in * d_out
    bits = np.zeros((total + 7) // 8, dtype=np.uint8)
    weights: list[float] = []
    for i in range(d_in):
        for j in range(d_out):
            w, l = genome.evaluate(input_vector(d_in, d_out, i, j))
            idx = i * d_out + j
            if l > tau:
                bits[idx // 8] |= np.uint8(1 << (idx % 8))
                weights.append(w)
    return Topology(bits, total, weights)
