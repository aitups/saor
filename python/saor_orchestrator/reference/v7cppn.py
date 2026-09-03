"""Vía B-v7: CPPN que GENERA los pesos del modelo (rewrite total, opción b).

La CPPN v7 (sustrato 9-D global: x1,y1,x2,y2,z1,z2 + dx,dy,dz; 2×H ocultos con
activaciones heterogéneas tanh/seno/gaussiana) se evalúa sobre la geometría de
cada matriz del FFN (gate/up/down por capa) y produce los pesos W[j,i] y el
link l[j,i]. El decode devuelve las matrices completas para la inyección.

Genoma: w0[H,9]+b0[H]+w1[H,H]+b1[H]+w2[2,H]+b2[2] (+ multiplicadores por
bloque para permitir que el mismo CPPN pinte gate/up/down con escalas propias).
"""
from __future__ import annotations

import numpy as np


class V7Cppn:
    """CPPN generadora de pesos del modelo (opción b)."""

    def __init__(self, hidden: int = 64, n_layers_cppn: int = 3):
        self.hidden = hidden
        self.n_layers_cppn = n_layers_cppn
        # capa 0: 9 -> hidden; capas intermedias: hidden -> hidden;
        # última: hidden -> 2 (w, l). Genoma aplanado.
        n = 9 * hidden + hidden + (n_layers_cppn - 1) * (hidden * hidden + hidden) + 2 * hidden + 2
        self.param_count = n
        self.input_dim = 9

    @classmethod
    def from_flatten(cls, flat: np.ndarray, hidden: int = 64, n_layers_cppn: int = 3) -> "V7Cppn":
        self = cls(hidden, n_layers_cppn)
        o = 0
        self.layers = []
        self.w0 = flat[o:o + 9 * hidden].reshape(hidden, 9); o += 9 * hidden
        self.b0 = flat[o:o + hidden]; o += hidden
        ws, bs = [], []
        for _ in range(n_layers_cppn - 1):
            ws.append(flat[o:o + hidden * hidden].reshape(hidden, hidden)); o += hidden * hidden
            bs.append(flat[o:o + hidden]); o += hidden
        self.w_mid, self.b_mid = ws, bs
        self.w2 = flat[o:o + 2 * hidden].reshape(2, hidden); o += 2 * hidden
        self.b2 = flat[o:o + 2]
        return self

    def __call__(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """v: [N,9] -> (w: [N], l: [N])."""
        h = np.tanh(v @ self.w0.T + self.b0)
        h = np.sin(h * np.pi)  # activación rica en las capas intermedias
        for w, b in zip(self.w_mid, self.b_mid):
            h = np.tanh(h @ w.T + b)
            h = np.exp(-0.5 * (h * 1.2) ** 2)  # gaussiana
        out = h @ self.w2.T + self.b2
        return out[:, 0], 1.0 / (1.0 + np.exp(-out[:, 1]))


def geometry_matrix(d_in: int, d_out: int, z1: float, z2: float,
                    input_dim: int = 9) -> np.ndarray:
    """Grid [d_out*d_in, 9] de la geometría global para una matriz."""
    xi = np.linspace(-1, 1, d_in, dtype=np.float32)
    xj = np.linspace(-1, 1, d_out, dtype=np.float32)
    Xj, Xi = np.meshgrid(xj, xi, indexing="ij")
    if input_dim == 6:
        return np.stack([Xi, np.zeros_like(Xi), Xj, np.zeros_like(Xj),
                         np.full_like(Xi, z1), np.full_like(Xi, z2)],
                        axis=-1).reshape(-1, 6)
    return np.stack([Xi, np.zeros_like(Xi), Xj, np.zeros_like(Xj),
                     np.full_like(Xi, z1), np.full_like(Xi, z2),
                     Xj - Xi, np.zeros_like(Xj), np.zeros_like(Xi)],
                    axis=-1).reshape(-1, 9)


def decode_block(g: V7Cppn, d_in: int, d_out: int, z1: float, z2: float,
                 tau: float = 0.5) -> np.ndarray:
    """Decodifica W [d_out, d_in] para un bloque. Link>tau -> conexión con peso w."""
    grid = geometry_matrix(d_in, d_out, z1, z2)
    w, l = g(grid)
    W = np.where(l.reshape(d_out, d_in) > tau,
                 w.reshape(d_out, d_in).astype(np.float32), 0.0)
    return W
