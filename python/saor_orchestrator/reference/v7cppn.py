"""Vía B-v7: CPPN que GENERA los pesos del modelo (opción b) con GEOMETRÍA
APRENDIDA (corrección vía 2 del diseño): cada canal (hidden d_h e intermedio
d_i) tiene coordenadas libres en el genoma. La CPPN se evalúa sobre las
coordenadas aprendidas, permitiendo que la regresión del warm-up reordene los
canales hasta que los pesos sean expresables por la función suave.

Genoma = [CPPN] + [coord_h: d_h] + [coord_i: d_i].
"""
from __future__ import annotations

import numpy as np


class V7Cppn:
    """CPPN generadora de pesos con geometría aprendida por canal."""

    def __init__(self, hidden: int = 64, n_layers_cppn: int = 3,
                 d_h: int = 576, d_i: int = 1536):
        self.hidden = hidden
        self.n_layers_cppn = n_layers_cppn
        self.d_h, self.d_i = d_h, d_i
        self.input_dim = 6  # [c_in, c_out, c_out-c_in, z, 0, 0]
        cppn_n = (self.input_dim * hidden + hidden
                  + (n_layers_cppn - 1) * (hidden * hidden + hidden)
                  + 2 * hidden + 2)
        self.cppn_n = cppn_n
        self.param_count = cppn_n + d_h + d_i

    @classmethod
    def from_flatten(cls, flat: np.ndarray, hidden: int = 64, n_layers_cppn: int = 3,
                     d_h: int = 576, d_i: int = 1536) -> "V7Cppn":
        self = cls(hidden, n_layers_cppn, d_h, d_i)
        o = 0
        f = np.asarray(flat, np.float64)
        self.w0 = f[o:o + self.input_dim * hidden].reshape(hidden, self.input_dim)
        o += self.input_dim * hidden
        self.b0 = f[o:o + hidden]; o += hidden
        ws, bs = [], []
        for _ in range(n_layers_cppn - 1):
            ws.append(f[o:o + hidden * hidden].reshape(hidden, hidden))
            o += hidden * hidden
            bs.append(f[o:o + hidden]); o += hidden
        self.w_mid, self.b_mid = ws, bs
        self.w2 = f[o:o + 2 * hidden].reshape(2, hidden); o += 2 * hidden
        self.b2 = f[o:o + 2]; o += 2
        self.coord_h = f[o:o + d_h]; o += d_h
        self.coord_i = f[o:o + d_i]
        return self

    def flatten(self) -> np.ndarray:
        parts = [self.w0.ravel(), self.b0]
        parts += [w.ravel() for w in self.w_mid] + [b for b in self.b_mid]
        parts += [self.w2.ravel(), self.b2, self.coord_h, self.coord_i]
        return np.concatenate(parts).astype(np.float32)

    def __call__(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """v: [N, 6] -> (w: [N], l: [N])."""
        h = np.tanh(v @ self.w0.T + self.b0)
        for w, b in zip(self.w_mid, self.b_mid):
            h = np.sin(h * np.pi)
            h = np.tanh(h @ w.T + b)
            h = np.exp(-0.5 * (h * 1.2) ** 2)
        out = h @ self.w2.T + self.b2
        return out[:, 0], 1.0 / (1.0 + np.exp(-out[:, 1]))


def decode_block(g: V7Cppn, d_in: int, d_out: int, z: float,
                 which: str, tau: float = 0.5, n_sample: int | None = None,
                 idx_sample: np.ndarray | None = None) -> np.ndarray:
    """Decodifica W [d_out, d_in]. which="hi": hidden->inter; "ih": inter->hidden.

    Si `idx_sample` se da (indices fila-mayor), solo se decodifican esas
    posiciones (para la regresion del warm-up sobre una muestra).
    """
    if which == "hi":
        c_in, c_out = g.coord_h, g.coord_i
    else:
        c_in, c_out = g.coord_i, g.coord_h
    c_in = np.clip(c_in, -5.0, 5.0).astype(np.float32)
    c_out = np.clip(c_out, -5.0, 5.0).astype(np.float32)
    zc = np.float32(np.clip(z, -2.0, 2.0))

    def grid_for(ci: np.ndarray, co: np.ndarray) -> np.ndarray:
        Xj, Xi = np.meshgrid(co, ci, indexing="ij")
        return np.stack([Xi, Xj, Xj - Xi, np.full_like(Xi, zc),
                         np.zeros_like(Xi), np.zeros_like(Xj)], axis=-1)

    if idx_sample is None:
        F = grid_for(c_in, c_out).reshape(-1, 6)
        w, l = g(F)
        W = np.where(l.reshape(d_out, d_in) > tau,
                     w.reshape(d_out, d_in).astype(np.float32), 0.0)
        return W
    # Solo las posiciones muestreadas (regresion del warm-up).
    F = grid_for(c_in, c_out).reshape(-1, 6)[idx_sample]
    w, l = g(F)
    return w.astype(np.float32), (l > tau)
