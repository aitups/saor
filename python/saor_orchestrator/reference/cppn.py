"""CPPN (Red de Patrones de Composición) — genoma indirecto desacoplado.

Referencia NumPy de `saor_domain::cppn` (misma semántica: sustrato 2D en
[-1,1], vector de entrada de 8 dims, dos capas ocultas de 64 con activaciones
heterogéneas y salida `(w_ij, l_ij)` con `l_ij` pasada por sigmoide).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

CPPN_INPUT_DIM = 8
HIDDEN = 64
GENOME_SIZE = 32 * 1024


def _apply(name: str, x: np.ndarray) -> np.ndarray:
    if name == "gaussian":
        return np.exp(-x * x)
    if name == "tanh":
        return np.tanh(x)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if name == "sine":
        return np.sin(x)
    raise ValueError(f"activación desconocida: {name}")


def apply_activation(names: Sequence[str], x: np.ndarray) -> np.ndarray:
    """Aplica la activación indicada por cada elemento de `names` a `x`."""
    names = np.asarray(names)
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    for name in np.unique(names):
        mask = names == name
        out[mask] = _apply(str(name), x[mask])
    return out


def input_vector(d_in: int, d_out: int, i: int, j: int) -> np.ndarray:
    """Vector de entrada de 8 dims para el par `(i, j)` (especificación v4).

    Capa A (`x=-1`) y B (`x=+1`) con coordenada `y` uniforme en `[-1, 1]`.
    """
    y_i = -1.0 + 2.0 * i / (d_in - 1) if d_in > 1 else 0.0
    y_j = -1.0 + 2.0 * j / (d_out - 1) if d_out > 1 else 0.0
    x_i, x_j = -1.0, 1.0
    return np.array(
        [
            x_i,
            y_i,
            x_j,
            y_j,
            x_j - x_i,
            y_j - y_i,
            np.sin(np.pi * y_i),
            np.cos(np.pi * y_j),
        ],
        dtype=np.float32,
    )


class CppnGenome:
    """Genoma CPPN con topología fija de 2 capas ocultas (64+64)."""

    def __init__(self) -> None:
        self.w0 = np.zeros((HIDDEN, CPPN_INPUT_DIM), np.float32)
        self.w1 = np.zeros((HIDDEN, HIDDEN), np.float32)
        self.w2 = np.zeros((2, HIDDEN), np.float32)
        self.b0 = np.zeros((HIDDEN, 1), np.float32)
        self.b1 = np.zeros((HIDDEN, 1), np.float32)
        self.b2 = np.zeros((2, 1), np.float32)
        self.acts0 = np.full(HIDDEN, "tanh", dtype=object)
        self.acts1 = np.full(HIDDEN, "sine", dtype=object)

    @property
    def param_count(self) -> int:
        return (
            self.w0.size
            + self.w1.size
            + self.w2.size
            + self.b0.size
            + self.b1.size
            + self.b2.size
        )

    @classmethod
    def from_flatten(cls, flat: np.ndarray) -> "CppnGenome":
        """Reconstruye el genoma desde el aplanado fila-mayor del kernel OpenCL.

        Orden: `w0 | b0 | w1 | b1 | w2 | b2` (espejo de
        `saor_domain::cppn::CppnGenome::from_flatten`).
        """
        g = cls()
        flat = np.asarray(flat, np.float32)
        pos = 0
        for o in range(HIDDEN):
            for k in range(CPPN_INPUT_DIM):
                g.w0[o, k] = flat[pos]
                pos += 1
        for o in range(HIDDEN):
            g.b0[o, 0] = flat[pos]
            pos += 1
        for o in range(HIDDEN):
            for k in range(HIDDEN):
                g.w1[o, k] = flat[pos]
                pos += 1
        for o in range(HIDDEN):
            g.b1[o, 0] = flat[pos]
            pos += 1
        for r in range(2):
            for k in range(HIDDEN):
                g.w2[r, k] = flat[pos]
                pos += 1
        g.b2[0, 0] = flat[pos]
        pos += 1
        g.b2[1, 0] = flat[pos]
        pos += 1
        assert pos == len(flat), "aplanado de longitud incorrecta"
        return g

    def flatten(self) -> np.ndarray:
        """Aplana el genoma en el orden del kernel OpenCL (fila-mayor)."""
        parts = [
            self.w0,
            self.b0,
            self.w1,
            self.b1,
            self.w2,
            self.b2,
        ]
        return np.concatenate([p.reshape(-1, order="C").astype(np.float32) for p in parts])

    def evaluate(self, v: np.ndarray) -> tuple[float, float]:
        """Evalúa la CPPN para un par de neuronas. Devuelve `(w_ij, l_ij)`."""
        v = np.asarray(v, dtype=np.float32)
        h0 = apply_activation(self.acts0, self.b0[:, 0] + self.w0 @ v)
        h1 = apply_activation(self.acts1, self.b1[:, 0] + self.w1 @ h0)
        out = self.b2[:, 0] + self.w2 @ h1
        w, l_raw = float(out[0]), float(out[1])
        return w, 1.0 / (1.0 + np.exp(-l_raw))
