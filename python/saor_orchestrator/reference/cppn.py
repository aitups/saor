"""CPPN (Red de Patrones de Composición) — genoma indirecto desacoplado.

Referencia NumPy de `saor_domain::cppn` (misma semántica: sustrato 2D en
[-1,1], vector de entrada de 9 dims — 8 espaciales + `y_layer` de profundidad
(Vía B) — dos capas ocultas de 16 con activaciones heterogéneas y salida
`(w_ij, l_ij)` con `l_ij` pasada por sigmoide).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

CPPN_INPUT_DIM = 9
# 16+16 ocultos (alineado con saor_domain::cppn y el kernel OpenCL): reduce la
# presión de registros del decodificador y permite escalar a bloques reales
# (89M–201M conexiones).
HIDDEN = 16
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


def layer_coord(layer: int, n_layers: int) -> float:
    """Coordenada de profundidad `y_layer ∈ [-1, 1]` (centro de banda).

    Con `n_layers <= 1` devuelve 0.0 (compatibilidad mono-bloque).
    """
    if n_layers <= 1:
        return 0.0
    return -1.0 + 2.0 * (layer + 0.5) / n_layers


def input_vector(d_in: int, d_out: int, i: int, j: int, y_layer: float = 0.0) -> np.ndarray:
    """Vector de entrada de 9 dims para el par `(i, j)` de una capa dada.

    Capa A (`x=-1`) y B (`x=+1`) con coordenada `y` uniforme en `[-1, 1]`, más
    la coordenada de profundidad `y_layer` (Vía B: un solo CPPN por modelo).
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
            y_layer,
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

    def decode_global(
        self,
        d_in: int,
        d_out: int,
        tau: float,
        n_layers: int,
        dense_density: float | None = None,
        step: int = 1,
    ) -> tuple[list[float], np.ndarray]:
        """Decodifica la topología de **todas las capas** con un solo CPPN (Vía B).

        Cada capa se evalúa con su coordenada de profundidad `y_layer`; el
        resultado es una topología heterogénea por capa inducida por el mismo
        genoma. Devuelve `(densidades_por_capa, adjacency)` donde `adjacency`
        es `[n_layers, ceil(d_in*d_out/8)]` en el formato `ffn_dag_adjacency`.

        `step > 1` submuestrea el sustrato (retícula estridente): estima las
        densidades por capa ~`step²`× más rápido con error estadístico pequeño
        (std ≈ sqrt(p(1-p)/n_sampled)); las adyacencias NO son exactas con
        `step > 1` (solo útiles como estimador de densidad para el loop CMA-ES).

        Si `dense_density` se da, se reescala la densidad media global al
        objetivo (frontera de Pareto) moviendo el umbral efectivo.
        """
        total = d_in * d_out
        n_bytes = (total + 7) // 8
        # Sustrato vectorizado: retícula estridente cuando step > 1.
        ii_all = np.arange(0, d_in, step)
        jj_all = np.arange(0, d_out, step)
        ii, jj = np.meshgrid(ii_all, jj_all, indexing="ij")
        yi = -1.0 + 2.0 * ii.astype(np.float32) / (d_in - 1) if d_in > 1 else np.zeros_like(ii, np.float32)
        yj = -1.0 + 2.0 * jj.astype(np.float32) / (d_out - 1) if d_out > 1 else np.zeros_like(jj, np.float32)
        base = np.stack(
            [
                np.full_like(yi, -1.0),  # x_i
                yi,
                np.full_like(yi, 1.0),   # x_j
                yj,
                np.full_like(yi, 2.0),   # dx
                yj - yi,
                np.sin(np.pi * yi),
                np.cos(np.pi * yj),
            ],
            axis=-1,
        )  # [d_in/step, d_out/step, 8]
        n_sampled = base.shape[0] * base.shape[1]
        adjs = np.zeros((n_layers, n_bytes), np.uint8) if step == 1 else None
        densities: list[float] = []
        for layer in range(n_layers):
            yl = np.float32(layer_coord(layer, n_layers))
            v = np.concatenate([base, np.full_like(base[..., :1], yl)], axis=-1)
            flat = v.reshape(-1, CPPN_INPUT_DIM)
            # Activaciones por-neurona aplicadas por columna ([N, H]).
            h0 = np.empty((flat.shape[0], HIDDEN), np.float32)
            pre0 = self.b0[:, 0] + flat @ self.w0.T
            for name in np.unique(self.acts0):
                cols = np.where(self.acts0 == name)[0]
                h0[:, cols] = _apply(str(name), pre0[:, cols])
            h1 = np.empty((flat.shape[0], HIDDEN), np.float32)
            pre1 = self.b1[:, 0] + h0 @ self.w1.T
            for name in np.unique(self.acts1):
                cols = np.where(self.acts1 == name)[0]
                h1[:, cols] = _apply(str(name), pre1[:, cols])
            out = self.b2[:, 0] + h1 @ self.w2.T
            l = 1.0 / (1.0 + np.exp(-out[:, 1]))
            active_mask = l > tau
            active = int(active_mask.sum())
            if step == 1:
                bits = np.nonzero(active_mask.reshape(ii_all.size, jj_all.size))
                for idx in range(len(bits[0])):
                    bit = bits[0][idx] * d_out + bits[1][idx]
                    adjs[layer, bit // 8] |= 1 << (bit % 8)
            densities.append(active / n_sampled)
        if dense_density is not None and densities and sum(densities) > 0:
            k = np.clip(dense_density / (sum(densities) / n_layers), 1e-3, 1e3)
            densities = [min(1.0, d * k) for d in densities]
        return densities, adjs
