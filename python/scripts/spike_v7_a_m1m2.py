"""Spike Vía B-v7, interpretación A (training-free): M1/M2.

Valida el decode hypernetwork: una CPPN genera los pesos de una red
(`W[j,i] = f_CPPN(geom_out_j, geom_in_i)`) y CMA-ES la optimiza para reproducir
un **profesor sintético suave** (M2). Confirma que el decode + el bucle CMA-ES
funcionan antes de probar contra un profesor real (M3).

La CPPN: entrada 6-D `(x1,y1,x2,y2,z1,z2)` (+ derivadas dx,dy,dz → 9-D opcional),
2×16 ocultos, salida `[w, l]`. La capa generada: `W[j,i] = w_ij` donde `l_ij` es
el gate (solo si `l_ij > tau` la conexión participa) — sin usar |w| en ningún
momento (búsqueda no dirigida).
"""
from __future__ import annotations

import argparse

import numpy as np

from saor_orchestrator.reference.cmaes import CmaEsParams, CmaEsState

HIDDEN = 16


class HyperCppn:
    """CPPN compositor de pesos. Genoma aplanado: w0[H,IN]+b0[H]+w1[H,H]+b1[H]
    +w2[2,H]+b2[2]. IN = 6 (raw) o 9 (raw + dx,dy,dz). Salida [w, l]."""

    def __init__(self, input_dim: int = 9):
        self.input_dim = input_dim
        n = (HIDDEN * input_dim + HIDDEN + HIDDEN * HIDDEN + HIDDEN + 2 * HIDDEN + 2)
        self.param_count = n

    @classmethod
    def from_flatten(cls, flat: np.ndarray, input_dim: int = 9) -> "HyperCppn":
        self = cls(input_dim)
        o = 0
        self.w0 = flat[o:o + HIDDEN * input_dim].reshape(HIDDEN, input_dim)
        o += HIDDEN * input_dim
        self.b0 = flat[o:o + HIDDEN]
        o += HIDDEN
        self.w1 = flat[o:o + HIDDEN * HIDDEN].reshape(HIDDEN, HIDDEN)
        o += HIDDEN * HIDDEN
        self.b1 = flat[o:o + HIDDEN]
        o += HIDDEN
        self.w2 = flat[o:o + 2 * HIDDEN].reshape(2, HIDDEN)
        o += 2 * HIDDEN
        self.b2 = flat[o:o + 2]
        return self

    def __call__(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """v: [N, input_dim] -> (w: [N], l: [N])."""
        h0 = np.tanh(v @ self.w0.T + self.b0)
        h1 = np.tanh(h0 @ self.w1.T + self.b1)
        out = h1 @ self.w2.T + self.b2
        return out[:, 0], 1.0 / (1.0 + np.exp(-out[:, 1]))


def make_teacher_layer(d_out: int, d_in: int, seed: int) -> np.ndarray:
    """Profesor sintético SUAVE: W[j,i] suave en (i,j) — el caso CPPN-amigable."""
    rng = np.random.default_rng(seed)
    xx = np.linspace(-1, 1, d_in, dtype=np.float32)
    yy = np.linspace(-1, 1, d_out, dtype=np.float32)
    # f(i,j) = sin(pi*x_i*(1+y_j)/2) * cos(pi*y_j/2) — suave y separable-ish.
    W = np.sin(np.pi * xx[None, :] * (1 + yy[:, None]) / 2.0) * np.cos(np.pi * yy[:, None] / 2.0)
    W = (W * rng.normal(0, 0.6, (d_out, 1)).astype(np.float32)).astype(np.float32)
    return W


def geom_vec(x1: float, y1: float, x2: float, y2: float, z1: float, z2: float,
             derived: bool = True) -> np.ndarray:
    """Vector de entrada de la CPPN para el par de nodos."""
    if not derived:
        return np.array([x1, y1, z1, x2, y2, z2], np.float32)
    return np.array([x1, y1, x2, y2, z1, z2, x2 - x1, y2 - y1, z2 - z1], np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-in", type=int, default=48)
    ap.add_argument("--d-out", type=int, default=48)
    ap.add_argument("--n-data", type=int, default=200)
    ap.add_argument("--gens", type=int, default=220)
    ap.add_argument("--lambda-mult", type=int, default=4,
                    help="multiplicador de la poblacion CMA-ES (default 4x22=88)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--raw-6d", action="store_true", help="entrada 6-D cruda (default 9-D)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    X = rng.uniform(-1, 1, (args.n_data, args.d_in)).astype(np.float32)
    W1 = make_teacher_layer(args.d_out, args.d_in, 1)
    W2 = make_teacher_layer(4, args.d_out, 2)
    teacher_out = np.tanh(X @ W1.T) @ W2.T  # [n_data, 4]
    t_energy = float(np.mean(teacher_out ** 2))

    input_dim = 6 if args.raw_6d else 9
    n_params = (HIDDEN * input_dim + HIDDEN + HIDDEN * HIDDEN + HIDDEN + 2 * HIDDEN + 2)
    params = CmaEsParams(n_params, args.seed)
    params.lambda_ = max(8, params.lambda_ * args.lambda_mult)
    params.mu = params.lambda_ // 2
    mean0 = np.random.default_rng(args.seed).standard_normal(n_params).astype(np.float32) * 0.4
    state = CmaEsState(params, mean0)

    xs_in = np.linspace(-1, 1, args.d_in, dtype=np.float32)
    xs_out = np.linspace(-1, 1, args.d_out, dtype=np.float32)
    best = float("inf")
    for gen in range(args.gens):
        pop = state.spawn_population(args.seed + gen)
        scored = []
        for c in range(pop.candidates.shape[1]):
            g = HyperCppn.from_flatten(pop.candidates[:, c], input_dim)
            # Capa 1: W1n[j,i] (z1=z2=-0.5)
            g1 = np.stack([geom_vec(xs_in[i], 0.0, xs_out[j], 0.0, -0.5, -0.5, input_dim == 9)
                           for j in range(args.d_out) for i in range(args.d_in)])
            w1, l1 = g(g1)
            m1 = l1.reshape(args.d_out, args.d_in) > 0.5
            h1 = np.tanh(X @ np.where(m1, w1.reshape(args.d_out, args.d_in), 0.0).T)
            # Capa 2 (z1=z2=0.0)
            g2 = np.stack([geom_vec(xs_out[i], 0.0, xs_out[j], 0.0, 0.0, 0.0, input_dim == 9)
                           for j in range(4) for i in range(args.d_out)])
            w2, l2 = g(g2)
            m2 = l2.reshape(4, args.d_out) > 0.5
            out = h1 @ np.where(m2, w2.reshape(4, args.d_out), 0.0).T
            mse = float(np.mean((out - teacher_out) ** 2))
            scored.append((-mse, mse))
        scored.sort(key=lambda s: -s[0])
        order = sorted(range(len(scored)), key=lambda i: -scored[i][0])
        state.update(pop, order[: params.mu])
        best = min(best, scored[0][1])
        if gen % 10 == 0:
            print(f"gen {gen:3d} best_mse={best:.6f} (energia profesor={t_energy:.3f})", flush=True)
    print(f"RESULTADO M2: mse={best:.6f} vs energia profesor={t_energy:.3f}")
    print("Si mse << energia -> el hypernetwork CPPN EXPRESA el profesor suave (decode OK).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

