"""Vía B: sonda del CPPN global con coordenada de capa (sustrato v5).

Decodifica UN solo CPPN (genoma aleatorio o guardado) para todas las capas de
SmolLM2 (30 × 576→1536) y verifica que la coordenada de profundidad `y_layer`
induce topologías heterogéneas por capa — la hipótesis de compresión variable
por modelo, ahora codificada en el propio genoma.

Uso:
  python python/scripts/via_b_global_probe.py [--seed 7] [--layers 30]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saor_orchestrator.reference.cppn import CppnGenome, layer_coord  # noqa: E402


def random_genome(seed: int) -> CppnGenome:
    rng = np.random.default_rng(seed)
    g = CppnGenome()
    g.w0 = rng.standard_normal(g.w0.shape).astype(np.float32) * 0.8
    g.b0 = rng.standard_normal(g.b0.shape).astype(np.float32) * 0.3
    g.w1 = rng.standard_normal(g.w1.shape).astype(np.float32) * 0.5
    g.b1 = rng.standard_normal(g.b1.shape).astype(np.float32) * 0.2
    g.w2 = rng.standard_normal(g.w2.shape).astype(np.float32) * 0.5
    g.b2 = rng.standard_normal(g.b2.shape).astype(np.float32) * 0.2
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description="Sonda del CPPN global (Vía B)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--layers", type=int, default=30)
    ap.add_argument("--d-in", type=int, default=576)
    ap.add_argument("--d-out", type=int, default=1536)
    ap.add_argument("--tau", type=float, default=0.42)
    ap.add_argument("--target-density", type=float, default=None,
                    help="densidad global objetivo (reescala k)")
    args = ap.parse_args()

    g = random_genome(args.seed)
    n_params = g.param_count
    print(f"CPPN global: {n_params} params (input dim {g.w0.shape[1]})")

    # Sanidad: la coordenada de capa es monótona creciente en [-1, 1].
    coords = [layer_coord(l, args.layers) for l in range(args.layers)]
    assert all(a < b for a, b in zip(coords, coords[1:])), "y_layer no monótona"
    assert abs(coords[0] - (-1.0 + 1.0 / args.layers)) < 1e-6, "primer centro de banda"
    assert abs(coords[-1] - (1.0 - 1.0 / args.layers)) < 1e-6, "último centro de banda"

    densities, adjs = g.decode_global(
        args.d_in, args.d_out, args.tau, args.layers,
        dense_density=args.target_density,
    )
    d = np.asarray(densities)
    print(f"densidad por capa  : {d.round(3).tolist()}")
    print(f"media = {d.mean():.4f}  min = {d.min():.4f} (capa {d.argmin()})  "
          f"max = {d.max():.4f} (capa {d.argmax()})  std = {d.std():.4f}")
    print(f"adjacency total    : {adjs.nbytes / 1e6:.2f} MB  ({adjs.shape})")

    # Heterogeneidad: si std ~ 0, el CPPN global no varía por capa (coordenada
    # muerta o genoma degenerado). La Vía B exige varianza sustancial.
    rel_std = d.std() / (d.mean() + 1e-9)
    print(f"heterogeneidad rel  : {rel_std:.3f} (objetivo > 0.05)")
    ok = rel_std > 0.05 and d.mean() > 0.01
    print(f"veredicto           : {'OK — topología por capa inducida por y_layer' if ok else 'revisar genoma/τ'}")


if __name__ == "__main__":
    main()
