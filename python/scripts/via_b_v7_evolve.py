"""Vía B-v7 — evolución con la KL global (hayai) como ÚNICO fitness.

Sin fases de regresión en espacio de pesos: el genoma (CPPN generadora de pesos
+ geometría aprendida por canal) se inicializa y CMA-ES lo evoluciona con la KL
global medida por el motor (inyección D16 --all-blocks + kl_eval). La
inicialización acepta el ruido inicial (corrección vía 2 del diseño); la
evolución lo corrige orgánicamente.

Genoma = [CPPN] + [coord_h] + [coord_i] (ver v7cppn.py).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

import numpy as np

import sys
sys.path.insert(0, r"d:\Documents\pySrc\saor\python")
from saor_orchestrator.reference.cmaes import CmaEsParams, CmaEsState  # noqa: E402
from saor_orchestrator.reference.v7cppn import V7Cppn, decode_block  # noqa: E402

SMOL = r"d:\Documents\PySrc\hayai\models\SmolLM2-135M-Instruct-Q4_K_M.gguf"
W_GEN = r"d:\Documents\pySrc\.scratch\w_v7_gen"
EMBED = r"d:\Documents\PySrc\saor\target\release\embed_sparse.exe"
KLE = r"d:\Documents\PySrc\hayai\target\release\examples\kl_eval.exe"
P = r"d:\Documents\pySrc\.scratch\calib128.txt"
N_LAYERS = 30
D_H, D_I = 576, 1536
BLOCKS = [("ffn_gate", 1536, 576, "hi"), ("ffn_up", 1536, 576, "hi"),
          ("ffn_down", 576, 1536, "ih")]
TEACHER_CACHE = r"d:\Documents\pySrc\.scratch\v7_teacher.bin"


def decode_model(g: V7Cppn, tau: float = 0.5) -> None:
    if os.path.isdir(W_GEN):
        for f in os.listdir(W_GEN):
            os.remove(os.path.join(W_GEN, f))
    else:
        os.makedirs(W_GEN, exist_ok=True)
    dims = {}
    for layer in range(N_LAYERS):
        z = -1.0 + 2.0 * (layer + 0.5) / N_LAYERS
        for block, d_out, d_in, which in BLOCKS:
            W = decode_block(g, d_in, d_out, z, which, tau)
            with open(os.path.join(W_GEN, f"w.{layer}.{block}.bin"), "wb") as f:
                f.write(np.ascontiguousarray(W).astype(np.float32).tobytes())
            dims[f"blk.{layer}.{block}"] = {"d_in": d_in, "d_out": d_out}
    with open(os.path.join(W_GEN, "meta.json"), "w") as f:
        json.dump({"n_layers": N_LAYERS, **dims}, f)


def kl_fitness(g: V7Cppn, n_pos: int) -> float:
    decode_model(g)
    spf = os.path.join(r"d:\Documents\pySrc\.scratch", "v7_sp.txt")
    with open(spf, "w") as f:
        f.write("\n".join([f"{0.001:.4f}"] * N_LAYERS) + "\n")
    emb = r"d:\Documents\pySrc\.scratch\v7_cand.gguf"
    if os.path.exists(emb):
        os.remove(emb)
    r = subprocess.run([EMBED, "--model", SMOL, "--out", emb, "--weights", W_GEN,
                        "--sparsities", spf, "--all-blocks"], capture_output=True, text=True)
    if "ok" not in r.stdout:
        return float("inf")
    r = subprocess.run([KLE, "--orig", SMOL, "--sparse", emb, "--prompts", P,
                        "--n-positions", str(n_pos), "--device", "auto",
                        "--teacher-cache", TEACHER_CACHE],
                       capture_output=True, text=True, timeout=1800)
    outs = [l for l in (r.stdout + r.stderr).splitlines() if "kl_global" in l]
    if not outs:
        return float("inf")
    return float(json.loads(outs[-1])["kl_global"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-pos", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--lambda-mult", type=float, default=1.0)
    ap.add_argument("--sigma0", type=float, default=0.15)
    args = ap.parse_args()

    cppn_n = V7Cppn(hidden=args.hidden).cppn_n
    geo0 = np.concatenate([
        np.linspace(-1, 1, D_H, dtype=np.float64),
        np.linspace(-1, 1, D_I, dtype=np.float64)])
    rng = np.random.default_rng(123)
    cppn0 = rng.normal(0.0, 0.3, cppn_n)
    mean0 = np.concatenate([cppn0, geo0])
    params = CmaEsParams(len(mean0), args.seed, sigma0=args.sigma0)
    params.lambda_ = max(8, int(params.lambda_ * args.lambda_mult))
    params.mu = params.lambda_ // 2

    def build(flat):
        return V7Cppn.from_flatten(flat, args.hidden, d_h=D_H, d_i=D_I)

    state = CmaEsState(params, mean0)
    print(f"v7: genoma {len(mean0)} floats, lambda={params.lambda_}, fitness=KL", flush=True)
    best = float("inf")
    for gen in range(args.gens):
        pop = state.spawn_population(args.seed + gen)
        scored = []
        for c in range(pop.candidates.shape[1]):
            g = build(pop.candidates[:, c])
            kl = kl_fitness(g, args.n_pos)
            scored.append((-kl if np.isfinite(kl) else -1e9, kl))
        scored.sort(key=lambda s: -s[0])
        order = sorted(range(len(scored)), key=lambda i: -scored[i][0])
        state.update(pop, order[: params.mu])
        best = min(best, scored[0][1])
        print(f"gen {gen:2d} best_kl={best:.4f}", flush=True)
        with open(r"d:\Documents\pySrc\.scratch\v7_genome.bin", "wb") as f:
            f.write(pop.candidates[:, order[0]].astype(np.float32).tobytes())
    print(f"RESULTADO v7: mejor KL = {best:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
