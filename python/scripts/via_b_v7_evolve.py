"""Vía B-v7 — bucle evolutivo de extremo a extremo (smol):
genoma v7 -> decode pesos FFN completo -> embed --all-blocks (inyeccion) ->
kl_eval (KL global, fitness) -> CMA-ES.
Ejecución secuencial por candidato (lenta pero completa); el batch es la
optimización posterior.
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
W_TEACH = r"d:\Documents\pySrc\.scratch\w_smol2"  # dims reales (para meta.json)
W_GEN = r"d:\Documents\pySrc\.scratch\w_v7_gen"
EMBED = r"d:\Documents\PySrc\saor\target\release\embed_sparse.exe"
KLE = r"d:\Documents\PySrc\hayai\target\release\examples\kl_eval.exe"
P = r"d:\Documents\pySrc\.scratch\calib128.txt"
N_LAYERS = 30
BLOCKS = [("ffn_gate", 1536, 576), ("ffn_up", 1536, 576), ("ffn_down", 576, 1536)]


def decode_model(g: V7Cppn, tau: float = 0.5) -> None:
    """Decodifica todas las matrices del FFN del smol y las escribe como dump."""
    if os.path.isdir(W_GEN):
        for f in os.listdir(W_GEN):
            os.remove(os.path.join(W_GEN, f))
    else:
        os.makedirs(W_GEN, exist_ok=True)
    dims = {}
    for layer in range(N_LAYERS):
        z = -1.0 + 2.0 * (layer + 0.5) / N_LAYERS
        for block, d_out, d_in in BLOCKS:
            z1, z2 = z, z + 0.02
            W = decode_block(g, d_in, d_out, z1, z2, tau)
            with open(os.path.join(W_GEN, f"w.{layer}.{block}.bin"), "wb") as f:
                f.write(W.astype(np.float32).tobytes())
            dims[f"blk.{layer}.{block}"] = {"d_in": d_in, "d_out": d_out}
    with open(os.path.join(W_GEN, "meta.json"), "w") as f:
        json.dump({"n_layers": N_LAYERS, **dims}, f)


def eval_candidate(g: V7Cppn, n_pos: int) -> float:
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
                        "--teacher-cache", r"d:\Documents\pySrc\.scratch\v7_teacher.bin"],
                       capture_output=True, text=True, timeout=1800)
    outs = [l for l in (r.stdout + r.stderr).splitlines() if "kl_global" in l]
    if not outs:
        return float("inf")
    return float(json.loads(outs[-1])["kl_global"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-pos", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--lambda-mult", type=float, default=2.0)
    args = ap.parse_args()

    cppn = V7Cppn(hidden=args.hidden)
    params = CmaEsParams(cppn.param_count, args.seed)
    params.lambda_ = max(8, int(params.lambda_ * args.lambda_mult))
    params.mu = params.lambda_ // 2
    mean0 = np.random.default_rng(args.seed).standard_normal(cppn.param_count).astype(np.float32) * 0.3
    state = CmaEsState(params, mean0)
    best = float("inf")
    print(f"v7: genoma {cppn.param_count} floats, lambda={params.lambda_}", flush=True)
    for gen in range(args.gens):
        pop = state.spawn_population(args.seed + gen)
        scored = []
        for c in range(pop.candidates.shape[1]):
            g = V7Cppn.from_flatten(pop.candidates[:, c], args.hidden)
            kl = eval_candidate(g, args.n_pos)
            scored.append((-kl if np.isfinite(kl) else -1e9, kl))
        scored.sort(key=lambda s: -s[0])
        order = sorted(range(len(scored)), key=lambda i: -scored[i][0])
        state.update(pop, order[: params.mu])
        best = min(best, scored[0][1])
        print(f"gen {gen:2d} best_kl={best:.4f}", flush=True)
        with open(r"d:\Documents\pySrc\.scratch\v7_genome.bin", "wb") as f:
            f.write(pop.candidates[:, order[0]].astype(np.float32).tobytes())
    print(f"RESULTADO v7: mejor KL = {best:.4f} (el genoma queda en v7_genome.bin)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
