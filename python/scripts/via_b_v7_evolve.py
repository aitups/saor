"""Vía B-v7 — bucle evolutivo con GEOMETRÍA APRENDIDA + WARM-UP por regresión.

Fase warm-up: CMA-ES minimiza ||W_decode(genoma) - W_profesor||² sobre una
muestra de conexiones (inicializa el genoma para que pinte una replica del
profesor; el ruido residual es aceptado por diseño).
Fase evolución: CMA-ES con fitness = KL global (el motor, la inyección D16
--all-blocks) desde el genoma calentado.

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
W_TEACH = r"d:\Documents\pySrc\.scratch\w_smol2"
W_GEN = r"d:\Documents\pySrc\.scratch\w_v7_gen"
EMBED = r"d:\Documents\PySrc\saor\target\release\embed_sparse.exe"
KLE = r"d:\Documents\PySrc\hayai\target\release\examples\kl_eval.exe"
P = r"d:\Documents\pySrc\.scratch\calib128.txt"
N_LAYERS = 30
D_H, D_I = 576, 1536
BLOCKS = [("ffn_gate", 1536, 576, "hi"), ("ffn_up", 1536, 576, "hi"),
          ("ffn_down", 576, 1536, "ih")]


def teacher_matrices() -> dict:
    mats = {}
    for layer in range(N_LAYERS):
        for block, d_out, d_in, _ in BLOCKS:
            p = os.path.join(W_TEACH, f"w.{layer}.{block}.bin")
            mats[(layer, block)] = np.fromfile(p, np.float32).reshape(d_out, d_in)
    return mats


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
                        "--teacher-cache", r"d:\Documents\pySrc\.scratch\v7_teacher.bin"],
                       capture_output=True, text=True, timeout=1800)
    outs = [l for l in (r.stdout + r.stderr).splitlines() if "kl_global" in l]
    if not outs:
        return float("inf")
    return float(json.loads(outs[-1])["kl_global"])

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup-gens", type=int, default=25)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-pos", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--lambda-mult", type=float, default=1.0)
    args = ap.parse_args()

    mats = teacher_matrices()
    rng = np.random.default_rng(args.seed)
    sample_idx, sample_w = {}, {}
    for (layer, block), W in mats.items():
        d_out, d_in = W.shape
        n_s = min(6000, d_out * d_in // 2)
        idx = rng.choice(d_out * d_in, n_s, replace=False)
        sample_idx[(layer, block)] = idx
        sample_w[(layer, block)] = W.ravel()[idx].astype(np.float32)
    geo0 = np.concatenate([
        np.linspace(-1, 1, D_H, dtype=np.float64),
        np.linspace(-1, 1, D_I, dtype=np.float64)])

    def build(flat):
        return V7Cppn.from_flatten(flat, args.hidden, d_h=D_H, d_i=D_I)

    cppn0 = np.random.default_rng(args.seed + 1).standard_normal(
        V7Cppn(hidden=args.hidden).cppn_n).astype(np.float64) * 0.3
    mean0 = np.concatenate([cppn0, geo0])
    params = CmaEsParams(len(mean0), args.seed)
    params.lambda_ = max(10, int(params.lambda_ * args.lambda_mult))
    params.mu = params.lambda_ // 2

    print(f"warm-up: genoma {len(mean0)} floats, lambda={params.lambda_}", flush=True)
    state = CmaEsState(params, mean0)
    for gen in range(args.warmup_gens):
        pop = state.spawn_population(args.seed + gen)
        scored = []
        for c in range(pop.candidates.shape[1]):
            g = build(pop.candidates[:, c])
            tot = 0.0
            cnt = 0
            for (layer, block), W in mats.items():
                d_out, d_in = W.shape
                which = "hi" if block != "ffn_down" else "ih"
                z = -1.0 + 2.0 * (layer + 0.5) / N_LAYERS
                w = decode_block(g, d_in, d_out, z, which, args.tau,
                                 idx_sample=sample_idx[(layer, block)])[0]
                tot += float(np.mean((w - sample_w[(layer, block)]) ** 2))
                cnt += 1
            scored.append((-tot / cnt, tot / cnt))
        scored.sort(key=lambda s: -s[0])
        order = sorted(range(len(scored)), key=lambda i: -scored[i][0])
        state.update(pop, order[: params.mu])
        if gen % 5 == 0 or gen == args.warmup_gens - 1:
            print(f"  warm gen {gen:2d} mse_medio={scored[0][1]:.5f}", flush=True)
    warm_mean = state.mean.astype(np.float32)
    with open(r"d:\Documents\pySrc\.scratch\v7_warmup_genome.bin", "wb") as f:
        f.write(warm_mean.tobytes())
    print("warm-up completado, genoma guardado", flush=True)

    state = CmaEsState(params, warm_mean.astype(np.float64))
    best = float("inf")
    for gen in range(args.gens):
        pop = state.spawn_population(args.seed + 1000 + gen)
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
