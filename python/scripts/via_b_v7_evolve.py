"""Vía B-v7 — evolución con la KL global (hayai) como ÚNICO fitness y decode
GPU del genoma v7 DIRECTO a GGUF (sin decode numpy en CPU ni .bin intermedios).

El genoma (CPPN generadora de pesos + geometría aprendida por canal, 7250 f32)
se inicializa con ruido aceptado (corrección vía 2) y CMA-ES lo evoluciona con
la KL global medida por el motor: `embed_sparse --genome-v7 --gpu` decodifica
cada candidato en la GPU (kernel `cppn_decode_v7`, kernel OpenCL) y lo embebe
en el GGUF; `kl_eval --teacher-cache` mide la divergencia.

Genoma = [CPPN 48h] + [coord_h 576] + [coord_i 1536] (ver v7cppn.py).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

import numpy as np

import sys
sys.path.insert(0, r"d:\Documents\PySrc\saor\python")
from saor_orchestrator.reference.cmaes import CmaEsParams, CmaEsState  # noqa: E402
from saor_orchestrator.reference.v7cppn import V7Cppn  # noqa: E402

SMOL = r"d:\Documents\PySrc\hayai\models\SmolLM2-135M-Instruct-Q4_K_M.gguf"
EMBED = r"d:\Documents\PySrc\saor\target\release\embed_sparse.exe"
KLE = r"d:\Documents\PySrc\hayai\target\release\examples\kl_eval.exe"
P = r"d:\Documents\pySrc\.scratch\calib128.txt"
SCRATCH = r"d:\Documents\pySrc\.scratch"
W_DIR = os.path.join(SCRATCH, "w_v7_meta")
N_LAYERS = 30
D_H, D_I = 576, 1536
BLOCKS = [("ffn_gate", 1536, 576, "hi"), ("ffn_up", 1536, 576, "hi"),
          ("ffn_down", 576, 1536, "ih")]
TEACHER_CACHE = r"d:\Documents\pySrc\.scratch\v7_teacher.bin"
EMB_GGUF = os.path.join(SCRATCH, "v7_cand.gguf")
GENOME_BIN = os.path.join(SCRATCH, "v7_genome.bin")


def ensure_meta() -> None:
    """Escribe una vez el meta.json de dimensiones (invariante por candidato)."""
    if os.path.exists(os.path.join(W_DIR, "meta.json")):
        return
    os.makedirs(W_DIR, exist_ok=True)
    dims = {"n_layers": N_LAYERS}
    for layer in range(N_LAYERS):
        for block, d_out, d_in, _which in BLOCKS:
            dims[f"blk.{layer}.{block}"] = {"d_in": d_in, "d_out": d_out}
    with open(os.path.join(W_DIR, "meta.json"), "w") as f:
        json.dump(dims, f)


def kl_fitness(flat: np.ndarray, n_pos: int) -> float:
    """Decode GPU (kernel cppn_decode_v7) + embed directo + KL global."""
    with open(GENOME_BIN, "wb") as f:
        f.write(np.ascontiguousarray(flat, np.float32).tobytes())
    if os.path.exists(EMB_GGUF):
        os.remove(EMB_GGUF)
    r = subprocess.run(
        [EMBED, "--model", SMOL, "--out", EMB_GGUF, "--weights", W_DIR,
         "--genome-v7", GENOME_BIN, "--tau", "0.5", "--gpu", "--all-blocks"],
        capture_output=True, text=True, timeout=1800)
    if '"ok":true' not in r.stdout:
        return float("inf")
    r = subprocess.run([KLE, "--orig", SMOL, "--sparse", EMB_GGUF,
                        "--prompts", P, "--n-positions", str(n_pos),
                        "--device", "auto", "--teacher-cache", TEACHER_CACHE],
                       capture_output=True, text=True, timeout=1800)
    outs = [l for l in (r.stdout + r.stderr).splitlines() if "kl_global" in l]
    if not outs:
        return float("inf")
    return float(json.loads(outs[-1])["kl_global"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-pos", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--lambda-mult", type=float, default=1.0)
    ap.add_argument("--sigma0", type=float, default=0.15)
    args = ap.parse_args()

    ensure_meta()
    cppn_n = V7Cppn(hidden=args.hidden, d_h=D_H, d_i=D_I).cppn_n
    geo0 = np.concatenate([
        np.linspace(-1, 1, D_H, dtype=np.float64),
        np.linspace(-1, 1, D_I, dtype=np.float64)])
    rng = np.random.default_rng(123)
    cppn0 = rng.normal(0.0, 0.3, cppn_n)
    mean0 = np.concatenate([cppn0, geo0])
    params = CmaEsParams(len(mean0), args.seed, sigma0=args.sigma0)
    params.lambda_ = max(8, int(params.lambda_ * args.lambda_mult))
    params.mu = params.lambda_ // 2

    state = CmaEsState(params, mean0)
    print(f"v7: genoma {len(mean0)} floats, lambda={params.lambda_}, fitness=KL (decode GPU)", flush=True)
    best = float("inf")
    for gen in range(args.gens):
        pop = state.spawn_population(args.seed + gen)
        scored = []
        for c in range(pop.candidates.shape[1]):
            kl = kl_fitness(pop.candidates[:, c], args.n_pos)
            scored.append((-kl if np.isfinite(kl) else -1e9, kl))
        scored.sort(key=lambda s: -s[0])
        order = sorted(range(len(scored)), key=lambda i: -scored[i][0])
        state.update(pop, order[: params.mu])
        best = min(best, scored[0][1])
        print(f"gen {gen:2d} best_kl={best:.4f}", flush=True)
        with open(GENOME_BIN, "wb") as f:
            f.write(pop.candidates[:, order[0]].astype(np.float32).tobytes())
    print(f"RESULTADO v7: mejor KL = {best:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
