"""Warm-up gen-0 (v7-b): regresion de la CPPN sobre el gate real del profesor +
inyeccion densa + KL. Mide si el warm-up deja KL ~ 0 (premisa de la v7).

Pipeline:
1. Regresion ELM (la mejor replica que una CPPN geometrica puede pintar) del
   gate 0 del smol -> W_approx.
2. Escribe W_approx como dump w.{layer}.ffn_gate.bin (para el embed).
3. `embed_sparse --sparsities sp=0.001` con ese dump -> D16 all-active con los
   pesos de la CPPN (gen-0 densa del warm-up, solo la capa 0 reemplazada).
4. `kl_eval` -> el KL global real.
"""
from __future__ import annotations

import subprocess
import json
import os

import numpy as np

SMOL = r"d:\Documents\PySrc\hayai\models\SmolLM2-135M-Instruct-Q4_K_M.gguf"
W_TEACH = r"d:\Documents\pySrc\.scratch\w_smol2"
W_APPROX = r"d:\Documents\pySrc\.scratch\w_smol_warm"
EMBED = r"d:\Documents\PySrc\saor\target\release\embed_sparse.exe"
KLE = r"d:\Documents\PySrc\hayai\target\release\examples\kl_eval.exe"
P = r"d:\Documents\pySrc\.scratch\calib128.txt"


def fit_cppn_slice(w: np.ndarray, hidden: int = 96, seed: int = 1,
                   n_sample: int = 30000) -> np.ndarray:
    """Aproximacion suave de W: ELM ajustado sobre una muestra de conexiones."""
    d_out, d_in = w.shape
    xi = np.linspace(-1, 1, d_in, dtype=np.float32)
    xj = np.linspace(-1, 1, d_out, dtype=np.float32)
    rng = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed + 1)
    W0 = rng.normal(0, 1.0, (hidden, 3)).astype(np.float32)
    b0 = rng.normal(0, 0.5, hidden).astype(np.float32)
    js = rng2.integers(0, d_out, n_sample)
    is_ = rng2.integers(0, d_in, n_sample)
    feat = np.stack([xi[is_], xj[js], xj[js] - xi[is_]], axis=-1).astype(np.float32)
    H = np.tanh(feat @ W0.T + b0)
    coef, *_ = np.linalg.lstsq(H, w[js, is_], rcond=None)
    # Aplicar a toda la rejilla.
    Xj, Xi = np.meshgrid(xj, xi, indexing="ij")
    F = np.stack([Xi, Xj, Xj - Xi], axis=-1).reshape(-1, 3).astype(np.float32)
    HF = np.tanh(F @ W0.T + b0)
    return (HF @ coef).reshape(d_out, d_in).astype(np.float32)


def main() -> int:
    import shutil
    os.makedirs(W_APPROX, exist_ok=True)
    for f in os.listdir(W_APPROX):
        os.remove(os.path.join(W_APPROX, f))
    dims = {}
    for layer in range(30):
        for block, (d_out, d_in) in {
                "ffn_gate": (1536, 576), "ffn_up": (1536, 576),
                "ffn_down": (576, 1536)}.items():
            wt = os.path.join(W_TEACH, f"w.{layer}.{block}.bin")
            if not os.path.exists(wt):
                continue
            w = np.fromfile(wt, np.float32).reshape(d_out, d_in)
            approx = fit_cppn_slice(w, seed=layer * 3 + (0 if block == "ffn_gate" else 1))
            with open(os.path.join(W_APPROX, f"w.{layer}.{block}.bin"), "wb") as f:
                f.write(approx.astype(np.float32).tobytes())
            dims[f"blk.{layer}.{block}"] = {"d_in": d_in, "d_out": d_out}
    meta = {"n_layers": 30, **dims}
    with open(os.path.join(W_APPROX, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"warm-up: {len(dims)} matrices del FFN completo (30 capas) aproximadas", flush=True)
    # Sparsities: todas las capas a 0.001 (gen-0 casi densa).
    spf = os.path.join(r"d:\Documents\pySrc\.scratch", "smol_warm_sp.txt")
    with open(spf, "w") as f:
        f.write("\n".join([f"{0.001:.4f}"] * 30) + "\n")
    emb = r"d:\Documents\pySrc\.scratch\smol_warm_full.gguf"
    if os.path.exists(emb):
        os.remove(emb)
    r = subprocess.run([EMBED, "--model", SMOL, "--out", emb, "--weights", W_APPROX,
                        "--sparsities", spf, "--all-blocks"], capture_output=True, text=True)
    print(f"embed: {r.stdout.strip()[-80:] if 'ok' in r.stdout else r.stderr[-150:]}", flush=True)
    r = subprocess.run([KLE, "--orig", SMOL, "--sparse", emb, "--prompts", P,
                        "--n-positions", "8", "--device", "auto"],
                       capture_output=True, text=True, timeout=1800)
    outs = [l for l in (r.stdout + r.stderr).splitlines() if "kl_global" in l]
    print(f"KL gen-0 (FFN completo reescrito por la CPPN): {outs[-1] if outs else 'ERR ' + r.stderr[-150:]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
