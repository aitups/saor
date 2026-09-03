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


def fit_cppn_slice(w: np.ndarray, hidden: int = 128, seed: int = 1) -> np.ndarray:
    """Mejor aproximacion suave de W via ELM sobre la geometria (i,j)."""
    d_out, d_in = w.shape
    xi = np.linspace(-1, 1, d_in, dtype=np.float32)
    xj = np.linspace(-1, 1, d_out, dtype=np.float32)
    Xj, Xi = np.meshgrid(xj, xi, indexing="ij")
    feat = np.stack([Xi, Xj, Xj - Xi], axis=-1).reshape(-1, 3).astype(np.float32)
    rng = np.random.default_rng(seed)
    W0 = rng.normal(0, 1.0, (hidden, 3)).astype(np.float32)
    b0 = rng.normal(0, 0.5, hidden).astype(np.float32)
    H = np.tanh(feat @ W0.T + b0)
    coef, *_ = np.linalg.lstsq(H, w.reshape(-1), rcond=None)
    recon = (H @ coef).reshape(d_out, d_in).astype(np.float32)
    return recon


def main() -> int:
    os.makedirs(W_APPROX, exist_ok=True)
    # Solo la capa 0 reemplazada para el primer dato (impacto incremental).
    w0 = np.fromfile(os.path.join(W_TEACH, "w.0.ffn_gate.bin"), np.float32).reshape(1536, 576)
    approx = fit_cppn_slice(w0)
    with open(os.path.join(W_APPROX, "w.0.ffn_gate.bin"), "wb") as f:
        f.write(approx.astype(np.float32).tobytes())
    # meta.json (dims del smol, 30 capas).
    meta = {"n_layers": 30}
    for layer in range(30):
        meta[f"blk.{layer}.ffn_gate"] = {"d_in": 576, "d_out": 1536}
        meta[f"blk.{layer}.ffn_up"] = {"d_in": 576, "d_out": 1536}
        meta[f"blk.{layer}.ffn_down"] = {"d_in": 1536, "d_out": 576}
    with open(os.path.join(W_APPROX, "meta.json"), "w") as f:
        json.dump(meta, f)
    # CKA del warm-up (fidelidad de la replica).
    tvar = float(np.var(w0))
    mse = float(np.mean((approx - w0) ** 2))
    print(f"warm-up capa 0: CKA-fidelidad replica = {1 - mse/tvar:.4f} "
          f"(1 = replica perfecta)", flush=True)
    # Sparsities: capa 0 -> 0.001 (casi densa), resto -> 0.
    spf = os.path.join(r"d:\Documents\pySrc\.scratch", "smol_warm_sp.txt")
    with open(spf, "w") as f:
        f.write("\n".join([f"{0.001:.4f}"] + ["0.0000"] * 29) + "\n")
    emb = r"d:\Documents\pySrc\.scratch\smol_warm0.gguf"
    if os.path.exists(emb):
        os.remove(emb)
    r = subprocess.run([EMBED, "--model", SMOL, "--out", emb, "--weights", W_APPROX,
                        "--sparsities", spf], capture_output=True, text=True)
    print(f"embed: {r.stdout.strip()[-80:] if 'ok' in r.stdout else r.stderr[-150:]}", flush=True)
    r = subprocess.run([KLE, "--orig", SMOL, "--sparse", emb, "--prompts", P,
                        "--n-positions", "8", "--device", "auto"],
                       capture_output=True, text=True, timeout=1800)
    outs = [l for l in (r.stdout + r.stderr).splitlines() if "kl_global" in l]
    print(f"KL gen-0 (solo gate0 = aprox CPPN): {outs[-1] if outs else 'ERR ' + r.stderr[-150:]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
