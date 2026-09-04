"""Spike: valida el decode GPU directo a GGUF (--genome-v7) contra la vía
antigua (decode numpy + .bin + magnitude-prune 0.001) con el MISMO genoma:
ambas KL deben coincidir dentro de la tolerancia de la cuantización Q4_K.
Mide tiempos y confirma que la GPU se usa durante el embed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, r"d:\Documents\PySrc\saor\python")
from saor_orchestrator.reference.v7cppn import V7Cppn, decode_block  # noqa: E402

SMOL = r"d:\Documents\PySrc\hayai\models\SmolLM2-135M-Instruct-Q4_K_M.gguf"
EMBED = r"d:\Documents\PySrc\saor\target\release\embed_sparse.exe"
KLE = r"d:\Documents\PySrc\hayai\target\release\examples\kl_eval.exe"
P = r"d:\Documents\pySrc\.scratch\calib128.txt"
SCRATCH = r"d:\Documents\pySrc\.scratch"
N_LAYERS = 30
BLOCKS = [("ffn_gate", 1536, 576, "hi"), ("ffn_up", 1536, 576, "hi"),
          ("ffn_down", 576, 1536, "ih")]
TEACHER_CACHE = os.path.join(SCRATCH, "v7_teacher.bin")
W_OLD = os.path.join(SCRATCH, "w_v7_old")
W_META = os.path.join(SCRATCH, "w_v7_meta")
GENOME_BIN = os.path.join(SCRATCH, "v7_genome.bin")
EMB_OLD = os.path.join(SCRATCH, "v7_old.gguf")
EMB_NEW = os.path.join(SCRATCH, "v7_new.gguf")


def build_genome() -> np.ndarray:
    rng = np.random.default_rng(123)
    cppn0 = rng.normal(0.0, 0.3, 5138)
    geo = np.concatenate([np.linspace(-1, 1, 576), np.linspace(-1, 1, 1536)])
    return np.concatenate([cppn0, geo]).astype(np.float64)


def write_meta() -> None:
    os.makedirs(W_META, exist_ok=True)
    dims = {"n_layers": N_LAYERS}
    for layer in range(N_LAYERS):
        for block, d_out, d_in, _which in BLOCKS:
            dims[f"blk.{layer}.{block}"] = {"d_in": d_in, "d_out": d_out}
    with open(os.path.join(W_META, "meta.json"), "w") as f:
        json.dump(dims, f)
    # la vía antigua comparte el mismo dir de meta + escribe w.{...}.bin ahí
    os.makedirs(W_OLD, exist_ok=True)
    with open(os.path.join(W_OLD, "meta.json"), "w") as f:
        json.dump(dims, f)


def old_decode(flat64: np.ndarray) -> None:
    """Vía antigua: numpy decode_block → w.{layer}.{block}.bin + sp 0.001."""
    g = V7Cppn.from_flatten(flat64, hidden=48, d_h=576, d_i=1536)
    for f in os.listdir(W_OLD):
        os.remove(os.path.join(W_OLD, f))
    dims = {"n_layers": N_LAYERS}
    for layer in range(N_LAYERS):
        for block, d_out, d_in, _which in BLOCKS:
            dims[f"blk.{layer}.{block}"] = {"d_in": d_in, "d_out": d_out}
    with open(os.path.join(W_OLD, "meta.json"), "w") as f:
        json.dump(dims, f)
    sp = [0.001] * N_LAYERS
    with open(os.path.join(W_OLD, "sp.txt"), "w") as f:
        f.write("\n".join(f"{s:.4f}" for s in sp) + "\n")
    for layer in range(N_LAYERS):
        z = -1.0 + 2.0 * (layer + 0.5) / N_LAYERS
        for block, d_out, d_in, which in BLOCKS:
            W = decode_block(g, d_in, d_out, z, which, tau=0.5)
            with open(os.path.join(W_OLD, f"w.{layer}.{block}.bin"), "wb") as f:
                f.write(np.ascontiguousarray(W, np.float32).tobytes())


def embed_old() -> str:
    r = subprocess.run(
        [EMBED, "--model", SMOL, "--out", EMB_OLD, "--weights", W_OLD,
         "--sparsities", os.path.join(W_OLD, "sp.txt"), "--all-blocks"],
        capture_output=True, text=True, timeout=1800)
    return r.stdout


def embed_new() -> str:
    r = subprocess.run(
        [EMBED, "--model", SMOL, "--out", EMB_NEW, "--weights", W_META,
         "--genome-v7", GENOME_BIN, "--tau", "0.5", "--gpu", "--all-blocks"],
        capture_output=True, text=True, timeout=1800)
    return r.stdout


def kl(path: str) -> float:
    r = subprocess.run([KLE, "--orig", SMOL, "--sparse", path,
                        "--prompts", P, "--n-positions", "4",
                        "--device", "auto", "--teacher-cache", TEACHER_CACHE],
                       capture_output=True, text=True, timeout=1800)
    outs = [l for l in (r.stdout + r.stderr).splitlines() if "kl_global" in l]
    return float(json.loads(outs[-1])["kl_global"])


def main() -> int:
    write_meta()
    flat64 = build_genome()
    flat32 = flat64.astype(np.float32)
    with open(GENOME_BIN, "wb") as f:
        f.write(np.ascontiguousarray(flat32, np.float32).tobytes())

    t0 = time.time()
    old_decode(flat64)
    t_old_decode = time.time() - t0
    t0 = time.time()
    out_old = embed_old()
    t_old_embed = time.time() - t0

    t0 = time.time()
    out_new = embed_new()
    t_new_embed = time.time() - t0
    dev = [x for x in out_new.splitlines() if "decode por GPU" in x
           or '"ok":true' in x]
    kl_old = kl(EMB_OLD)
    kl_new = kl(EMB_NEW)
    print(f"decode numpy (vieja): {t_old_decode:.1f}s | embed vieja: {t_old_embed:.1f}s")
    print(f"embed GPU directa (nueva): {t_new_embed:.1f}s | {dev[-1][:120] if dev else out_new[:200]}")
    print(f"KL vieja (numpy+bin+prune): {kl_old:.4f}")
    print(f"KL nueva (GPU directa):     {kl_new:.4f}")
    print(f"delta = {abs(kl_new - kl_old):.4f}")
    ok = abs(kl_new - kl_old) < 0.10
    print("OK" if ok else "FALLO: delta alto")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
