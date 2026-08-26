"""Barrido de la frontera de Pareto (KL vs D_arch) sobre el path de producción.

Pipeline streaming (sin OpenCL, sin evaluador disk-backed):
  1. `dump_weights` (hayai)  -> pesos F32 del profesor (una vez).
  2. `embed_sparse` (saor-streamer) -> GGUF embebido D16 (poda por magnitud).
  3. `kl_eval` (hayai)        -> KL_global y D_arch_global (StreamingGenerator).

Uso:
  python python/scripts/frontier_stream_sweep.py [--sps 0.05 0.10 0.15 0.20]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

TMP = Path(r"d:\Documents\pySrc\.scratch")  # D: (rápido, ~190 GB) — no usar C:
MODEL = Path(r"d:\Documents\pySrc\hayai\models\SmolLM2-135M-Instruct-Q4_K_M.gguf")
DUMP = Path(r"d:\Documents\pySrc\hayai\target\release\examples\dump_weights.exe")
KL_EVAL = Path(r"d:\Documents\pySrc\hayai\target\release\examples\kl_eval.exe")
EMBED = Path(r"d:\Documents\pySrc\saor\target\release\embed_sparse.exe")
PROMPTS = TMP / "calib128.txt"
N_POS = 24
N_LAYERS = 30


def sh(cmd: str, timeout: int = 3600) -> str:
    r = subprocess.run(["cmd", "/c", cmd], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd}\n{r.stderr[-1500:]}")
    return r.stdout


def main() -> None:
    ap = argparse.ArgumentParser(description="Barrido de frontera en el path de producción")
    ap.add_argument("--sps", type=float, nargs="*", default=[0.05, 0.10, 0.15, 0.20, 0.25])
    ap.add_argument("--n-layers", type=int, default=N_LAYERS)
    ap.add_argument("--model", type=str, default=str(MODEL))
    ap.add_argument("--name", type=str, default="frontier")
    ap.add_argument("--blocks", type=str, default="all", help="'gate' para solo los gates")
    ap.add_argument("--n-positions", type=int, default=N_POS)
    args = ap.parse_args()

    model = Path(args.model)
    wdir = TMP / f"w_{args.name}"
    if not (wdir / "meta.json").exists():
        print(sh(f"{DUMP} --model {model} --out {wdir} --blocks {args.blocks}").strip(), flush=True)

    points = []
    for sp in args.sps:
        sp_file = wdir / f"sp_{int(sp*100)}.txt"
        sp_file.write_text("\n".join([f"{sp:.4f}"] * args.n_layers) + "\n")
        emb = TMP / f"{args.name}_sp{int(sp*100)}.gguf"
        print(f"[sp={sp:.2f}] embebiendo...", flush=True)
        print(sh(f"{EMBED} --model {model} --out {emb} --weights {wdir} "
                 f"--sparsities {sp_file}").strip(), flush=True)
        print(f"[sp={sp:.2f}] evaluando...", flush=True)
        out = sh(f"{KL_EVAL} --orig {model} --sparse {emb} --prompts {PROMPTS} "
                 f"--n-positions {args.n_positions} --device cpu")
        r = json.loads(out.strip())
        points.append({"sp": sp, **r})
        print(f"  -> KL={r['kl_global']:.4f} D_arch={r['d_arch_global']:.4f}", flush=True)
        emb.unlink(missing_ok=True)

    print("\n=== FRONTERA (KL vs D_arch) ===")
    for p in points:
        print(f"  sp={p['sp']:.2f}  D_arch={p['d_arch_global']:.4f}  KL={p['kl_global']:.4f}")
    (TMP / f"{args.name}_points.json").write_text(json.dumps(points, indent=2))


if __name__ == "__main__":
    main()
