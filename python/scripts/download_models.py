"""Descarga de los GGUFs objetivo (Fase 7.0): ALIA-40b y Qwen3.8-27B (Q4_K_M).

Uso: python download_models.py [--dir <destino>]
Reanudable (resume_download); imprime progreso por archivo.
"""

from __future__ import annotations

import argparse
import os
import sys

from huggingface_hub import hf_hub_download

DEFAULT_DIR = r"D:\Documents\pySrc\saor\models"

TARGETS = [
    ("mradermacher/ALIA-40b-GGUF", "ALIA-40b.Q4_K_M.gguf"),
    ("unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q4_K_M.gguf"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--only", default=None, help="subcadena del nombre de archivo")
    args = ap.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    for repo, fname in TARGETS:
        if args.only and args.only not in fname:
            continue
        print(f"[download] {repo}/{fname} ...", flush=True)
        try:
            hf_hub_download(
                repo_id=repo,
                filename=fname,
                local_dir=args.dir,
                resume_download=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[download] ERROR {fname}: {exc}", flush=True)
            return 1
        # Localizar el archivo descargado (ruta plana o con subdirectorio).
        for root, _dirs, files in os.walk(args.dir):
            if fname in files:
                full = os.path.join(root, fname)
                print(f"[download] OK {full} ({os.path.getsize(full) / 1e9:.1f} GB)", flush=True)
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
