"""Piloto de bloque real (Fase 7): evolución contra un profesor real de un GGUF.

Flujo end-to-end sobre un bloque FFN/proyección de un modelo GGUF real:
  1. `hayai dump_tensor_f32`  -> vuelca el profesor W0 (F32) del tensor elegido.
  2. Genera X de calibración (B=128).
  3. `saor-engine evolve --teacher-w/--teacher-x` -> evolución con decode GPU.
  4. `saor-engine consolidate` -> GGUF disperso del mejor candidato.
  5. `hayai plan` + `hayai load_saor_sparse` -> validación hayai (D_arch, SpMM).
  6. Reporte del contrato (D_arch, fidelidad CKA, sparsity).

Uso: python pilot_block.py --gguf <modelo.gguf> --tensor blk.0.ffn_gate.weight
      [--d-in N] [--d-out N] [--batch 128] [--gens 8] [--seed 42] [--workdir tmp]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from saor_orchestrator import SaorEngineClient


def sh(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} falló (rc={r.returncode}): {r.stderr[-2000:]}")
    return r.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="modelo GGUF real")
    ap.add_argument("--tensor", required=True, help="tensor profesor (ej. blk.0.ffn_gate.weight)")
    ap.add_argument("--d-in", type=int, default=None, help="d_in del bloque (auto si se omite)")
    ap.add_argument("--d-out", type=int, default=None, help="d_out del bloque (auto si se omite)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subspace", type=int, default=120)
    ap.add_argument("--workdir", default=os.path.join(os.environ.get("TEMP", "."), "saor_pilot"))
    args = ap.parse_args()

    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    engine = SaorEngineClient().engine_bin
    hayai_bin = r"d:\Documents\pySrc\hayai\target\release\hayai-cli.exe"

    # 1) Dimensiones del bloque (si no se dan) y volcado del profesor.
    from saor_orchestrator.hooks.gguf_audit import read_gguf_header

    h = read_gguf_header(args.gguf)
    tensors = {t.name: t for t in h.tensors}
    if args.tensor not in tensors:
        raise SystemExit(f"tensor '{args.tensor}' no existe en el GGUF")
    ti = tensors[args.tensor]
    d_in = args.d_in or int(ti.shape[0])
    d_out = args.d_out or int(ti.shape[1])
    print(f"[pilot] tensor={args.tensor} dims={list(ti.shape)} -> d_in={d_in} d_out={d_out}")

    w0_bin = wd / "w0.bin"
    sh(
        [
            "cmd", "/c",
            f"set PATH=C:\\msys64\\mingw64\\bin;C:\\msys64\\usr\\bin;%PATH% && "
            f"cd /d d:\\Documents\\pySrc\\hayai && "
            f"cargo run --release --example dump_tensor_f32 -- {args.gguf} {args.tensor} {w0_bin}",
        ]
    )
    x_bin = wd / "x.bin"
    rng = np.random.default_rng(args.seed)
    x = rng.normal(0, 1, (args.batch, d_in)).astype(np.float32)
    x.tofile(x_bin)

    # 2) Evolución contra el profesor real.
    evolve_out = wd / "evolve.json"
    sh(
        [
            engine, "evolve",
            "--d-in", str(d_in), "--d-out", str(d_out), "--batch", str(args.batch),
            "--gens", str(args.gens), "--seed", str(args.seed),
            "--subspace", str(args.subspace),
            "--teacher-w", str(w0_bin), "--teacher-x", str(x_bin),
            "--out", str(evolve_out),
        ]
    )
    report = json.loads(evolve_out.read_text())
    print("[pilot] evolución:", json.dumps(report["result"]))

    # 3) Consolidación en GGUF disperso.
    gguf_out = wd / "candidate.gguf"
    cons = sh(
        [
            engine, "consolidate",
            "--d-in", str(d_in), "--d-out", str(d_out), "--batch", str(args.batch),
            "--gens", str(args.gens), "--seed", str(args.seed),
            "--teacher-w", str(w0_bin), "--teacher-x", str(x_bin),
            "--out-gguf", str(gguf_out),
        ]
    )
    cons_rpt = json.loads(cons)
    print("[pilot] consolidación:", json.dumps(cons_rpt))

    # 4) Validación con hayai.
    plan_out = sh([hayai_bin, "plan", "--model", str(gguf_out)])
    ok_plan = "status: OK" in plan_out and "FfnDagAdjacency" in plan_out
    load_out = sh(
        [
            "cmd", "/c",
            f"set PATH=C:\\msys64\\mingw64\\bin;C:\\msys64\\usr\\bin;%PATH% && "
            f"cd /d d:\\Documents\\pySrc\\hayai && "
            f"cargo run --release --example load_saor_sparse -- {gguf_out}",
        ]
    )
    ok_load = "== OK: bloque saor cargado y SpMM validado ==" in load_out

    print("[pilot] hayai plan:", "OK" if ok_plan else "FALLO")
    print("[pilot] hayai load:", "OK" if ok_load else "FALLO")
    summary = {
        "tensor": args.tensor,
        "d_in": d_in,
        "d_out": d_out,
        "d_arch": cons_rpt.get("d_arch"),
        "best_cka": cons_rpt.get("best_cka"),
        "active": cons_rpt.get("active_connections"),
        "gguf": str(gguf_out),
        "hayai_plan_ok": ok_plan,
        "hayai_load_ok": ok_load,
    }
    (wd / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    print("[pilot] resumen:", json.dumps(summary, indent=2))
    return 0 if (ok_plan and ok_load) else 1


if __name__ == "__main__":
    sys.exit(main())
