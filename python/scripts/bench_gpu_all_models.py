# -*- coding: utf-8 -*-
"""Benchmark GPU (§5.2) de los 4 modelos Vía B: original vs. esparso Q4_K.

EJECUTAR SOLO DESPUÉS de que termine la evolución de Qwen3.8-27B (regla D34:
**un proceso OpenCL a la vez**; este script aborta si `via_b_evolve.py` sigue
en ejecución). Estrictamente secuencial por modelo:

  1. `dump_weights --blocks gate`   — solo si falta el dump (re-crea `w_alia`).
  2. `embed_sparse --genome --gpu`  — export **Q4_K por defecto** (skip si existe).
  3. `kl_eval`                      — KL y D_arch del GGUF esparso Q4 embebido.
  4. `bench_speed --device auto`    — original y esparso, en `Minimal` y `AutoFit`.

Uso:
    python python/scripts/bench_gpu_all_models.py [--device auto] [--n-pos 8]
        [--n-tokens 32] [--force] [--skip-embed] [--ignore-running]

Salidas: tabla en stdout + `.scratch/bench_gpu_results.csv` (para el informe §5.2).
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

TMP = Path(r"d:\Documents\pySrc\.scratch")
DUMP = Path(r"d:\Documents\pySrc\hayai\target\release\examples\dump_weights.exe")
EMBED = Path(r"d:\Documents\pySrc\saor\target\release\embed_sparse.exe")
BENCH = Path(r"d:\Documents\pySrc\hayai\target\release\examples\bench_speed.exe")
KL_EVAL = Path(r"d:\Documents\pySrc\hayai\target\release\examples\kl_eval.exe")
PROMPTS = TMP / "calib128.txt"
TAU = 0.42

# Configuración por modelo (paths verificado en disco el 2026-08-29).
# `dump`: directorio de `dump_weights --blocks gate` (pesos del gate).
# `genome`: genoma CPPN final (`via_b_best_genome_<name>.bin`).
MODELS = [
    {
        "name": "smol",
        "model": r"d:\Documents\pySrc\hayai\models\SmolLM2-135M-Instruct-Q4_K_M.gguf",
        "genome": "via_b_best_genome.bin",
        "dump": "w_smol",
    },
    {
        "name": "qwen35",
        "model": r"d:\Documents\pySrc\hayai\models\Qwen_Qwen3.5-4B-Q4_K_M.gguf",
        "genome": "via_b_best_genome_qwen35.bin",
        "dump": "w_qwen35",
    },
    {
        "name": "alia",
        "model": r"d:\Documents\pySrc\saor\models\ALIA-40b.Q4_K_M.gguf",
        "genome": "via_b_best_genome_alia.bin",
        "dump": "w_alia",  # dump BORRADO → se re-crea (48 gates, ~36 GB, ~20-40 min)
    },
    {
        "name": "qwen27",
        "model": r"d:\Documents\pySrc\saor\models\Qwen3.8-27B-UD-Q4_K_M.gguf",
        "genome": "via_b_best_genome_qwen27.bin",
        "dump": "w_qwen27",
    },
]


def sh(cmd: str, timeout: int) -> str:
    print(f"\n>>> {cmd}", flush=True)
    r = subprocess.run(["cmd", "/c", cmd], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"fallo: {cmd}\nstderr: {r.stderr[-1500:]}")
    if r.stdout.strip():
        print(r.stdout.strip(), flush=True)
    if r.stderr.strip():
        print(r.stderr.strip(), flush=True)
    return r.stdout


def via_b_running() -> bool:
    """Regla D34: ¿hay un `via_b_evolve.py` activo? (OpenCL ocupado)."""
    ps = (
        "Get-CimInstance Win32_Process | Where-Object { "
        "$_.CommandLine -match 'via_b_evolve' } | Select-Object -ExpandProperty ProcessId"
    )
    shells = [
        r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
        r"C:\Program Files\PowerShell\7\pwsh.exe",
        "powershell",
        "pwsh",
    ]
    for shell in shells:
        try:
            out = subprocess.run(
                [shell, "-NoProfile", "-Command", ps],
                capture_output=True, text=True, timeout=30,
            )
            return bool(out.stdout.strip())
        except (FileNotFoundError, OSError):
            continue
        except Exception as exc:  # noqa: BLE001 — ante la duda, no lanzar un 2º OpenCL
            print(f"aviso: no se pudo comprobar via_b_evolve ({exc}); trato como activo")
            return True
    print("aviso: sin PowerShell disponible; trato como activo (D34 seguro)")
    return True



def parse_kl(text: str) -> tuple[float, float]:
    start = text.find("{")
    data = json.loads(text[start:])
    return float(data["kl_global"]), float(data["d_arch_global"])


def parse_bench(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        for strategy in ("Minimal", "AutoFit"):
            if line.lstrip().startswith(f"{strategy}:"):
                out[strategy] = float(line.split()[1])
    return out


def bench_one(gguf: Path, device: str, n_tokens: int, label: str) -> dict[str, float]:
    print(f"\n=== bench {label}: {gguf.name} (device {device}) ===", flush=True)
    out = sh(
        f'"{BENCH}" --model "{gguf}" --prompts "{PROMPTS}" --n-tokens {n_tokens} --device {device}',
        3600,
    )
    return parse_bench(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto", help="device del bench y kl_eval (auto = OpenCL)")
    ap.add_argument("--n-pos", type=int, default=8, help="posiciones de kl_eval")
    ap.add_argument("--n-tokens", type=int, default=32, help="tokens del bench")
    ap.add_argument("--force", action="store_true", help="re-embeder aunque exista el GGUF esparso")
    ap.add_argument("--skip-embed", action="store_true", help="usar los GGUF esparsos existentes")
    ap.add_argument("--ignore-running", action="store_true", help="saltar el guard D34")
    args = ap.parse_args()

    for tool in (DUMP, EMBED, BENCH, KL_EVAL):
        if not tool.exists():
            sys.exit(f"falta el binario {tool} (¿cargo build --release?)")
    if not PROMPTS.exists():
        sys.exit(f"falta el corpus {PROMPTS}")

    if not args.ignore_running and via_b_running():
        sys.exit(
            "ABORTO (D34): via_b_evolve.py está en ejecución (GPU ocupada).\n"
            "Espera a que termine la evolución de Qwen3.8-27B o usa --ignore-running "
            "solo si sabes que no hay otro proceso OpenCL."
        )

    rows = []
    for cfg in MODELS:
        name = cfg["name"]
        print(f"\n{'=' * 72}\nMODELO: {name}\n{'=' * 72}", flush=True)
        model = Path(cfg["model"])
        genome = TMP / cfg["genome"]
        dump_dir = TMP / cfg["dump"]
        sparse = TMP / f"{name}_best_q4.gguf"
        if not model.exists():
            print(f"[SKIP] falta {model}", flush=True)
            continue
        if not genome.exists():
            print(f"[SKIP] falta {genome} (¿terminó la evolución de {name}?)", flush=True)
            continue

        # 1) Dump del gate (solo si falta meta.json — ALIA lo necesita).
        if not (dump_dir / "meta.json").exists():
            print(f"[1/4] dump_weights de {name} → {dump_dir.name} (solo gates)", flush=True)
            sh(f'"{DUMP}" --model "{model}" --out "{dump_dir}" --blocks gate', 7200)
        else:
            print(f"[1/4] dump existente: {dump_dir.name}", flush=True)

        # 2) Embed Q4 (skip si el GGUF esparso ya existe).
        if sparse.exists() and not args.force and not args.skip_embed:
            print(f"[2/4] GGUF esparso Q4 existente: {sparse.name}", flush=True)
        elif not args.skip_embed:
            print(f"[2/4] embed_sparse (Q4_K) → {sparse.name}", flush=True)
            sh(
                f'"{EMBED}" --model "{model}" --out "{sparse}" --weights "{dump_dir}" '
                f'--genome "{genome}" --tau {TAU:.4f} --gpu',
                7200,
            )
        else:
            print(f"[2/4] --skip-embed: se usa {sparse.name} (si existe)", flush=True)

        # 3) Validación KL del esparso Q4.
        print(f"[3/4] kl_eval (n_pos {args.n_pos})", flush=True)
        try:
            kl_out = sh(
                f'"{KL_EVAL}" --orig "{model}" --sparse "{sparse}" --prompts "{PROMPTS}" '
                f"--n-positions {args.n_pos} --device {args.device}",
                7200,
            )
            kl, d_arch = parse_kl(kl_out)
        except Exception as exc:  # noqa: BLE001 — la KL es informativa
            print(f"[3/4] kl_eval falló ({exc}); fila sin KL", flush=True)
            kl, d_arch = float("nan"), float("nan")

        # 4) Benchmark (original y esparso, ambas estrategias).
        print(f"[4/4] bench_speed ({args.n_tokens} tokens, device {args.device})", flush=True)
        orig_b = bench_one(model, args.device, args.n_tokens, f"{name} original")
        sparse_b = bench_one(sparse, args.device, args.n_tokens, f"{name} esparso Q4")

        size_orig = model.stat().st_size / 1e6
        size_sparse = sparse.stat().st_size / 1e6
        row = {
            "model": name,
            "orig_MB": round(size_orig, 1),
            "sparse_MB": round(size_sparse, 1),
            "d_arch": d_arch,
            "kl": kl,
            "orig_Minimal": orig_b.get("Minimal", float("nan")),
            "orig_AutoFit": orig_b.get("AutoFit", float("nan")),
            "sparse_Minimal": sparse_b.get("Minimal", float("nan")),
            "sparse_AutoFit": sparse_b.get("AutoFit", float("nan")),
        }
        rows.append(row)

    # Reporte
    print("\n" + "=" * 72)
    print("RESULTADOS §5.2 — velocidad en GPU (device: %s)" % args.device)
    print("=" * 72)
    hdr = f"{'modelo':<8}{'D_arch':>8}{'KL':>8}{'orig_MB':>10}{'sparse_MB':>10} | "
    hdr += f"{'orig Min':>9}{'orig Fit':>9} | {'esp Min':>9}{'esp Fit':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['model']:<8}{r['d_arch']:>8.3f}{r['kl']:>8.2f}"
            f"{r['orig_MB']:>10.0f}{r['sparse_MB']:>10.0f} | "
            f"{r['orig_Minimal']:>9.2f}{r['orig_AutoFit']:>9.2f} | "
            f"{r['sparse_Minimal']:>9.2f}{r['sparse_AutoFit']:>9.2f}"
        )
    print("(tokens/s; Min = MemoryStrategy::Minimal, Fit = AutoFit)")

    csv_path = TMP / "bench_gpu_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["model"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV guardado: {csv_path}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nfin ({time.time() - t0:.0f} s)", flush=True)
