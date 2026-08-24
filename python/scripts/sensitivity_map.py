"""Fase 0 (D20): mapa de sensibilidad por capa de SmolLM2.

Mide KL_global al esparsificar el gate de CADA capa individualmente (poda por
magnitud) a esparsidades fijas. Identifica qué capas toleran esparsidad alta
(heterogeneidad) y cuantifica la factibilidad de D_arch_global 0.40 @ KL<=0.50.
"""
import json
import subprocess
import sys

TMP = "C:/Users/epoke/AppData/Local/Temp"
MODEL = "d:\\Documents\\pySrc\\hayai\\models\\SmolLM2-135M-Instruct-Q4_K_M.gguf"
EVAL = "d:\\Documents\\pySrc\\hayai\\target\\release\\examples\\eval_sparse.exe"
PROMPTS = f"{TMP}/calib128.txt"
N_LAYERS = 30


def sh(cmd):
    r = subprocess.run(["cmd", "/c", cmd], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd}\n{r.stderr[-1200:]}")
    return r.stdout


def eval_sp(sps: list[float], n_pos: int = 64) -> dict:
    path = f"{TMP}/sp_sweep.txt"
    with open(path, "w") as f:
        f.write("\n".join(f"{s:.4f}" for s in sps) + "\n")
    out = sh(
        f'{EVAL} --model {MODEL} --prompts {PROMPTS} --sparsities {path} '
        f"--n-positions {n_pos} --device cpu"
    )
    return json.loads(out.strip())


def main():
    print(f"{'capa':>4} | {'KL@0.4':>8} | {'KL@0.6':>8}")
    results = []
    for layer in range(N_LAYERS):
        sps = [0.0] * N_LAYERS
        sps[layer] = 0.4
        kl4 = eval_sp(sps)["kl_global"]
        sps[layer] = 0.6
        kl6 = eval_sp(sps)["kl_global"]
        results.append((layer, kl4, kl6))
        print(f"{layer:4d} | {kl4:8.4f} | {kl6:8.4f}")

    total4 = sum(r[1] for r in results)
    total6 = sum(r[2] for r in results)
    print(f"\nSuma KL individual @0.4 = {total4:.3f} | @0.6 = {total6:.3f}")
    # Mejores candidatos a esparsificar (menor KL) y peores (más sensibles).
    by4 = sorted(results, key=lambda r: r[1])
    print("Top 5 menos sensibles (KL@0.4 menor):", [r[0] for r in by4[:5]])
    print("Top 5 más sensibles (KL@0.4 mayor):   ", [r[0] for r in by4[-5:]])
    with open(f"{TMP}/sens_map.json", "w") as f:
        json.dump([{"layer": l, "kl_04": a, "kl_06": b} for l, a, b in results], f, indent=2)


if __name__ == "__main__":
    main()
