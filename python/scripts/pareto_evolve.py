"""Vía A (D20): evolución global CMA-ES bajo Frontera de Pareto.

Maximiza D_arch_global sujeto a KL_global <= 0.50. Variable: esparsidad por
capa (logit, mapeada por sigmoide). Evaluador: `eval_sparse` de hayai (poda por
magnitud del gate del profesor en lockstep original/candidato).

Fitness: D_arch - lambda_pen * max(0, KL - 0.50)   (penalización de restricción).
"""
import json
import subprocess
import sys

import numpy as np

sys.path.insert(0, r"d:\Documents\pySrc\saor\python")
from saor_orchestrator.reference.cmaes import CmaEsParams, CmaEsState

TMP = "C:/Users/epoke/AppData/Local/Temp"
MODEL = "d:\\Documents\\pySrc\\hayai\\models\\SmolLM2-135M-Instruct-Q4_K_M.gguf"
EVAL = "d:\\Documents\\pySrc\\hayai\\target\\release\\examples\\eval_sparse.exe"
PROMPTS = f"{TMP}/calib128.txt"
N_LAYERS = 30
KL_MAX = 0.50
LAMBDA_PEN = 2.0
N_POS = 24
GENS = 12
SEED = 7


def sh(cmd):
    r = subprocess.run(["cmd", "/c", cmd], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd}\n{r.stderr[-1200:]}")
    return r.stdout


def evaluate(sps: list[float]) -> dict:
    path = f"{TMP}/sp_candidate.txt"
    with open(path, "w") as f:
        f.write("\n".join(f"{s:.4f}" for s in sps) + "\n")
    out = sh(
        f'{EVAL} --model {MODEL} --prompts {PROMPTS} --sparsities {path} '
        f"--n-positions {N_POS} --device cpu"
    )
    return json.loads(out.strip())


def sp_from_z(z: np.ndarray) -> list[float]:
    return (1.0 / (1.0 + np.exp(-z)) * 0.95).tolist()


def main():
    params = CmaEsParams(N_LAYERS, SEED)
    mean0 = np.full(N_LAYERS, -2.2)  # sigmoid(-2.2) ~ 0.1
    state = CmaEsState(params, mean0)
    history = []
    best_feasible = {"d_arch": -1.0, "kl": None, "sps": None}
    best_darch = {"d_arch": -1.0, "kl": None, "sps": None}

    for gen in range(GENS):
        pop = state.spawn_population(SEED + gen)
        scored = []
        for col in range(pop.candidates.shape[1]):
            z = pop.candidates[:, col]
            sps = sp_from_z(z)
            r = evaluate(sps)
            d_arch = r["d_arch_global"]
            kl = r["kl_global"]
            fitness = d_arch - LAMBDA_PEN * max(0.0, kl - KL_MAX)
            scored.append((fitness, d_arch, kl, sps))
            if kl <= KL_MAX and d_arch > best_feasible["d_arch"]:
                best_feasible = {"d_arch": d_arch, "kl": kl, "sps": sps}
            if d_arch > best_darch["d_arch"]:
                best_darch = {"d_arch": d_arch, "kl": kl, "sps": sps}
        scored.sort(key=lambda s: -s[0])
        order = sorted(range(len(scored)), key=lambda i: -scored[i][0])
        elite = order[: params.mu]
        state.update(pop, elite)
        gen_best = scored[0]
        history.append(
            {
                "gen": gen,
                "best_fitness": round(gen_best[0], 4),
                "best_darch": round(gen_best[1], 4),
                "best_kl": round(gen_best[2], 4),
                "best_feasible_darch": round(best_feasible["d_arch"], 4),
                "best_feasible_kl": (
                    round(best_feasible["kl"], 4)
                    if best_feasible["kl"] is not None
                    else None
                ),
            }
        )
        print(
            f"gen {gen:2d} best_fit={gen_best[0]:.4f} darch={gen_best[1]:.3f} "
            f"kl={gen_best[2]:.3f} | factible: darch={best_feasible['d_arch']:.3f} "
            f"kl={best_feasible['kl'] if best_feasible['kl'] is not None else float('nan'):.3f}",
            flush=True,
        )

    print("\n=== Mejor FACTIBLE (KL <= 0.50) ===")
    if best_feasible["sps"]:
        print(f"D_arch={best_feasible['d_arch']:.4f} KL={best_feasible['kl']:.4f}")
        for l, s in enumerate(best_feasible["sps"]):
            if s > 0.02:
                print(f"  capa {l}: sp={s:.3f}")
    else:
        print("  (ninguna configuracion con KL <= 0.50 encontrada)")
    print("\n=== Mejor D_arch (sin restriccion) ===")
    print(f"D_arch={best_darch['d_arch']:.4f} KL={best_darch['kl']:.4f}")

    with open(f"{TMP}/pareto_history.json", "w") as f:
        json.dump({"history": history, "best_feasible": best_feasible, "best_darch": best_darch}, f, indent=2)


if __name__ == "__main__":
    main()
