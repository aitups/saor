"""Vía B (D21): evolución global CMA-ES bajo Frontera de Pareto con CPPN global.

UN solo genoma CPPN (sustrato v5, con coordenada de capa `y_layer`) genera la
topología de TODAS las capas. Maximiza `D_arch_global` sujeto a `KL_global <= 0.50`.

Variables CMA-ES:
  z[0:466] = genoma CPPN aplanado (fila-mayor, layout del kernel OpenCL).

Formulación por nivel de frontera: fijar D_arch global objetivo (`--darch`),
rho = 1 - darch, y el CPPN evoluciona el PERFIL por capa para minimizar KL_global.
Barriendo `--darch` se traza la frontera de Pareto (KL_global <= 0.50).

Decodificación por candidato:
  genome.decode_global(d_in, d_out, tau_fijo, n_layers, dense_density=rho, step=4)
    -> densidades por capa (perfil del CPPN reescalado a media rho)
    -> sparsities = 1 - densidades  ->  eval_sparse (hayai, poda por magnitud).

Uso:
  python python/scripts/via_b_evolve.py --darch 0.10 --gens 12
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"d:\Documents\pySrc\saor\python")
from saor_orchestrator.reference.cmaes import CmaEsParams, CmaEsState  # noqa: E402
from saor_orchestrator.reference.cppn import CppnGenome  # noqa: E402

TMP = r"d:\Documents\pySrc\.scratch"  # D: (rápido, ~190 GB) — no usar C:
MODEL = r"d:\Documents\pySrc\hayai\models\SmolLM2-135M-Instruct-Q4_K_M.gguf"
EVAL = r"d:\Documents\pySrc\hayai\target\release\examples\eval_sparse.exe"
PROMPTS = f"{TMP}/calib128.txt"
N_LAYERS = 30
D_IN, D_OUT = 576, 1536
TAU_CPPN = 0.42
KL_MAX = 0.50
LAMBDA_PEN = 2.0
N_POS = 24
SP_CAP = 0.95  # esparsidad máxima por capa (eval_sparse)
# Prefijo de las salidas (via_b_best_genome_{NAME}.bin / via_b_history_{NAME}.json)
# para no pisar resultados entre modelos.
NAME = "smol"
# Device del `kl_eval` del path streaming ("auto" = OpenCL si hay GPU, "cpu").
# ALIA-40b en la RTX 4050 (6 GB) puede requerir "cpu" si el path GPU no cabe.
KL_DEVICE = "auto"
# Decode de la topología CPPN en la GPU (embed_sparse --gpu). Se activa con
# `--gpu`; el embed falla con un mensaje claro si no hay dispositivo OpenCL.
GPU_EMBED = False


def sh(cmd: str, timeout: int = 600) -> str:
    r = subprocess.run(["cmd", "/c", cmd], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd}\n{r.stderr[-1200:]}")
    return r.stdout


def random_genome(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g = CppnGenome()
    g.w0 = rng.standard_normal(g.w0.shape).astype(np.float32) * 0.8
    g.b0 = rng.standard_normal(g.b0.shape).astype(np.float32) * 0.3
    g.w1 = rng.standard_normal(g.w1.shape).astype(np.float32) * 0.5
    g.b1 = rng.standard_normal(g.b1.shape).astype(np.float32) * 0.2
    g.w2 = rng.standard_normal(g.w2.shape).astype(np.float32) * 0.5
    g.b2 = rng.standard_normal(g.b2.shape).astype(np.float32) * 0.2
    return g.flatten()


def decode_sparsities(z: np.ndarray, rho: float, step: int = 4) -> list[float]:
    """Perfil por capa del CPPN global reescalado a densidad media rho.

    `step > 1` submuestrea el sustrato (estimador de densidad rápido para el
    loop CMA-ES); la topología exacta se decodifica después con step=1.
    """
    genome = CppnGenome.from_flatten(z[:466].astype(np.float32))
    densities, _ = genome.decode_global(
        D_IN, D_OUT, TAU_CPPN, N_LAYERS, dense_density=rho, step=step
    )
    return [float(np.clip(1.0 - d, 0.0, SP_CAP)) for d in densities]


EMBED = Path(r"d:\Documents\pySrc\saor\target\release\embed_sparse.exe")
KL_EVAL = Path(r"d:\Documents\pySrc\hayai\target\release\examples\kl_eval.exe")
DUMP_WEIGHTS = Path(r"d:\Documents\pySrc\hayai\target\release\examples\dump_weights.exe")
W_DIR = Path(r"d:\Documents\pySrc\.scratch\w_frontier")  # dump de pesos (una vez)


def evaluate(
    sparsities: list[float],
    genome_z: np.ndarray | None = None,
    tau: float = TAU_CPPN,
    streaming: bool = False,
    n_pos: int = N_POS,
) -> dict:
    """Evalúa un candidato. `genome_z=None` → poda por magnitud (densidades);
    con genoma → topología CPPN real (`eval_sparse --genome` o `embed_sparse
    --genome` + `kl_eval` en el path de producción con `--streaming`)."""
    if genome_z is None:
        path = f"{TMP}/vib_sp_candidate.txt"
        with open(path, "w") as f:
            f.write("\n".join(f"{s:.4f}" for s in sparsities) + "\n")
        out = sh(
            f'{EVAL} --model {MODEL} --prompts {PROMPTS} --sparsities {path} '
            f"--n-positions {n_pos} --device cpu"
        )
    elif streaming:
        # Path de producción: CPPN → embed D16 → kl_eval (StreamingGenerator).
        gpath = f"{TMP}/vib_genome.bin"
        with open(gpath, "wb") as f:
            f.write(np.asarray(genome_z, np.float32).tobytes())
        emb = f"{TMP}/vib_candidate.gguf"
        gpu_flag = " --gpu" if GPU_EMBED else ""
        sh(
            f"{EMBED} --model {MODEL} --out {emb} --weights {W_DIR} "
            f"--genome {gpath} --tau {tau:.4f}{gpu_flag}"
        )
        out = sh(
            f"{KL_EVAL} --orig {MODEL} --sparse {emb} --prompts {PROMPTS} "
            f"--n-positions {n_pos} --device {KL_DEVICE}"
        )
        os.remove(emb)
    else:
        gpath = f"{TMP}/vib_genome.bin"
        with open(gpath, "wb") as f:
            f.write(np.asarray(genome_z, np.float32).tobytes())
        out = sh(
            f'{EVAL} --model {MODEL} --prompts {PROMPTS} --genome {gpath} '
            f"--tau {tau:.4f} --n-positions {n_pos} --device cpu"
        )
    return json.loads(out.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description="CMA-ES global Vía B (CPPN con y_layer)")
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--darch", type=float, default=0.10,
        help="D_arch global objetivo (fijo); el CPPN evoluciona el perfil para minimizar KL",
    )
    ap.add_argument(
        "--full", action="store_true",
        help="evaluar la topología CPPN REAL (eval_sparse --genome) en vez del proxy de densidad",
    )
    ap.add_argument(
        "--streaming", action="store_true",
        help="topología CPPN REAL en el path de producción (embed_sparse --genome + kl_eval)",
    )
    ap.add_argument("--tau", type=float, default=TAU_CPPN, help="umbral CPPN (modo topología)")
    ap.add_argument("--n-pos", type=int, default=N_POS, help="posiciones del evaluador KL")
    ap.add_argument(
        "--gpu",
        action="store_true",
        help="decodificar la topología CPPN en la GPU (embed_sparse --gpu); "
        "necesario para ALIA-40b (inviable en CPU). Sin él, decode por CPU.",
    )
    ap.add_argument("--model", type=str, default=MODEL, help="GGUF original del profesor")
    ap.add_argument("--weights", type=str, default=str(W_DIR), help="dir de dump_weights")
    ap.add_argument("--n-layers", type=int, default=N_LAYERS, help="capas del modelo")
    ap.add_argument("--d-in", type=int, default=D_IN, help="d_in del FFN (hidden)")
    ap.add_argument("--d-out", type=int, default=D_OUT, help="d_out del FFN (intermedio)")
    ap.add_argument(
        "--device", type=str, default=KL_DEVICE,
        help="device del kl_eval del streaming (auto|cpu); ALIA puede requerir cpu",
    )
    ap.add_argument(
        "--name", type=str, default=NAME,
        help="prefijo de salidas (via_b_best_genome_<name>.bin, via_b_history_<name>.json)",
    )
    args = ap.parse_args()
    global GPU_EMBED, MODEL, N_LAYERS, D_IN, D_OUT, W_DIR, KL_DEVICE, NAME
    GPU_EMBED = args.gpu
    MODEL = args.model
    N_LAYERS = args.n_layers
    D_IN = args.d_in
    D_OUT = args.d_out
    W_DIR = Path(args.weights)
    KL_DEVICE = args.device
    NAME = args.name

    if args.streaming and not (W_DIR / "meta.json").exists():
        # Solo el gate: up/down quedan densos (semántica D16); evita volcar
        # ~110 GB de up/down en ALIA-40b.
        print(sh(f"{DUMP_WEIGHTS} --model {MODEL} --out {W_DIR} --blocks gate").strip(), flush=True)

    genome_dim = CppnGenome().param_count  # 466
    rho = float(np.clip(1.0 - args.darch, 0.05, 0.95))  # densidad media fija
    params = CmaEsParams(genome_dim, args.seed)
    mean0 = random_genome(args.seed)  # perfil campana inicial
    state = CmaEsState(params, mean0)

    history = []
    best_kl = {"kl": float("inf"), "darch": None, "z": None, "sps": None}
    best_profile = {"kl": None, "darch": None, "z": None, "sps": None}
    topology = args.full or args.streaming

    for gen in range(args.gens):
        pop = state.spawn_population(args.seed + gen)
        scored = []
        for col in range(pop.candidates.shape[1]):
            z = pop.candidates[:, col]
            if topology:
                # Topología CPPN real: maximizar D_arch sujeto a KL <= 0.50.
                sps = None
                r = evaluate([], genome_z=z, tau=args.tau, streaming=args.streaming,
                             n_pos=args.n_pos)
                d_arch, kl = r["d_arch_global"], r["kl_global"]
                fitness = d_arch - LAMBDA_PEN * max(0.0, kl - KL_MAX)
            else:
                sps = decode_sparsities(z, rho)
                r = evaluate(sps)
                d_arch, kl = r["d_arch_global"], r["kl_global"]
                fitness = -kl  # minimizar KL al D_arch fijado por rho
            scored.append((fitness, kl, d_arch, sps, z))
            if kl < best_kl["kl"]:
                best_kl = {"kl": kl, "darch": d_arch, "z": z, "sps": sps}
        scored.sort(key=lambda s: -s[0])
        order = sorted(range(len(scored)), key=lambda i: -scored[i][0])
        state.update(pop, order[: params.mu])
        gen_best = scored[0]
        if gen > 0 and gen_best[0] <= prev_fitness:
            # Reinicio de sigma: el paso no mejoró (CMA-ES divergente). Re-seed
            # con el mejor genoma conocido y sigma inicial para estabilizar.
            mean0 = best_kl["z"].copy()
            state = CmaEsState(params, mean0)
            print(
                f"  [restart sigma gen {gen}: fitness {prev_fitness:.3f} "
                f"-> {gen_best[0]:.3f}]",
                flush=True,
            )
        prev_fitness = gen_best[0]
        history.append(
            {
                "gen": gen,
                "best_kl": round(gen_best[1], 4),
                "best_darch": round(gen_best[2], 4),
                "best_fitness": round(gen_best[0], 4),
            }
        )
        print(
            f"gen {gen:2d} best_kl={gen_best[1]:.4f} darch={gen_best[2]:.3f} "
            f"(target {args.darch:.2f}) | mejora kl={best_kl['kl']:.4f}",
            flush=True,
        )

    print(f"\n=== Mejor perfil para D_arch ~ {args.darch:.2f} (KL={best_kl['kl']:.4f}, "
          f"D_arch_real={best_kl['darch']:.4f}) ===")
    if best_kl["sps"] is not None:
        print("  esparsidad por capa:", [round(s, 3) for s in best_kl["sps"]])
    with open(f"{TMP}/via_b_best_genome_{NAME}.bin", "wb") as f:
        f.write(np.asarray(best_kl["z"], np.float32).tobytes())
    print(f"  genoma guardado en {TMP}/via_b_best_genome_{NAME}.bin")

    with open(f"{TMP}/via_b_history_{NAME}.json", "w") as f:
        json.dump(
            {
                "history": history,
                "best": {
                    "kl": best_kl["kl"],
                    "darch": best_kl["darch"],
                    "sps": best_kl["sps"],
                    "z": None if best_kl["z"] is None else best_kl["z"].tolist(),
                },
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()

