"""Prueba rápida de configuraciones heterogéneas (D20): D_arch vs KL_global.

Configuraciones manuales que concentran la esparsidad en las capas tolerantes
(del mapa de sensibilidad) para calibrar la factibilidad de D_arch 0.40 @ KL 0.50.
"""
import json
import subprocess

TMP = "C:/Users/epoke/AppData/Local/Temp"
MODEL = "d:\\Documents\\pySrc\\hayai\\models\\SmolLM2-135M-Instruct-Q4_K_M.gguf"
EVAL = "d:\\Documents\\pySrc\\hayai\\target\\release\\examples\\eval_sparse.exe"
PROMPTS = f"{TMP}/calib128.txt"


def sh(cmd):
    r = subprocess.run(["cmd", "/c", cmd], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd}\n{r.stderr[-1200:]}")
    return r.stdout


def eval_sp(sps: list[float], n_pos: int = 48) -> dict:
    path = f"{TMP}/sp_cfg.txt"
    with open(path, "w") as f:
        f.write("\n".join(f"{s:.4f}" for s in sps) + "\n")
    out = sh(
        f'{EVAL} --model {MODEL} --prompts {PROMPTS} --sparsities {path} '
        f"--n-positions {n_pos} --device cpu"
    )
    return json.loads(out.strip())


def cfg(name, fn):
    sps = [0.0] * 30
    for layer, s in fn.items():
        sps[layer] = s
    d_arch = sum(sps) / 30  # mismo peso por capa (gates del mismo tamaño)
    r = eval_sp(sps)
    print(f"{name:28s} | D_arch={d_arch:5.2f} | KL={r['kl_global']:.4f}")
    return d_arch, r["kl_global"]


def main():
    print("Configuraciones heterogéneas (solo capas tolerantes esparsificadas):")
    # C1: capas 12-27 a 0.6 (16 capas) -> D_arch ~0.32
    cfg("C1 tol12-27@0.6", {l: 0.6 for l in range(12, 28)})
    # C2: C1 + capas 4-7 a 0.5 -> D_arch ~0.387
    cfg("C2 +4-7@0.5", {**{l: 0.6 for l in range(12, 28)}, **{l: 0.5 for l in range(4, 8)}})
    # C3: C2 + capas 1,3,10 a 0.3 -> D_arch ~0.417
    cfg("C3 +1,3,10@0.3", {**{l: 0.6 for l in range(12, 28)}, **{l: 0.5 for l in range(4, 8)}, **{1: 0.3, 3: 0.3, 10: 0.3}})
    # C4: conservadora para KL: solo 16-27 a 0.6 + 12-15 a 0.3 -> D_arch ~0.28
    cfg("C4 16-27@0.6 +12-15@0.3", {**{l: 0.6 for l in range(16, 28)}, **{l: 0.3 for l in range(12, 16)}})


if __name__ == "__main__":
    main()
