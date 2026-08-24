"""Curva KL a nivel de modelo vs esparsidad (poda por magnitud del gate blk.0).

Genera bloques dispersos a esparsidades fijas (poda por magnitud — la cota
superior que la topología CPPN debería aproximar), los embebe en el GGUF de
SmolLM2 y vuelca logits para medir la KL simétrica con dump_logits de hayai.
"""
import subprocess
import sys

import numpy as np

sys.path.insert(0, r"d:\Documents\pySrc\saor\python")

D_IN, D_OUT = 576, 1536
TMP = "C:/Users/epoke/AppData/Local/Temp"
MODEL = "d:\\Documents\\pySrc\\hayai\\models\\SmolLM2-135M-Instruct-Q4_K_M.gguf"
EMBED = "d:\\Documents\\pySrc\\saor\\target\\debug\\saor-engine.exe"
DUMP = "d:\\Documents\\pySrc\\hayai\\target\\release\\examples\\dump_logits.exe"


def sh(cmd):
    r = subprocess.run(["cmd", "/c", cmd], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd}\n{r.stderr[-1500:]}")


def build_block(sp: float, w0: np.ndarray, tag: str) -> str:
    """Adyacencia por magnitud + pesos en orden i-mayor (conn=i*d_out+j)."""
    total = D_IN * D_OUT
    keep = int((1 - sp) * total)
    flat = np.argsort(-np.abs(w0).ravel())  # mayores primero
    mask = np.zeros(total, np.uint8)
    mask[flat[:keep]] = 1

    bits = np.zeros((total + 7) // 8, np.uint8)
    weights = []
    for i in range(D_IN):
        for j in range(D_OUT):
            conn = i * D_OUT + j
            if mask[conn]:
                bits[conn // 8] |= np.uint8(1 << (conn % 8))
                weights.append(w0[j, i])
    adj_path = f"{TMP}/adj_{tag}.bin"
    w_path = f"{TMP}/w_{tag}.bin"
    bits.tofile(adj_path)
    np.asarray(weights, np.float32).tofile(w_path)
    block = f"{TMP}/block_{tag}.gguf"
    sh(
        f"{EMBED} make-block --d-in {D_IN} --d-out {D_OUT} --tau 0.0 "
        f"--adj {adj_path} --weights {w_path} --out {block}"
    )
    return block


def measure_kl(gguf: str) -> float:
    sh(
        f"{DUMP} --model {gguf} --prompt-file {TMP}/calib_prompts.txt "
        f"--out {TMP}/logits_sweep.bin --device cpu"
    )
    vocab = 49152
    lo = np.fromfile(f"{TMP}/logits_orig.bin", np.float32).reshape(-1, vocab)
    ls = np.fromfile(f"{TMP}/logits_sweep.bin", np.float32).reshape(-1, vocab)
    po = np.exp(lo - lo.max(1, keepdims=True))
    po = po / po.sum(1, keepdims=True)
    ps = np.exp(ls - ls.max(1, keepdims=True))
    ps = ps / ps.sum(1, keepdims=True)
    eps = 1e-9
    return float(
        0.5
        * (
            (po * np.log((po + eps) / (ps + eps))).sum(1).mean()
            + (ps * np.log((ps + eps) / (po + eps))).sum(1).mean()
        )
    )


def main():
    w0 = np.fromfile(f"{TMP}/smol_gate_w0.bin", np.float32).reshape(D_OUT, D_IN)
    print("sparsity | KL simetrica")
    for sp in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        block = build_block(sp, w0, f"sp{int(sp*100)}")
        gguf = f"{TMP}/smol_sweep_{int(sp*100)}.gguf"
        sh(
            f"{EMBED} embed --src {MODEL} --dst {gguf} "
            f"--block blk.0.ffn_gate.weight --sparse {block}"
        )
        kl = measure_kl(gguf)
        print(f"{sp:7.1f} | {kl:.4f}")


if __name__ == "__main__":
    main()
