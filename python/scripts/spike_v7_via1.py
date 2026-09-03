"""Vía 1 de la v7-b: ¿una CPPN más expresiva reduce el error de la regresión del
warm-up sobre pesos reales? Compara el ELM básico (tanh, 96 oc) vs uno rico
(seno/coseno/gaussiana/tanh, mas ocultos) sobre el gate real del 4B.

Prediccion a verificar: si los pesos son ~aleatorios en el orden de canales, la
expresividad NO reduce el error (la cota es la no-suavidad, no la capacidad).
"""
from __future__ import annotations

import numpy as np


def cka(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    num = float(np.sum(a * b) ** 2)
    den = float(np.sum(a * a) * np.sum(b * b))
    return float(num / den) if den > 0 else 0.0


def regress(w: np.ndarray, hidden: int, seed: int, act: str) -> float:
    d_out, d_in = w.shape
    xi = np.linspace(-1, 1, d_in, dtype=np.float32)
    xj = np.linspace(-1, 1, d_out, dtype=np.float32)
    Xj, Xi = np.meshgrid(xj, xi, indexing="ij")
    F = np.stack([Xi, Xj, Xj - Xi], axis=-1).reshape(-1, 3).astype(np.float32)
    rng = np.random.default_rng(seed)
    W0 = rng.normal(0, 1.0, (hidden, 3)).astype(np.float32)
    b0 = rng.normal(0, 0.5, hidden).astype(np.float32)
    Z = F @ W0.T + b0
    if act == "tanh":
        H = np.tanh(Z)
    elif act == "rich":  # bloque de tanh + seno + coseno + gaussiana
        H = np.concatenate([
            np.tanh(Z),
            np.sin(Z * np.pi),
            np.cos(Z * np.pi),
            np.exp(-(Z * 0.7) ** 2),
        ], axis=1)
    else:
        raise ValueError(act)
    coef, *_ = np.linalg.lstsq(H, w.reshape(-1), rcond=None)
    recon = (H @ coef).reshape(d_out, d_in).astype(np.float32)
    return cka(recon, w)


def main() -> int:
    w = np.fromfile(r"d:\Documents\pySrc\.scratch\w_qwen35\w.0.ffn_gate.bin", np.float32)
    rng = np.random.default_rng(0)
    cols = np.sort(rng.choice(2560, 256, replace=False))
    Wt = w.reshape(9216, 2560)[:256][:, cols]
    for act, hid in [("tanh", 96), ("tanh", 512), ("rich", 96), ("rich", 512)]:
        c = regress(Wt, hid, 1, act)
        print(f"act={act:6s} hidden={hid:4d}: CKA(regresion, W real) = {c:.4f}")
    print("Si CKA sigue ~0 con mas capacidad/act. ricas -> la cota es la no-suavidad,")
    print("no la expresividad (la via 2, geometria aprendida, es la necesaria).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
