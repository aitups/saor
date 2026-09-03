"""Vía 2 de la v7-b: ¿un reordenamiento espectral de los canales hace los pesos
reales regresables? Si los canales tienen un orden latente (los pesos se vuelven
suaves tras una permutacion por similaridad), la geometria aprendida funciona.
"""
from __future__ import annotations

import numpy as np


def local_corr(w: np.ndarray) -> float:
    return float(np.corrcoef(w[:-1, :].ravel(), w[1:, :].ravel())[0, 1])


def cka(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    num = float(np.sum(a * b) ** 2)
    den = float(np.sum(a * a) * np.sum(b * b))
    return float(num / den) if den > 0 else 0.0


def main() -> int:
    w = np.fromfile(r"d:\Documents\pySrc\.scratch\w_qwen35\w.0.ffn_gate.bin", np.float32)
    rng = np.random.default_rng(0)
    cols = np.sort(rng.choice(2560, 256, replace=False))
    Wt = w.reshape(9216, 2560)[:256][:, cols].astype(np.float32)
    print(f"corr filas adyacentes (orden crudo): {local_corr(Wt):.3f}")
    # Orden espectral: coordenada 1D de cada fila = 1er autovector de la
    # similaridad entre filas (normalizado).
    Wn = Wt - Wt.mean(0, keepdims=True)
    G = Wn @ Wn.T  # similaridad entre filas [d_out, d_out]
    evals, evecs = np.linalg.eigh(G)
    coord = evecs[:, -1]  # 1er autovector (mayor valor propio)
    order = np.argsort(coord)
    Wo = Wt[order]
    print(f"corr filas adyacentes (orden espectral): {local_corr(Wo):.3f}")
    # Regresion suave en el orden espectral.
    d_out, d_in = Wo.shape
    xi = np.linspace(-1, 1, d_in, dtype=np.float32)
    xj = np.linspace(-1, 1, d_out, dtype=np.float32)
    Xj, Xi = np.meshgrid(xj, xi, indexing="ij")
    F = np.stack([Xi, Xj, Xj - Xi], axis=-1).reshape(-1, 3).astype(np.float32)
    r2 = np.random.default_rng(3)
    W0 = r2.normal(0, 1.0, (256, 3)).astype(np.float32)
    b0 = r2.normal(0, 0.5, 256).astype(np.float32)
    H = np.tanh(F @ W0.T + b0)
    coef, *_ = np.linalg.lstsq(H, Wo.reshape(-1), rcond=None)
    recon = (H @ coef).reshape(d_out, d_in)
    print(f"CKA(regresion suave, W orden espectral): {cka(recon, Wo):.4f}")
    print("Si la corr/CKA suben mucho -> la geometria aprendida es viable")
    print("(los canales tienen un orden latente que la regresion puede explotar).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
