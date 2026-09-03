"""M3a — Cota de la regresión del warm-start (v7-b): ¿puede la CPPN pintar una
réplica de los pesos reales del profesor?

La CPPN (suave en la geometría) debe, en la gen-0 (warm-start), reproducir los
pesos W del profesor para partir de KL ~ 0. El hallazgo D15 previo sugiere que
los pesos FFN reales NO tienen estructura suave en (i,j) y la CPPN no puede
regresarlos. Este script mide la cota: reconstrucción de un slice del gate real
con una CPPN de una capa oculta (readout lineal por mínimos cuadrados — la mejor
réplica posible para esa arquitectura) + CKA / error relativo.

Si CKA es alto (~0.9) el warm-start es viable; si es bajo (~0.1-0.3), la gen-0
NO partirá de KL ~ 0 y la premisa del warm-start de la v7 queda en riesgo.
"""
from __future__ import annotations

import numpy as np


def cka(a: np.ndarray, b: np.ndarray) -> float:
    """Centered Kernel Alignment lineal entre dos matrices de pesos (2D)."""
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    num = float(np.sum(a * b) ** 2)
    den = float(np.sum(a * a) * np.sum(b * b))
    return float(num / den) if den > 0 else 0.0


def main() -> int:
    w = np.fromfile(r"d:\Documents\pySrc\.scratch\w_qwen35\w.0.ffn_gate.bin", np.float32)
    d_in, d_out = 2560, 9216
    w = w.reshape(d_out, d_in)
    # Slice: primeras filas (d_out_s) y una sub-columna (i aleatoria) — la cota de
    # una CPPN suave en (i,j). Usamos d_out_s filas completas para el readout.
    rng = np.random.default_rng(0)
    d_out_s, d_in_s = 512, 512
    cols = np.sort(rng.choice(d_in, d_in_s, replace=False))
    rows = np.arange(d_out_s)
    Wt = w[rows][:, cols]  # [512, 512] slice real
    # Geometría: i (canal de entrada), j (canal de salida) en [-1,1].
    xj = np.linspace(-1, 1, d_out_s, dtype=np.float32)
    xi = np.linspace(-1, 1, d_in_s, dtype=np.float32)
    Xj, Xi = np.meshgrid(xj, xi, indexing="ij")  # [d_out_s, d_in_s]
    # features geométricas 9-D (sustrato global v7): pares de coordenadas.
    feat = np.stack([
        Xi, np.zeros_like(Xi), Xj, np.zeros_like(Xj),
        np.zeros_like(Xi), np.zeros_like(Xj), Xj - Xi, np.zeros_like(Xj), np.zeros_like(Xi),
    ], axis=-1).reshape(-1, 9).astype(np.float32)  # [N, 9]
    # Método B (ELM): capa oculta aleatoria + readout lineal por lstsq.
    rng2 = np.random.default_rng(1)
    hidden = 256
    W0 = rng2.normal(0, 1.0, (hidden, 9)).astype(np.float32)
    b0 = rng2.normal(0, 0.5, hidden).astype(np.float32)
    H = np.tanh(feat @ W0.T + b0)  # [N, hidden]
    target = Wt.reshape(-1)  # [N]
    coef, *_ = np.linalg.lstsq(H, target, rcond=None)
    recon = (H @ coef).reshape(d_out_s, d_in_s).astype(np.float32)
    mse = float(np.mean((recon - Wt) ** 2))
    var = float(np.var(Wt))
    # Suavidad local del gate real: autocorrelacion fila/columna adyacente.
    r_c = float(np.corrcoef(Wt[:, :-1].ravel(), Wt[:, 1:].ravel())[0, 1])
    c_c = float(np.corrcoef(Wt[:-1, :].ravel(), Wt[1:, :].ravel())[0, 1])
    # Cota: regresion sobre el SUAVIZADO (media 3x3) — si la CPPN lo ajusta y el
    # crudo no, el limite es la no-suavidad intrinseca de los pesos reales.
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.pad(Wt, 1, mode="edge")
    Ws = sliding_window_view(pad, (3, 3)).mean(axis=(2, 3))
    Hs = np.tanh(feat @ W0.T + b0)
    coef_s, *_ = np.linalg.lstsq(Hs, Ws.reshape(-1), rcond=None)
    recon_s = (Hs @ coef_s).reshape(d_out_s, d_in_s).astype(np.float32)
    print(f"slice gate real [{d_out_s}x{d_in_s}]:")
    print(f"  suavidad local: corr filas adyacentes={r_c:.3f}, corr cols adyacentes={c_c:.3f}")
    print(f"  CPPN vs W crudo:  MSE_rel={mse/var:.4f}  CKA={cka(recon, Wt):.4f}")
    print(f"  CPPN vs W suave:  MSE_rel={float(np.mean((recon_s-Ws)**2))/float(np.var(Ws)):.4f}  CKA={cka(recon_s, Ws):.4f}")
    print("Lectura: corr adyacente ~0 -> pesos sin estructura local (no-suaves); la CPPN")
    print("geometrica no puede pintar la replica del profesor (warm-start gen-0 en riesgo).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
