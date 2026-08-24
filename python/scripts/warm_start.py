"""Warm-Start de la CPPN (Fase 4b) — ancla el genoma al tensor real del profesor.

Implementa la especificación `diseno_heuristica_warm_start_v2.md`:

* Sustrato `v_in` EXACTO del kernel OpenCL / `saor_domain::cppn`
  (`x_i=-1`, `x_j=1`, `y_i/y_j` lineales, `Δx=2`, `Δy=y_j-y_i`, sen/cos).
* Método A (torch): regresión de Frobenius con Adam + Ridge (spec §2).
* Método B (NumPy): proyección analítica de Tikhonov sobre la última capa
  oculta (ELM) — sin dependencias pesadas, instantánea.
* Alineación de `l_ij`: `w2[1,:]=0`, `b2[1]=logit(ρ_objetivo)`.
* Salida: genoma plano `.bin` en el orden `flatten()` de `saor_domain::cppn`
  (w0|b0|w1|b1|w2|b2, fila-mayor) + reporte de CKA sobre activaciones (B=128).

Uso: python warm_start.py --gguf <modelo.gguf> --tensor blk.0.ffn_gate.weight
      [--method b|a] [--out-dir <dir>] [--d-in N] [--d-out N] [--target-active 0.5]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from saor_orchestrator.reference.cka import centered_cka, gram_matrix
from saor_orchestrator.reference.cppn import CPPN_INPUT_DIM, HIDDEN, CppnGenome
from saor_orchestrator.reference.topology import dense_row_major, instantiate

GENOME_ORDER = "w0|b0|w1|b1|w2|b2"


# --------------------------------------------------------------------- sustrato

def build_substrate(d_in: int, d_out: int) -> np.ndarray:
    """v_in [N, 8] con N = d_in*d_out, fila-mayor `k = i*d_out + j`.

    Paridad exacta con `saor_domain::cppn::input_vector` (y con el kernel
    `cppn_decode.cl`): x_i=-1.0, x_j=1.0, y_i/y_j lineales en [-1,1].
    """
    n = d_in * d_out
    i = np.arange(d_in)[:, None].repeat(d_out, axis=1).ravel()
    j = np.tile(np.arange(d_out), d_in)
    y_i = -1.0 + 2.0 * i / (d_in - 1) if d_in > 1 else np.zeros(n)
    y_j = -1.0 + 2.0 * j / (d_out - 1) if d_out > 1 else np.zeros(n)
    return np.stack(
        [
            -np.ones(n),
            y_i,
            np.ones(n),
            y_j,
            2.0 * np.ones(n),
            y_j - y_i,
            np.sin(np.pi * y_i),
            np.cos(np.pi * y_j),
        ],
        axis=1,
    ).astype(np.float32)


# ------------------------------------------------------------------ Método B

def h1_activations(v: np.ndarray, w0, b0, w1, b1) -> np.ndarray:
    """h1 [N, HIDDEN]: última capa oculta (activación sin) del sustrato."""
    h0 = np.tanh(v @ w0.T + b0)
    return np.sin(h0 @ w1.T + b1)


def method_b_pseudoinverse(
    target_flat: np.ndarray, d_in: int, d_out: int, ridge: float = 1e-4, seed: int = 0
) -> np.ndarray:
    """ELM: capas ocultas aleatorias fijas + capa de salida por Tikhonov.

    `θ_out = (AᵀA + λI)⁻¹ Aᵀ vec(W)` con A = [h1 | 1] (última capa oculta + bias).
    """
    rng = np.random.default_rng(seed)
    v = build_substrate(d_in, d_out)
    w0 = rng.normal(0, 1, (HIDDEN, CPPN_INPUT_DIM)).astype(np.float32)
    b0 = rng.normal(0, 1, (HIDDEN,)).astype(np.float32)
    w1 = rng.normal(0, 1, (HIDDEN, HIDDEN)).astype(np.float32)
    b1 = rng.normal(0, 1, (HIDDEN,)).astype(np.float32)

    h1 = h1_activations(v, w0, b0, w1, b1)  # [N, HIDDEN]
    a = np.hstack([h1, np.ones((h1.shape[0], 1), np.float32)])  # [N, H+1]
    at_a = a.T @ a + ridge * np.eye(HIDDEN + 1, dtype=np.float32)
    beta_w = np.linalg.solve(at_a, a.T @ target_flat)

    w2 = np.zeros((2, HIDDEN), np.float32)
    b2 = np.zeros((2,), np.float32)
    w2[0, :] = beta_w[:HIDDEN]
    b2[0] = beta_w[HIDDEN]
    return stack_genome(w0, b0, w1, b1, w2, b2)


def stack_genome(w0, b0, w1, b1, w2, b2) -> np.ndarray:
    """Genoma plano en el orden `flatten()` de saor_domain::cppn (fila-mayor)."""
    return np.concatenate(
        [
            w0.reshape(-1, order="C"),
            b0,
            w1.reshape(-1, order="C"),
            b1,
            w2.reshape(-1, order="C"),
            b2,
        ]
    ).astype(np.float32)


# ------------------------------------------------------------------ Método A

def method_a_torch(
    target_flat: np.ndarray, d_in: int, d_out: int, epochs: int = 300, lr: float = 0.01
) -> np.ndarray:
    """Regresión de Frobenius con Adam (torch) — spec §2."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "torch no está instalado; usa --method b (NumPy) o instala torch"
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class WarmCPPN(nn.Module):
        def __init__(self, input_dim=8, hidden_dim=HIDDEN, output_dim=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sin(),  # type: ignore[attr-defined]
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, coords):
            return self.net(coords)

    cppn = WarmCPPN().to(device)
    optimizer = optim.Adam(cppn.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    v = torch.tensor(build_substrate(d_in, d_out), device=device)
    y = torch.tensor(target_flat.reshape(-1, 1), device=device)
    cppn.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = cppn(v)  # [N, 2]
        loss = criterion(out[:, 0:1], y)
        loss.backward()
        optimizer.step()

    params = [p.detach().cpu().numpy().reshape(-1) for p in cppn.parameters()]
    return np.concatenate(params).astype(np.float32)  # w0,b0,w1,b1,w2,b2


# ------------------------------------------------------------- alineación l_ij

def align_l_output(flat: np.ndarray, target_active: float) -> np.ndarray:
    """Fija la salida `l_ij`: w2[1,:]=0 y b2[1]=logit(ρ) para E[l]=ρ."""
    g = flat.reshape(-1).copy()
    w2_start = len(g) - 2 * HIDDEN - 2
    g[w2_start + HIDDEN : w2_start + 2 * HIDDEN] = 0.0
    rho = min(max(float(target_active), 1e-6), 1 - 1e-6)
    g[-1] = np.log(rho / (1.0 - rho))
    return g


# ---------------------------------------------------------------- fidelidad

def compute_fidelity(
    flat: np.ndarray, w_dense: np.ndarray, x: np.ndarray, d_in: int, d_out: int
) -> dict:
    """CKA sobre activaciones (B=128) + error de Frobenius relativo.

    `proxy_mse`: error cuadrático medio de reconstrucción de las activaciones
    del profesor sobre el lote de calibración (proxy de `proxy_nll` de la spec;
    cuanto menor, más fiel el bloque — la copia densa da ≈ 0).
    """
    genome = CppnGenome.from_flatten(flat)
    topo = instantiate(genome, d_in, d_out, 0.0)  # denso (τ=0)
    w_cand = dense_row_major(topo, d_in, d_out)  # [d_out, d_in]
    h0 = (x @ w_dense.T).astype(np.float32)
    h1 = (x @ w_cand.T).astype(np.float32)
    cka = centered_cka(gram_matrix(h0), gram_matrix(h1))
    frob = float(
        np.linalg.norm(w_dense - w_cand, "fro") / max(1e-9, np.linalg.norm(w_dense, "fro"))
    )
    proxy_mse = float(np.mean((h0 - h1) ** 2))
    return {"cka": cka, "frob_rel": frob, "proxy_mse": proxy_mse}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--method", choices=["a", "b"], default="b")
    ap.add_argument("--d-in", type=int, default=None)
    ap.add_argument("--d-out", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--target-active", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=os.path.join(os.environ.get("TEMP", "."), "saor_warmstart"))
    args = ap.parse_args()

    # 1) Cargar el tensor profesor (W_dense [d_out, d_in]) vía hayai dump.
    from saor_orchestrator.hooks.gguf_audit import read_gguf_header

    h = read_gguf_header(args.gguf)
    ti = {t.name: t for t in h.tensors}[args.tensor]
    d_in = args.d_in or int(ti.shape[0])
    d_out = args.d_out or int(ti.shape[1])
    wd = Path(args.out_dir)
    wd.mkdir(parents=True, exist_ok=True)

    w0_bin = wd / "w0.bin"
    subprocess.check_call(
        [
            "cmd", "/c",
            f"set PATH=C:\\msys64\\mingw64\\bin;C:\\msys64\\usr\\bin;%PATH% && "
            f"cd /d d:\\Documents\\pySrc\\hayai && "
            f"cargo run --release --example dump_tensor_f32 -- {args.gguf} {args.tensor} {w0_bin}",
        ],
        stdout=subprocess.DEVNULL,
        timeout=3600,
    )
    w_dense = np.fromfile(w0_bin, np.float32).reshape(d_out, d_in)

    # 2) Sustrato + target plano (índice k = i*d_out+j, paridad GPU).
    target_flat = w_dense.T.reshape(-1).astype(np.float32)

    # 3) Regresión.
    if args.method == "b":
        flat = method_b_pseudoinverse(target_flat, d_in, d_out, args.ridge, args.seed)
    else:
        flat = method_a_torch(target_flat, d_in, d_out, args.epochs, args.lr)
    flat = align_l_output(flat, args.target_active)

    # 4) Fidelidad sobre activaciones (B=128).
    x = np.random.default_rng(args.seed + 1).normal(0, 1, (args.batch, d_in)).astype(np.float32)
    fid = compute_fidelity(flat, w_dense, x, d_in, d_out)

    # 5) Guardar genoma + reporte.
    genome_bin = wd / "warm_genome.bin"
    genome_bin.write_bytes(flat.tobytes())
    x.tofile(wd / "x.bin")
    report = {
        "method": args.method,
        "tensor": args.tensor,
        "d_in": d_in,
        "d_out": d_out,
        "genome_len": int(len(flat)),
        "genome_order": GENOME_ORDER,
        "cka": float(fid["cka"]),
        "frob_rel": float(fid["frob_rel"]),
        "proxy_mse": float(fid["proxy_mse"]),
        "genome": str(genome_bin),
        "w0": str(w0_bin),
        "x": str(wd / "x.bin"),
    }
    (wd / "warm_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(
        f"[warm_start] método={args.method} CKA={fid['cka']:.4f} "
        f"frob_rel={fid['frob_rel']:.4f} proxy_mse={fid['proxy_mse']:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


