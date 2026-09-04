"""Harness de validación cruzada del kernel OpenCL `cppn_decode_v7` (Vía B-v7:
la CPPN genera los pesos con geometría aprendida por canal) contra la
referencia NumPy **f32 canónica** (D1: misma secuencia de operaciones f32).

Ejecuta `saor-engine decode-v7` sobre bloques reales del smol (hi y ih) con un
genoma determinista y compara, con el remapeo de layout i-mayor/j-mayor del
bit-tensor `ffn_dag_adjacency`:

* adyacencia (máscara `sigmoid(link) > tau`): se cuentan los *flips*
  (conexiones que cruzan el umbral por diferencias de ulp OpenCL/libm);
  se exige una tasa despreciable (< 0.05 % de los activos);
* `max_abs_err` de W sobre el conjunto activo común dentro de tolerancia f32.

Uso: python -m saor_orchestrator.validate_opencl_v7
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from saor_orchestrator import SaorEngineClient

D_IN_HI, D_OUT_HI = 576, 1536  # gate/up: hidden -> inter
D_IN_IH, D_OUT_IH = 1536, 576  # down: inter -> hidden
N_LAYERS = 30
HIDDEN = 48
D_H, D_I = 576, 1536
CPPN_N = 6 * HIDDEN + HIDDEN + 2 * (HIDDEN * HIDDEN + HIDDEN) + 2 * HIDDEN + 2
_OFF_B0 = 6 * HIDDEN
_OFF_WM0 = _OFF_B0 + HIDDEN
_OFF_BM0 = _OFF_WM0 + HIDDEN * HIDDEN
_OFF_WM1 = _OFF_BM0 + HIDDEN
_OFF_BM1 = _OFF_WM1 + HIDDEN * HIDDEN
_OFF_W2 = _OFF_BM1 + HIDDEN
_OFF_B2 = _OFF_W2 + 2 * HIDDEN
_OFF_COORD_H = _OFF_B2 + 2
_OFF_COORD_I = _OFF_COORD_H + D_H

def ref_decode_f32(flat32: np.ndarray, d_in: int, d_out: int, layer: int,
                   which: str, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Referencia f32 canónica (espejo del kernel): W [d_out, d_in] f32 y la
    máscara [d_out, d_in] con la semántica exacta de v7cppn.py."""
    f = np.asarray(flat32, np.float32)
    hid = HIDDEN
    w0 = f[0:6 * hid].reshape(hid, 6)
    b0 = f[_OFF_B0:_OFF_B0 + hid]
    wms, bms = [], []
    for base_w, base_b in ((_OFF_WM0, _OFF_BM0), (_OFF_WM1, _OFF_BM1)):
        wms.append(f[base_w:base_w + hid * hid].reshape(hid, hid))
        bms.append(f[base_b:base_b + hid])
    w2 = f[_OFF_W2:_OFF_W2 + 2 * hid].reshape(2, hid)
    b2 = f[_OFF_B2:_OFF_B2 + 2]
    coord_h = f[_OFF_COORD_H:_OFF_COORD_H + D_H]
    coord_i = f[_OFF_COORD_I:_OFF_COORD_I + D_I]
    ci = np.clip(coord_i if which == "ih" else coord_h, -5.0, 5.0)
    co = np.clip(coord_h if which == "ih" else coord_i, -5.0, 5.0)
    zc = np.float32(-1.0) + np.float32(2.0) * (
        np.float32(layer) + np.float32(0.5)) / np.float32(N_LAYERS)
    zc = np.float32(np.clip(zc, -2.0, 2.0))

    Xj, Xi = np.meshgrid(co, ci, indexing="ij")  # [d_out, d_in]
    v = np.stack([Xi, Xj, Xj - Xi, np.full_like(Xi, zc),
                  np.zeros_like(Xi), np.zeros_like(Xj)], axis=-1).reshape(-1, 6)
    h = np.tanh(v @ w0.T + b0)
    for w, b in zip(wms, bms):
        h = np.sin(h * np.float32(np.pi))
        h = np.tanh(h @ w.T + b)
        h = np.exp(np.float32(-0.5) * (h * np.float32(1.2)) ** np.float32(2))
    out = h @ w2.T + b2
    l = np.float32(1.0) / (np.float32(1.0) + np.exp(-out[:, 1]))
    W = np.where((l.reshape(d_out, d_in) > np.float32(tau)),
                 out[:, 0].reshape(d_out, d_in).astype(np.float32),
                 np.float32(0.0))
    return W, (W != 0.0)


def _mask_bytes(mask_ij: np.ndarray) -> np.ndarray:
    """Empaqueta una máscara [d_in, d_out] i-mayor (conn = i*d_out + j) en u8
    LSB-first, igual que `cppn_decode_v7` + `v7_pack_adjacency`."""
    bits = np.zeros((mask_ij.size + 7) // 8, np.uint8)
    conns = np.flatnonzero(mask_ij.reshape(-1))
    if conns.size:
        bits[conns // 8] |= ((1 << (conns % 8)) & 0xFF).astype(np.uint8)
    return bits

def run_decode(
    engine_bin: str,
    genome: Path,
    d_in: int,
    d_out: int,
    layer: int,
    mode: str,
    tau: float,
    out_dir: Path,
) -> dict:
    subprocess.run(
        [
            engine_bin, "decode-v7",
            "--genome", str(genome),
            "--d-in", str(d_in),
            "--d-out", str(d_out),
            "--layer", str(layer),
            "--n-layers", str(N_LAYERS),
            "--mode", mode,
            "--tau", str(tau),
            "--out", str(out_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    meta = json.loads((out_dir / "meta.json").read_text())
    w_dense = np.fromfile(out_dir / "w.bin", np.float32)  # i-mayor [d_in*d_out]
    adj = np.fromfile(out_dir / "adj.bin", np.uint8)
    return {**meta, "w_dense": w_dense, "adj": adj}


def validate_block(
    engine_bin: str,
    genome_file: Path,
    d_in: int,
    d_out: int,
    layer: int,
    mode: str,
    which: str,
    tau: float,
    flat32: np.ndarray,
    tmp: Path,
) -> dict:
    out = run_decode(engine_bin, genome_file, d_in, d_out, layer, mode, tau, tmp)
    W, mask_ji = ref_decode_f32(flat32, d_in, d_out, layer, which, tau)
    mask_ij_ref = mask_ji.T  # [d_in, d_out]: conn = i*d_out + j
    adj_ref = _mask_bytes(mask_ij_ref)

    bits = np.unpackbits(np.asarray(out["adj"], np.uint8), bitorder="little")
    mask_ij_kernel = bits[: d_in * d_out].reshape(d_in, d_out)
    flips = int(np.sum(mask_ij_ref != mask_ij_kernel))

    w_kernel_ji = out["w_dense"].reshape(d_in, d_out).T  # [d_out, d_in]
    common = mask_ji & (w_kernel_ji != 0.0)
    if common.sum():
        err = float(np.max(np.abs(W.astype(np.float32)[common]
                                  - w_kernel_ji[common])))
    else:
        err = 0.0
    active_ref = int(mask_ij_ref.sum())
    return {
        "mode": mode,
        "d_in": d_in,
        "d_out": d_out,
        "layer": layer,
        "flips": flips,
        "flip_rate": flips / max(active_ref, 1),
        "max_err": err,
        "active_kernel": int(out["active"]),
        "active_ref": active_ref,
        "decode_ms": int(out["decode_ms"]),
        "device": out["device"],
    }


def main() -> int:
    engine_bin = SaorEngineClient().engine_bin
    tmp = Path(tempfile.mkdtemp(prefix="v7_parity_"))
    rng = np.random.default_rng(7)
    flat32 = np.concatenate([
        rng.normal(0.0, 1.5, CPPN_N),
        rng.uniform(-3.0, 3.0, D_H),
        rng.uniform(-3.0, 3.0, D_I),
    ]).astype(np.float32)
    genome_file = tmp / "genome.bin"
    genome_file.write_bytes(
        np.ascontiguousarray(flat32, np.float32).tobytes())

    results = []
    results.append(validate_block(
        engine_bin, genome_file, D_IN_HI, D_OUT_HI, layer=0, mode="hi", which="hi",
        tau=0.5, flat32=flat32, tmp=tmp / "hi"))
    results.append(validate_block(
        engine_bin, genome_file, D_IN_IH, D_OUT_IH, layer=3, mode="ih", which="ih",
        tau=0.5, flat32=flat32, tmp=tmp / "ih"))
    results.append(validate_block(
        engine_bin, genome_file, D_IN_HI, D_OUT_HI, layer=29, mode="hi", which="hi",
        tau=0.5, flat32=flat32, tmp=tmp / "hi29"))

    print("Validación cruzada cppn_decode_v7 (OpenCL vs NumPy f32):")
    ok = True
    for r in results:
        print(
            f"  [{r['mode']} d_in={r['d_in']} d_out={r['d_out']} layer={r['layer']}] "
            f"flips={r['flips']}/{r['active_ref']} ({r['flip_rate']:.2e}) "
            f"active_k={r['active_kernel']} max_err={r['max_err']:.3e} "
            f"decode_ms={r['decode_ms']} device={r['device']}"
        )
        if r["flip_rate"] > 5e-4:
            ok = False
        if r["max_err"] > 1e-3:
            ok = False
    print("OK" if ok else "FALLO")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

