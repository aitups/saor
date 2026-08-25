"""Harness de validación cruzada: kernels OpenCL vs referencia NumPy (Fase 3).

Ejecuta `saor-engine kernels-run`, carga el reporte (genoma, X, salidas de
kernel y métricas del motor) y recalcula la referencia con NumPy a partir de
los datos crudos, verificando:

* bit-tensor `ffn_dag_adjacency` idéntico al de `saor_orchestrator.reference`;
* `max_abs_err` de W densa, `spmm_dense` y `spmm_csr` dentro de tolerancia f32;
* CKA finito y consistente con las matrices de Gram reportadas.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from saor_orchestrator import SaorEngineClient
from saor_orchestrator.reference.cka import centered_cka
from saor_orchestrator.reference.cppn import CPPN_INPUT_DIM, HIDDEN, CppnGenome
from saor_orchestrator.reference.topology import dense_row_major, instantiate


def run_engine(engine_bin: str | None = None, out_path: Path | None = None) -> dict[str, Any]:
    """Ejecuta `kernels-run` y devuelve el reporte parseado."""
    engine_bin = engine_bin or SaorEngineClient().engine_bin
    out_path = out_path or Path(tempfile.gettempdir()) / "saor_kernels_validate.json"
    subprocess.run(
        [engine_bin, "kernels-run", "--out", str(out_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return json.loads(out_path.read_text())


def validate_report(report: dict[str, Any], tolerance: float = 1e-3) -> dict[str, float]:
    """Valida el reporte del motor contra la referencia NumPy."""
    assert report.get("ok"), f"el motor no validó sus kernels: {report.get('metrics')}"
    p = report["params"]
    d_in, d_out, batch, tau = p["d_in"], p["d_out"], p["batch"], p["tau"]
    data = report["data"]

    genome = np.asarray(data["genome"], np.float32)
    x = np.asarray(data["x"], np.float32).reshape(batch, d_in)

    # --- Referencia NumPy ---
    expected = CPPN_INPUT_DIM * HIDDEN + HIDDEN * HIDDEN + 2 * HIDDEN + HIDDEN + HIDDEN + 2
    if genome.size != expected:
        raise RuntimeError(
            f"binario saor-engine stale: genoma de {genome.size} f32 "
            f"(se esperaban {expected} con el sustrato de {CPPN_INPUT_DIM} dims); "
            "recompilar saor-engine para validar los kernels"
        )
    g = CppnGenome.from_flatten(genome)
    topo = instantiate(g, d_in, d_out, tau)
    w_ref = dense_row_major(topo, d_in, d_out)  # [d_out, d_in]
    y_ref = x @ w_ref.T  # [batch, d_out]

    # --- Comparación de adyacencia (debe ser idéntica) ---
    adj_ref = topo.adjacency_bits
    adj_kernel = np.asarray(data["adjacency"], np.uint8)
    assert np.array_equal(adj_ref, adj_kernel), "el bit-tensor de adyacencia difiere"

    # --- Salidas del kernel ---
    w_kernel = np.asarray(data["w_dense"], np.float32).reshape(d_out, d_in)
    y_dense = np.asarray(data["y_dense"], np.float32).reshape(batch, d_out)
    y_csr = np.asarray(data["y_csr"], np.float32).reshape(batch, d_out)

    err_w = float(np.max(np.abs(w_ref - w_kernel)))
    err_y_dense = float(np.max(np.abs(y_ref - y_dense)))
    err_y_csr = float(np.max(np.abs(y_ref - y_csr)))

    assert err_w < tolerance, f"cppn_decode W: max_abs_err={err_w} >= {tolerance}"
    assert err_y_dense < tolerance, f"spmm_dense: max_abs_err={err_y_dense} >= {tolerance}"
    assert err_y_csr < tolerance, f"spmm_csr: max_abs_err={err_y_csr} >= {tolerance}"

    # --- CKA (matrices de Gram del kernel + recomputación NumPy) ---
    k0 = np.asarray(data["gram_h0"], np.float32).reshape(batch, batch)
    k1 = np.asarray(data["gram_h1"], np.float32).reshape(batch, batch)
    cka_kernel = centered_cka(k0, k1)
    cka_metric = float(report["metrics"]["cka"])
    assert np.isfinite(cka_kernel)
    assert abs(cka_kernel - cka_metric) < 1e-4, "CKA del motor y NumPy deben coincidir"

    return {
        "err_w": err_w,
        "err_y_dense": err_y_dense,
        "err_y_csr": err_y_csr,
        "cka": cka_kernel,
        "active": int(topo.active_connections()),
        "sparsity": float(topo.sparsity()),
    }


def main() -> int:
    report = run_engine()
    results = validate_report(report)
    print("Validación cruzada OpenCL vs NumPy:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
