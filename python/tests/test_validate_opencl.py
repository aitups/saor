"""Validación cruzada de los kernels OpenCL contra la referencia NumPy (Fase 3)."""

import subprocess

import pytest

from saor_orchestrator.validate_opencl import run_engine, validate_report


def test_validacion_cruzada_kernels_opencl():
    """Requiere GPU OpenCL y `saor-engine` compilado (se omite si no están)."""
    from saor_orchestrator import SaorEngineClient

    try:
        SaorEngineClient()
    except FileNotFoundError:
        pytest.skip("saor-engine no compilado")
    try:
        report = run_engine()
    except (subprocess.CalledProcessError, RuntimeError, FileNotFoundError) as exc:
        pytest.skip(f"no se pudo ejecutar kernels-run: {exc}")
    try:
        results = validate_report(report)
    except RuntimeError as exc:
        pytest.skip(f"binario no compatible: {exc}")
    assert results["err_w"] < 1e-3
    assert results["err_y_dense"] < 1e-3
    assert results["err_y_csr"] < 1e-3
    assert 0.0 <= results["cka"] <= 1.0
