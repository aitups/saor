"""Smoke tests de Fase 0: importabilidad, IPC y (si hay GPU) OpenCL."""

import pytest


def test_version_del_paquete():
    import saor_orchestrator

    assert saor_orchestrator.__version__ == "0.1.0"


def test_cliente_ipc_device_info_con_driver():
    """Si el binario está compilado y hay OpenCL, device-info debe responder ok."""
    from saor_orchestrator import SaorEngineClient

    try:
        client = SaorEngineClient()
    except FileNotFoundError:
        pytest.skip("saor-engine no compilado aún")
    report = client.device_info()
    assert "ok" in report
    if report.get("ok"):
        assert report["devices"], "debe listar al menos un dispositivo"
        gpus = [d for d in report["devices"] if d["device_type"] == "GPU"]
        if gpus:
            # Restricción del experimento: ~6 GB de VRAM.
            assert gpus[0]["global_mem_mib"] > 5000
