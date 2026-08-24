"""Fase 6: consolidación en GGUF disperso y contrato de Fase 2."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from saor_orchestrator import SaorEngineClient
from saor_orchestrator.contract import evaluate_contract
from saor_orchestrator.hooks.gguf_audit import read_gguf_header, read_sparse_block


def test_consolidacion_gguf_disperso_y_contrato():
    """Requiere GPU OpenCL y `saor-engine` compilado (se omite si no están)."""
    try:
        engine = SaorEngineClient().engine_bin
    except FileNotFoundError:
        pytest.skip("saor-engine no compilado")

    out = Path(tempfile.gettempdir()) / "saor_candidate_sparse_test.gguf"
    proc = subprocess.run(
        [engine, "consolidate", "--gens", "6", "--seed", "42", "--out-gguf", str(out)],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    report = json.loads(proc.stdout)
    assert report["ok"]
    assert out.exists()

    # 1) Contrato desde el motor: distancia arquitectónica >= 0.4.
    assert report["d_arch"] >= 0.4, f"D_arch={report['d_arch']} < 0.4"
    assert report["best_cka"] >= 0.8, "la fidelidad del mejor candidato debe ser alta"

    # 2) El auditor Python lee el GGUF disperso (validación cruzada).
    h = read_gguf_header(out)
    assert h.kv["saor.d_in"] == report["d_in"]
    assert h.kv["saor.d_out"] == report["d_out"]
    names = {t.name for t in h.tensors}
    assert names == {"ffn_dag_adjacency", "ffn_dag_weights"}

    # 3) Lectura del bloque disperso y contrato estructural.
    block = read_sparse_block(out)
    assert block.d_in == report["d_in"]
    assert block.d_out == report["d_out"]
    assert abs(block.sparsity() - report["d_arch"]) < 1e-6
    assert block.adjacency.size == (block.d_in * block.d_out + 7) // 8

    # 4) Evaluador del contrato con hooks sintéticos (mecánica completa).
    rng = np.random.default_rng(0)
    hook_x = rng.normal(0, 1, (32, block.d_in)).astype(np.float32)
    hook_h0 = rng.normal(0, 1, (32, block.d_out)).astype(np.float32)
    result = evaluate_contract(block, hook_x, hook_h0)
    assert abs(result.d_arch - block.sparsity()) < 1e-6
    assert 0.0 <= result.cka <= 1.0
    assert result.response_ratio > 0.0
    assert set(result.rules) == {"distancia", "fidelidad", "no_dormant"}

    # El veredicto debe ser determinista y reproducible.
    result2 = evaluate_contract(block, hook_x, hook_h0)
    assert result2.verdict == result.verdict

    out.unlink(missing_ok=True)
