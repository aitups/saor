"""Loop evolutivo integrado (Fase 4): validación del pipeline end-to-end."""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from saor_orchestrator import SaorEngineClient


def _run_evolve(gens: int, seed: int, out: Path) -> dict:
    engine = SaorEngineClient().engine_bin
    subprocess.run(
        [engine, "evolve", "--gens", str(gens), "--seed", str(seed), "--out", str(out)],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return json.loads(out.read_text())


def test_loop_evolutivo_sintetico():
    """Requiere GPU OpenCL y `saor-engine` compilado (se omite si no están)."""
    try:
        SaorEngineClient()
    except FileNotFoundError:
        pytest.skip("saor-engine no compilado")

    gens = 6
    out = Path(tempfile.gettempdir()) / "saor_evolve_test.json"
    report = _run_evolve(gens, 42, out)

    assert report["ok"], f"el loop evolutivo no validó: {report.get('error')}"
    assert len(report["history"]) == gens

    history = report["history"]
    # best_so_far no decreciente (selección élite de máximos).
    best = [h["best_so_far"] for h in history]
    assert all(best[i] <= best[i + 1] for i in range(len(best) - 1))

    # CKA siempre finito y en [0, 1].
    for h in history:
        assert 0.0 <= h["best_cka"] <= 1.0

    # La evolución debe esparsificar el bloque (D_arch alcanza >= 0.4).
    max_sparsity = max(h["best_sparsity"] for h in history)
    assert max_sparsity >= 0.4, f"la evolución no esparsificó: max_sparsity={max_sparsity}"

    # El mejor fitness de la última generación supera al de la primera.
    assert history[-1]["best_fitness"] > history[0]["best_fitness"]
