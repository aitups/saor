"""Tests de Fase 5: auditoría GGUF, catálogo de roles y calibración."""

import struct
from pathlib import Path

import numpy as np
import pytest

from saor_orchestrator.hooks.calibration import (
    SyntheticTeacherRuntime,
    activation_variance,
    build_calibration_batch,
    hot_channel_indices,
)
from saor_orchestrator.hooks.gguf_audit import (
    GGUF_MAGIC,
    read_gguf_header,
)
from saor_orchestrator.hooks.role_catalog import (
    SPARSIFIABLE_ROLES,
    classify,
    coverage_report,
    identity_adjacency,
    init_weights_from_base,
    is_sparsifiable,
    select_candidate_blocks,
    sparsifiable_tensors,
)


class _FakeInfo:
    """Mínimo `TensorInfo`-like para tests de cobertura (name + numel)."""

    def __init__(self, name: str, numel: int) -> None:
        self.name = name
        self.numel = numel


def _write_mini_gguf(path: Path) -> None:
    """Escribe un GGUF v3 mínimo con 4 tensores (2 F32, 1 F16, 1 Q4_0)."""
    tensors = [
        ("token_embd.weight", (1024, 512), 0, 1024 * 512 * 4),  # F32
        ("blk.0.attn_q.weight", (1024, 1024), 1, 1024 * 1024 * 2),  # F16
        ("blk.0.ffn_gate.weight", (2048, 1024), 0, 2048 * 1024 * 4),  # F32
        ("blk.1.ffn_down.weight", (1024, 2048), 2, (2048 // 32) * 18 * 1024),  # Q4_0
        ("output_norm.weight", (1024,), 1, 1024 * 2),  # F16
    ]
    buf = bytearray()
    buf += struct.pack("<I", GGUF_MAGIC)
    buf += struct.pack("<I", 3)  # version
    buf += struct.pack("<Q", len(tensors))
    buf += struct.pack("<Q", 3)  # kv_count
    for key, vtype, payload in [
        ("general.architecture", 8, b"\x03\x00\x00\x00\x00\x00\x00\x00llm"),
        ("llama.context_length", 10, struct.pack("<Q", 4096)),
        ("general.quantization_version", 6, struct.pack("<f", 2.0)),
    ]:
        kb = key.encode()
        buf += struct.pack("<Q", len(kb)) + kb
        buf += struct.pack("<I", vtype) + payload
    offset = 0
    for name, shape, gtype, nbytes in tensors:
        nb = name.encode()
        buf += struct.pack("<Q", len(nb)) + nb
        buf += struct.pack("<I", len(shape))
        for d in shape:
            buf += struct.pack("<Q", d)
        buf += struct.pack("<i", gtype)
        buf += struct.pack("<Q", offset)
        offset += nbytes
    for _name, _shape, _gtype, nbytes in tensors:
        buf += b"\x00" * nbytes
    path.write_bytes(buf)


def test_audit_lee_solo_la_cabecera():
    path = Path(".") / "tmp_mini.gguf"
    _write_mini_gguf(path)
    h = read_gguf_header(path)
    assert h.version == 3
    assert h.total_params == 1024 * 512 + 1024 * 1024 + 2048 * 1024 + 1024 * 2048 + 1024
    assert len(h.tensors) == 5
    assert h.kv["general.architecture"] == "llm"
    assert h.kv["llama.context_length"] == 4096
    # El tensor Q4_0: 2048x1024 numel, 18 bytes por bloque de 32.
    q4 = next(t for t in h.tensors if t.name == "blk.1.ffn_down.weight")
    assert q4.nbytes() == (q4.numel // 32) * 18
    assert h.quantized_ratio > 0.0
    prof = h.profile()
    assert prof["total_params"] == h.total_params
    assert "ffn" in prof["params_by_role"]
    assert prof["top_tensors"][0]["name"] == "blk.0.ffn_gate.weight"
    path.unlink(missing_ok=True)


def test_audit_rechaza_magic_invalido():
    path = Path(".") / "tmp_bad.gguf"
    path.write_bytes(b"\x00\x00\x00\x00" + b"\x00" * 16)
    with pytest.raises(ValueError):
        read_gguf_header(path)
    path.unlink(missing_ok=True)


def test_classify_roles():
    assert classify("blk.3.ffn_gate.weight").role == "ffn.gate"
    assert classify("blk.3.ffn_gate.weight").layer == 3
    assert classify("blk.0.attn_k_norm.weight").role == "norm.k"
    assert classify("output_norm.weight").role == "norm.output"
    assert classify("output.weight").role == "lm_head"
    assert classify("model.layers.5.mlp.gate_proj.weight").role == "ffn.gate"
    assert classify("algo.raro.weight") is None


def test_classify_roles_ssm_y_attn_qwen():
    """Roles medidos del GGUF Qwen3.8-27B-UD (NextN: attn + SSM + FFN)."""
    assert classify("blk.0.attn_qkv.weight").role == "attn.qkv"
    assert classify("blk.0.attn_gate.weight").role == "attn.gate"
    assert classify("blk.0.attn_q.weight").role == "attn.q"
    assert classify("blk.0.attn_output.weight").role == "attn.out"
    # SSM: proyecciones esparcibles vs. núcleo recurrente denso.
    assert classify("blk.0.ssm_out.weight").role == "ssm.out"
    assert classify("blk.0.ssm_alpha.weight").role == "ssm.alpha"
    assert classify("blk.0.ssm_beta.weight").role == "ssm.beta"
    assert classify("blk.0.ssm_a").role == "ssm.core"
    assert classify("blk.0.ssm_dt.bias").role == "ssm.core"
    assert classify("blk.0.ssm_norm.weight").role == "ssm.core"
    assert classify("blk.0.ssm_conv1d.weight").role == "ssm.core"
    assert classify("nextn.eh_proj").role == "nextn"


def test_roles_esparcibles():
    """FFN + atención + proyecciones SSM son esparcibles; núcleo S6/norms no."""
    assert is_sparsifiable("blk.0.ffn_gate.weight")
    assert is_sparsifiable("blk.0.attn_qkv.weight")
    assert is_sparsifiable("blk.0.attn_gate.weight")
    assert is_sparsifiable("blk.0.ssm_out.weight")
    assert is_sparsifiable("blk.0.ssm_alpha.weight")
    assert is_sparsifiable("blk.0.ssm_beta.weight")
    assert not is_sparsifiable("blk.0.ssm_a")
    assert not is_sparsifiable("blk.0.ssm_dt.bias")
    assert not is_sparsifiable("blk.0.ssm_conv1d.weight")
    assert not is_sparsifiable("blk.0.attn_norm.weight")
    assert not is_sparsifiable("token_embd.weight")
    assert not is_sparsifiable("output.weight")
    assert not is_sparsifiable("nextn.eh_proj")

    names = [
        "blk.0.ffn_gate.weight", "blk.0.attn_qkv.weight", "blk.0.ssm_out.weight",
        "blk.0.ssm_a", "blk.0.attn_norm.weight", "token_embd.weight",
    ]
    assert sparsifiable_tensors(names) == [
        "blk.0.ffn_gate.weight", "blk.0.attn_qkv.weight", "blk.0.ssm_out.weight",
    ]
    assert "ffn.gate" in SPARSIFIABLE_ROLES
    assert "ssm.core" not in SPARSIFIABLE_ROLES


def test_coverage_report_fraccion_esparcible():
    """La cobertura >90% (FFN+attn+SSM-proj) es el objetivo de Fase 0."""
    infos = [
        _FakeInfo("blk.0.ffn_gate.weight", 10_000),
        _FakeInfo("blk.0.ffn_up.weight", 10_000),
        _FakeInfo("blk.0.attn_qkv.weight", 5_000),
        _FakeInfo("blk.0.ssm_out.weight", 3_000),
        _FakeInfo("blk.0.ssm_a", 100),
        _FakeInfo("blk.0.attn_norm.weight", 500),
        _FakeInfo("token_embd.weight", 2_000),
        _FakeInfo("output.weight", 2_000),
    ]
    r = coverage_report(infos)
    total = 10_000 + 10_000 + 5_000 + 3_000 + 100 + 500 + 2_000 + 2_000
    esp = 10_000 + 10_000 + 5_000 + 3_000
    assert r["total_params"] == total
    assert r["sparsifiable_params"] == esp
    assert abs(r["coverage"] - esp / total) < 1e-9
    assert r["params_by_role"]["ssm.out"] == 3_000
    assert r["params_by_role"]["ssm.core"] == 100


def test_seleccion_de_bloques_candidatos():
    names = [
        "blk.0.ffn_gate.weight",
        "blk.1.ffn_gate.weight",
        "blk.2.ffn_gate.weight",
        "blk.0.attn_q.weight",
    ]
    sel = select_candidate_blocks(names, role="ffn.gate", max_blocks=2)
    assert sel == ["blk.0.ffn_gate.weight", "blk.1.ffn_gate.weight"]
    assert select_candidate_blocks(names, role="attn.q") == ["blk.0.attn_q.weight"]


def test_inicializacion_estricta_desde_la_base():
    d_in, d_out = 8, 4
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, (d_out, d_in)).astype(np.float32)
    # Adyacencia con las conexiones (i=1,j=0) y (i=3,j=2) activas.
    total = d_in * d_out
    bits = np.zeros((total + 7) // 8, dtype=np.uint8)
    for conn in (1 * d_out + 0, 3 * d_out + 2):
        bits[conn // 8] |= np.uint8(1 << (conn % 8))
    w = init_weights_from_base(base, d_in, d_out, bits)
    assert w[0, 1] == base[0, 1]
    assert w[2, 3] == base[2, 3]
    assert w[0, 0] == 0.0  # inactiva
    assert w[0, 2] == 0.0


def test_adyacencia_identidad_densa():
    bits = identity_adjacency(4, 3)
    active = sum(int(b).bit_count() for b in bits)
    assert active == 12


def test_lote_de_calibracion_determinista():
    a = build_calibration_batch(128, seed=42)
    b = build_calibration_batch(128, seed=42)
    c = build_calibration_batch(128, seed=43)
    assert a == b
    assert a != c
    assert len(a) == 128
    assert all("c42" in t or "c43" in t for t in a)


def test_varianza_de_activacion_y_canales_calientes():
    rng = np.random.default_rng(1)
    h0 = rng.normal(0, [1.0, 2.0, 0.5, 5.0], (64, 4)).astype(np.float32)
    var = activation_variance(h0)
    assert var.shape == (4,)
    hot = hot_channel_indices(var, 2)
    assert len(hot) == 2
    assert hot[0] == 3  # mayor varianza


def test_backend_sintetico_determinista():
    rt = SyntheticTeacherRuntime(seed=7)
    texts = build_calibration_batch(32)
    h1 = rt.capture_block("blk.0.ffn_gate.weight", 64, 48, texts)
    h2 = rt.capture_block("blk.0.ffn_gate.weight", 64, 48, texts)
    assert h1.x.shape == (32, 64)
    assert h1.h0.shape == (32, 48)
    np.testing.assert_array_equal(h1.x, h2.x)
    np.testing.assert_array_equal(h1.h0, h2.h0)
    assert h1.fisher_diag.shape == (48,)
