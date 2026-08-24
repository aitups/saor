"""Catálogo de roles (`ROLE_CATALOG`) para el Paso 1 del diseño.

Clasifica los tensores del modelo base por **regex de profundidad** y permite:

* `classify(name)` — rol + índice de capa de un tensor.
* `select_candidate_blocks(...)` — selección de bloques candidatos (FFN o
  proyección) a convertir en topologías no dirigidas.
* `init_weights_from_base(...)` — **inicialización estricta**: copia los pesos
  del bloque base en las posiciones activas del DAG candidato, de modo que el
  punto de partida sea funcional (proxy loss sana, no un cuerpo aleatorio).

Las convenciones de nombre cubren llama.cpp (`blk.N.ffn_gate.weight`,
`blk.N.attn_q.weight`) y transformers (`model.layers.N.mlp.gate_proj.weight`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# (regex de profundidad, rol). El grupo 1 captura el índice de capa cuando existe.
ROLE_CATALOG: list[tuple[re.Pattern, str]] = [
    (re.compile(r"blk\.(\d+)\.ffn_gate\.weight"), "ffn.gate"),
    (re.compile(r"blk\.(\d+)\.ffn_up\.weight"), "ffn.up"),
    (re.compile(r"blk\.(\d+)\.ffn_down\.weight"), "ffn.down"),
    (re.compile(r"blk\.(\d+)\.attn_qkv\.weight"), "attn.qkv"),
    (re.compile(r"blk\.(\d+)\.attn_gate\.weight"), "attn.gate"),
    (re.compile(r"blk\.(\d+)\.attn_q\.weight"), "attn.q"),
    (re.compile(r"blk\.(\d+)\.attn_k\.weight"), "attn.k"),
    (re.compile(r"blk\.(\d+)\.attn_v\.weight"), "attn.v"),
    (re.compile(r"blk\.(\d+)\.attn_output\.weight"), "attn.out"),
    # Bloques SSM (Mamba2/NextN) de Qwen: proyecciones lineales esparcibles.
    (re.compile(r"blk\.(\d+)\.ssm_out\.weight"), "ssm.out"),
    (re.compile(r"blk\.(\d+)\.ssm_alpha\.weight"), "ssm.alpha"),
    (re.compile(r"blk\.(\d+)\.ssm_beta\.weight"), "ssm.beta"),
    # Núcleo recurrente S6: NO esparcible (se conserva denso).
    (re.compile(r"blk\.(\d+)\.ssm_conv1d\.weight"), "ssm.core"),
    (re.compile(r"blk\.(\d+)\.ssm_conv1d\.bias"), "ssm.core"),
    (re.compile(r"blk\.(\d+)\.ssm_dt\.bias"), "ssm.core"),
    (re.compile(r"blk\.(\d+)\.ssm_norm\.weight"), "ssm.core"),
    (re.compile(r"blk\.(\d+)\.ssm_a$"), "ssm.core"),
    (re.compile(r"blk\.(\d+)\.attn_norm\.weight"), "norm.attn"),
    (re.compile(r"blk\.(\d+)\.ffn_norm\.weight"), "norm.ffn"),
    (re.compile(r"blk\.(\d+)\.attn_q_norm\.weight"), "norm.q"),
    (re.compile(r"blk\.(\d+)\.attn_k_norm\.weight"), "norm.k"),
    (re.compile(r"model\.layers\.(\d+)\.mlp\.gate_proj\.weight"), "ffn.gate"),
    (re.compile(r"model\.layers\.(\d+)\.mlp\.up_proj\.weight"), "ffn.up"),
    (re.compile(r"model\.layers\.(\d+)\.mlp\.down_proj\.weight"), "ffn.down"),
    (re.compile(r"model\.layers\.(\d+)\.self_attn\.q_proj\.weight"), "attn.q"),
    (re.compile(r"model\.layers\.(\d+)\.self_attn\.k_proj\.weight"), "attn.k"),
    (re.compile(r"model\.layers\.(\d+)\.self_attn\.v_proj\.weight"), "attn.v"),
    (re.compile(r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight"), "attn.out"),
    (re.compile(r"nextn\."), "nextn"),
    (re.compile(r"output_norm\.weight"), "norm.output"),
    (re.compile(r"token_embd\.weight"), "embd"),
    (re.compile(r"output\.weight"), "lm_head"),
]

# Roles esparcibles por el pipeline CPPN/SpMM (Fase 0, cobertura > 90%). El
# resto (norms, embd, lm_head, núcleo recurrente S6, nextn) permanece denso.
SPARSIFIABLE_ROLES = frozenset(
    {
        "ffn.gate",
        "ffn.up",
        "ffn.down",
        "attn.q",
        "attn.k",
        "attn.v",
        "attn.out",
        "attn.qkv",
        "attn.gate",
        "ssm.out",
        "ssm.alpha",
        "ssm.beta",
    }
)


def is_sparsifiable(name: str) -> bool:
    """¿El tensor es candidato a topología no dirigida (CPPN/SpMM)?"""
    r = classify(name)
    return bool(r and r.role in SPARSIFIABLE_ROLES)


def sparsifiable_tensors(tensor_names: list[str]) -> list[str]:
    """Todos los tensores esparcibles del modelo, en orden."""
    return [n for n in tensor_names if is_sparsifiable(n)]


def coverage_report(tensor_infos) -> dict:
    """Parámetros por rol y fracción esparcible (Fase 0 / objetivo >90%).

    `tensor_infos`: iterable con `.name` y `.numel` (p. ej. los `TensorInfo`
    de `gguf_audit.read_gguf_header`). Calcula el % del volumen total del
    modelo cubierto por el pipeline CPPN/SpMM (FFN + atención + SSM-proj).
    """
    total = 0
    sparsifiable = 0
    by_role: dict[str, int] = {}
    for t in tensor_infos:
        numel = int(t.numel)
        total += numel
        r = classify(t.name)
        role = r.role if r else "other"
        by_role[role] = by_role.get(role, 0) + numel
        if r and r.role in SPARSIFIABLE_ROLES:
            sparsifiable += numel
    return {
        "total_params": total,
        "sparsifiable_params": sparsifiable,
        "coverage": sparsifiable / max(1, total),
        "params_by_role": dict(sorted(by_role.items(), key=lambda kv: -kv[1])),
    }


@dataclass(frozen=True)
class Role:
    """Rol de un tensor del modelo base."""

    name: str
    role: str
    layer: int | None


def classify(name: str) -> Role | None:
    """Clasifica un nombre de tensor contra el catálogo de roles."""
    for pattern, role in ROLE_CATALOG:
        m = pattern.match(name)
        if m:
            layer = int(m.group(1)) if m.lastindex and m.group(1).isdigit() else None
            return Role(name=name, role=role, layer=layer)
    return None


def select_candidate_blocks(
    tensor_names: list[str],
    role: str = "ffn.gate",
    max_blocks: int | None = None,
) -> list[str]:
    """Selecciona bloques candidatos (tensores con el rol dado), en orden."""
    candidates = [n for n in tensor_names if (c := classify(n)) and c.role == role]
    if max_blocks is not None:
        candidates = candidates[:max_blocks]
    return candidates


def init_weights_from_base(
    base_weight: np.ndarray,
    d_in: int,
    d_out: int,
    adjacency_bits: np.ndarray,
) -> np.ndarray:
    """Copia estricta de los pesos del bloque base en el DAG candidato.

    El candidato esparcido conserva los pesos del profesor en las conexiones
    activas (`A_ij = 1`), manteniendo la función inicial (proxy loss sana).

    * `base_weight` — `[d_out, d_in]` en float32 (fila-mayor).
    * Devuelve `[d_out, d_in]` con `0.0` donde la conexión está inactiva.
    """
    base = np.asarray(base_weight, np.float32)
    if base.shape != (d_out, d_in):
        base = base.reshape(d_out, d_in)
    w = np.zeros((d_out, d_in), np.float32)
    for i in range(d_in):
        for j in range(d_out):
            conn = i * d_out + j
            if int(adjacency_bits[conn // 8]) & (1 << (conn % 8)):
                w[j, i] = base[j, i]
    return w


def identity_adjacency(d_in: int, d_out: int) -> np.ndarray:
    """Adyacencia densa (todas las conexiones activas) para el arranque."""
    total = d_in * d_out
    bits = np.zeros((total + 7) // 8, dtype=np.uint8)
    for conn in range(total):
        bits[conn // 8] |= np.uint8(1 << (conn % 8))
    return bits
