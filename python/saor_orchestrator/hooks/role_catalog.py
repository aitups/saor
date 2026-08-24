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
    (re.compile(r"blk\.(\d+)\.attn_q\.weight"), "attn.q"),
    (re.compile(r"blk\.(\d+)\.attn_k\.weight"), "attn.k"),
    (re.compile(r"blk\.(\d+)\.attn_v\.weight"), "attn.v"),
    (re.compile(r"blk\.(\d+)\.attn_output\.weight"), "attn.out"),
    (re.compile(r"blk\.(\d+)\.attn_norm\.weight"), "norm.attn"),
    (re.compile(r"blk\.(\d+)\.ffn_norm\.weight"), "norm.ffn"),
    (re.compile(r"blk\.(\d+)\.attn_q_norm\.weight"), "norm.q"),
    (re.compile(r"blk\.(\d+)\.attn_k_norm\.weight"), "norm.k"),
    (re.compile(r"model\.layers\.(\d+)\.mlp\.gate_proj\.weight"), "ffn.gate"),
    (re.compile(r"model\.layers\.(\d+)\.mlp\.up_proj\.weight"), "ffn.up"),
    (re.compile(r"model\.layers\.(\d+)\.mlp\.down_proj\.weight"), "ffn.down"),
    (re.compile(r"model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight"), "attn"),
    (re.compile(r"output_norm\.weight"), "norm.output"),
    (re.compile(r"token_embd\.weight"), "embd"),
    (re.compile(r"output\.weight"), "lm_head"),
]


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
