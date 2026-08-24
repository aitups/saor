"""Auditoría de cabeceras GGUF sin cargar los pesos (Paso 1 del diseño).

Parsea magic/versión, metadatos KV y los tensor infos (nombre, forma, tipo
GGML, offset) leyendo **solo la cabecera** del archivo (streaming por chunks de
1 MiB). Produce un perfil de parámetros del modelo base: total de parámetros,
cuantización, tensores dominantes y desglose FFN vs atención.

Esto alimenta el `ROLE_CATALOG` y la selección de bloques candidatos sin tener
que mapear el modelo completo en RAM (crítico para un modelo de ~30B en 15 GB).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

GGUF_MAGIC = 0x4655_4747
GGUF_VERSION = 3

# Tipos de valor GGUF.
GGUF_UINT8 = 0
GGUF_INT8 = 1
GGUF_UINT16 = 2
GGUF_INT16 = 3
GGUF_UINT32 = 4
GGUF_INT32 = 5
GGUF_FLOAT32 = 6
GGUF_BOOL = 7
GGUF_STRING = 8
GGUF_ARRAY = 9
GGUF_UINT64 = 10
GGUF_INT64 = 11
GGUF_FLOAT64 = 12

# Tipos GGML: (numel por bloque, bytes por bloque). Densos: (1, elem_bytes).
GGML_TYPE: dict[str, tuple[int, int]] = {
    "F32": (1, 4), "F16": (1, 2), "Q4_0": (32, 18), "Q4_1": (32, 20),
    "Q5_0": (32, 22), "Q5_1": (32, 24), "Q8_0": (32, 34), "Q8_1": (32, 40),
    "Q2_K": (256, 84), "Q3_K": (256, 110), "Q4_K": (256, 144),
    "Q5_K": (256, 176), "Q6_K": (256, 210), "Q8_K": (256, 292),
    "IQ2_XXS": (256, 36), "IQ2_XS": (256, 40), "IQ1_S": (256, 36),
    "IQ4_NL": (32, 18), "IQ3_S": (256, 56), "IQ2_S": (256, 46),
    "IQ4_XS": (256, 64), "I8": (1, 1), "I16": (1, 2), "I32": (1, 4),
    "I64": (1, 8), "F64": (1, 8),
}

# Tipos GGML (id -> nombre). Sigue el enum GGML **actual** (llama.cpp moderno /
# hayai): los IQ ocupan 16-23 e `I8 = 24` (antes `I8 = 16`). Crítico para leer
# correctamente el bit-tensor `ffn_dag_adjacency` del GGUF disperso de saor.
GGML_TYPE_BY_ID: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
}


class _Reader:
    """Lector incremental de la cabecera (nunca carga el archivo completo)."""

    def __init__(self, f) -> None:
        self.f = f
        self.buf = bytearray()
        self.pos = 0

    def _fill(self, n: int) -> None:
        while len(self.buf) - self.pos < n:
            chunk = self.f.read(1 << 20)
            if not chunk:
                raise EOFError("cabecera GGUF truncada")
            self.buf += chunk

    def _take(self, n: int) -> bytes:
        self._fill(n)
        out = bytes(self.buf[self.pos : self.pos + n])
        self.pos += n
        return out

    def u32(self) -> int:
        return struct.unpack("<I", self._take(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self._take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self._take(8))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self._take(4))[0]

    def f64(self) -> float:
        return struct.unpack("<d", self._take(8))[0]

    def string(self) -> str:
        n = self.u64()
        return self._take(n).decode("utf-8")


def _read_kv_value(r: _Reader, vtype: int):
    if vtype == GGUF_UINT64:
        return r.u64()
    if vtype == GGUF_INT64:
        return struct.unpack("<q", r._take(8))[0]
    if vtype == GGUF_FLOAT32:
        return r.f32()
    if vtype == GGUF_FLOAT64:
        return r.f64()
    if vtype == GGUF_BOOL:
        return bool(r._take(1)[0])
    if vtype == GGUF_STRING:
        return r.string()
    if vtype == GGUF_ARRAY:
        elem_type = r.u32()
        count = r.u64()
        return [_read_kv_value(r, elem_type) for _ in range(count)]
    sizes = {
        GGUF_UINT8: 1, GGUF_INT8: 1, GGUF_UINT16: 2,
        GGUF_INT16: 2, GGUF_UINT32: 4, GGUF_INT32: 4,
    }
    size = sizes.get(vtype)
    if size is None:
        raise ValueError(f"tipo de KV no soportado: {vtype}")
    return r._take(size)


@dataclass
class TensorInfo:
    """Información de un tensor del modelo (sin datos)."""

    name: str
    shape: tuple[int, ...]
    ggml_type: str
    offset: int

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def nbytes(self) -> int:
        blk_numel, blk_bytes = GGML_TYPE.get(self.ggml_type, (1, 4))
        blocks = (self.numel + blk_numel - 1) // blk_numel
        return blocks * blk_bytes


@dataclass
class GgufHeader:
    """Cabecera GGUF auditada (metadata + tensor infos, sin pesos)."""

    path: Path
    version: int
    kv: dict = field(default_factory=dict)
    tensors: list[TensorInfo] = field(default_factory=list)
    header_end: int = 0
    total_tensor_bytes: int = 0

    @property
    def total_params(self) -> int:
        return sum(t.numel for t in self.tensors)

    @property
    def quantized_ratio(self) -> float:
        """Fracción de bytes en tensores cuantizados (≠ F16/F32)."""
        dense = sum(t.nbytes() for t in self.tensors if t.ggml_type in ("F16", "F32"))
        return 1.0 - dense / max(1, self.total_tensor_bytes)

    def by_prefix(self, prefix: str) -> list[TensorInfo]:
        return [t for t in self.tensors if t.name.startswith(prefix)]

    def profile(self) -> dict:
        """Resumen del perfil de parámetros (Paso 1 del diseño)."""
        by_role: dict[str, int] = {}
        for t in self.tensors:
            for role in ("ffn_", "attn_", "attn_norm", "ffn_norm", "token_embd", "output"):
                if role in t.name:
                    by_role[role.rstrip("_")] = by_role.get(role.rstrip("_"), 0) + t.numel
                    break
        top = sorted(self.tensors, key=lambda t: t.nbytes(), reverse=True)[:8]
        return {
            "path": str(self.path),
            "version": self.version,
            "n_tensors": len(self.tensors),
            "total_params": self.total_params,
            "total_tensor_bytes": self.total_tensor_bytes,
            "quantized_ratio": round(self.quantized_ratio, 4),
            "params_by_role": by_role,
            "top_tensors": [
                {
                    "name": t.name,
                    "shape": list(t.shape),
                    "type": t.ggml_type,
                    "bytes": t.nbytes(),
                }
                for t in top
            ],
        }


def read_gguf_header(path: str | Path) -> GgufHeader:
    """Parsea la cabecera GGUF v3 de un archivo sin cargar los pesos."""
    path = Path(path)
    with open(path, "rb") as f:
        r = _Reader(f)
        magic = r.u32()
        if magic != GGUF_MAGIC:
            raise ValueError(f"magic inválido: {magic:#x}")
        version = r.u32()
        if version != GGUF_VERSION:
            raise ValueError(f"versión no soportada: {version}")
        tensor_count = r.u64()
        kv_count = r.u64()
        kv: dict = {}
        for _ in range(kv_count):
            key = r.string()
            vtype = r.u32()
            kv[key] = _read_kv_value(r, vtype)
        tensors: list[TensorInfo] = []
        for _ in range(tensor_count):
            name = r.string()
            n_dim = r.u32()
            shape = tuple(r.u64() for _ in range(n_dim))
            gtype = GGML_TYPE_BY_ID.get(r.i32(), "?")
            offset = r.u64()
            tensors.append(TensorInfo(name, shape, gtype, offset))
        header_end = r.pos

    total_bytes = sum(t.nbytes() for t in tensors)
    return GgufHeader(
        path=path,
        version=version,
        kv=kv,
        tensors=tensors,
        header_end=header_end,
        total_tensor_bytes=total_bytes,
    )


@dataclass
class SparseBlock:
    """Bloque disperso leído de un GGUF `saor.*` (decisión D4)."""

    d_in: int
    d_out: int
    tau: float
    genome: np.ndarray  # [~32K]
    adjacency: np.ndarray  # uint8, bit-tensor ffn_dag_adjacency
    weights: np.ndarray  # float32, pesos activos

    def active_connections(self) -> int:
        return int(sum(int(b).bit_count() for b in self.adjacency))

    def sparsity(self) -> float:
        return 1.0 - self.active_connections() / (self.d_in * self.d_out)


def read_sparse_block(path: str | Path) -> SparseBlock:
    """Lee los datos del bloque disperso (adyacencia + pesos) del GGUF."""
    import numpy as np

    h = read_gguf_header(path)
    by_name = {t.name: t for t in h.tensors}
    if "ffn_dag_adjacency" not in by_name or "ffn_dag_weights" not in by_name:
        raise ValueError("el GGUF no contiene el bloque disperso saor (ffn_dag_*)")

    adj = by_name["ffn_dag_adjacency"]
    w = by_name["ffn_dag_weights"]
    # Espec GGUF v3: offsets relativos a la sección de datos alineada a 32.
    data_offset = (h.header_end + 31) // 32 * 32
    with open(path, "rb") as f:
        f.seek(data_offset + adj.offset)
        adj_bytes = f.read(adj.nbytes())
        f.seek(data_offset + w.offset)
        w_raw = f.read(w.nbytes())

    return SparseBlock(
        d_in=int(h.kv["saor.d_in"]),
        d_out=int(h.kv["saor.d_out"]),
        tau=float(h.kv["saor.tau"]),
        genome=np.asarray(h.kv["saor.genome"], np.float32),
        adjacency=np.frombuffer(adj_bytes, np.uint8).copy(),
        weights=np.frombuffer(w_raw, np.float32).copy(),
    )


