"""Precómputo offline de hooks de activación (Paso 2 del diseño).

* `build_calibration_batch` — lote determinista de `B` textos de alta entropía
  semántica (código, matemáticas y prosa literaria) vía plantillas + semilla.
* `activation_variance` — diagonal de la Fisher empírica (`Var(H0)` por canal),
  que alimenta el subespacio activo de CMA-ES (Fase 1, decisión D6).
* `TeacherRuntime` — abstracción del runtime del profesor (modelo de 30B).
  El backend real (llama-cpp-python/ggml, D8) se conecta aquí; el backend
  sintético permite testear el pipeline sin el modelo físico.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BlockHook:
    """Activaciones precómputadas de un bloque (guardadas en RAM pinned)."""

    block_name: str
    x: np.ndarray  # [B, d_in] activación de entrada
    h0: np.ndarray  # [B, d_out] activación de salida del profesor
    fisher_diag: np.ndarray  # [d_out] Var(H0) por canal

    def bytes_approx(self) -> int:
        return int(self.x.nbytes + self.h0.nbytes + self.fisher_diag.nbytes)


class TeacherRuntime(ABC):
    """Abstracción del runtime que ejecuta el modelo profesor capa a capa."""

    @abstractmethod
    def capture_block(
        self,
        block_name: str,
        d_in: int,
        d_out: int,
        texts: list[str],
    ) -> BlockHook:
        """Ejecuta el bloque del modelo y devuelve X, H0 y Var(H0)."""

    def capture_many(
        self,
        blocks: list[tuple[str, int, int]],
        texts: list[str],
    ) -> dict[str, BlockHook]:
        return {
            name: self.capture_block(name, d_in, d_out, texts)
            for name, d_in, d_out in blocks
        }


class SyntheticTeacherRuntime(TeacherRuntime):
    """Backend sintético determinista (tests): bloque lineal aleatorio."""

    def __init__(self, seed: int = 0, hidden: int = 512) -> None:
        self.seed = seed
        self.hidden = hidden

    @staticmethod
    def _name_seed(name: str) -> int:
        # `hash()` es aleatorio por proceso en Python; usar sha256 determinista.
        import hashlib

        return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")

    def capture_block(self, block_name, d_in, d_out, texts):
        rng = np.random.default_rng(self.seed + self._name_seed(block_name))
        b = len(texts)
        x = rng.normal(0, 1, (b, d_in)).astype(np.float32)
        w0 = rng.normal(0, 1 / np.sqrt(d_in), (d_out, d_in)).astype(np.float32)
        h0 = (x @ w0.T).astype(np.float32)
        fisher = h0.var(axis=0).astype(np.float32)
        return BlockHook(block_name, x, h0, fisher)


class LlamaCppTeacherRuntime(TeacherRuntime):
    """Backend real (Paso 2 con el modelo de 30B) vía `llama-cpp-python`.

    Requiere instalar `llama-cpp-python` y el GGUF del modelo base. La captura
    de activaciones intermedias necesita una build con hooks de capa; hasta que
    esté disponible se resuelve la integración (decisión D8).
    """

    def __init__(self, model_path: str, n_gpu_layers: int = 0) -> None:
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self._llm = None

    def _lazy_init(self):
        if self._llm is None:
            try:
                from llama_cpp import Llama  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "llama-cpp-python no está instalado; instálalo con "
                    "`pip install llama-cpp-python` para el backend real (D8)"
                ) from exc
            self._llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
            )
        return self._llm

    def capture_block(self, block_name, d_in, d_out, texts):
        self._lazy_init()
        raise NotImplementedError(
            "la captura de activaciones intermedias con llama-cpp-python requiere "
            "hooks de capa; ver decisión D8 en docs/decisiones.md"
        )


# ------------------------------------------------------------------------- batch

_CODE = [
    "def mergesort(a):",
    "let xs: Vec<i32> = v.iter().map(|x| x * 2).collect();",
    "if (errno == EAGAIN) { retry(fd, buf, n); }",
    "for i in range(len(a) - 1, 0, -1):",
    "SELECT id, name FROM users WHERE active = 1;",
    "float acc = 0.0f; for (int i = 0; i < n; i++) acc += x[i];",
]

_MATH = [
    "f(x) = x^3 - 2x + 1",
    "e^{i\\pi} + 1 = 0",
    "\\int_0^1 x^2 dx = 1/3",
    "gcd(a, b) = gcd(b, a mod b)",
    "d = \\sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}",
]

_PROSE = [
    "La luz se filtraba entre los pinos mientras el sendero se perdía en la niebla.",
    "The old bridge had witnessed a thousand crossings and held each memory.",
    "El viento arrastraba hojas secas por la plaza desierta al anochecer.",
]


def build_calibration_batch(b: int = 128, seed: int = 42) -> list[str]:
    """Lote determinista de `b` textos de alta entropía (código+mate+prosa)."""
    rng = np.random.default_rng(seed)
    pool = _CODE + _MATH + _PROSE
    out: list[str] = []
    for i in range(b):
        base = pool[i % len(pool)]
        variant = rng.integers(0, 100000)
        out.append(f"{base} #c{seed}-{i}-{variant}")
    return out


def activation_variance(h0: np.ndarray) -> np.ndarray:
    """Diagonal de la Fisher empírica: Var(H0) por canal (`[d_out]`)."""
    h0 = np.asarray(h0, np.float32)
    return h0.var(axis=0, dtype=np.float32)


def hot_channel_indices(fisher_diag: np.ndarray, d: int) -> np.ndarray:
    """Índices de los `d` canales calientes (Fase 1 del subespacio activo)."""
    return np.argsort(-fisher_diag, kind="stable")[:d].astype(np.int64)
