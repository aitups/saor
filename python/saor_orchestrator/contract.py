"""Contrato de Fase 2 (Paso 7 del diseño): criterios de cierre del experimento.

Las **3 reglas del contrato**:

1. **Distancia arquitectónica significativa:** `D_arch >= 0.4` (Hamming
   normalizado = esparcidad del candidato, decisión D5).
2. **Rendimiento superior al baseline:** fidelidad funcional (CKA vs. el
   profesor) por encima de un umbral; con el modelo real se exige
   `KL <= 0.05` y `ARC_search > ARC_base` (holdout ciego ARC-Challenge/GSM8K).
3. **Comportamiento responsivo (no dormant):** la salida del bloque candidato
   conserva energía y varianza (no colapsa a ceros ni a una constante).

Los proxies estructurales/funcionales se computan aquí desde el **GGUF
disperso** (decisión D4) + los hooks `X, H0` del Paso 2. La validación final
con el modelo real (KL, ARC, GSM8K) requiere el runtime del profesor (D8).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from saor_orchestrator.reference.cka import centered_cka, gram_matrix
from saor_orchestrator.hooks.gguf_audit import SparseBlock


@dataclass(frozen=True)
class ContractResult:
    """Resultado de la evaluación del contrato de Fase 2."""

    d_arch: float
    cka: float
    response_ratio: float
    kl_proxy: float
    rules: dict[str, bool]
    verdict: bool

    def report(self) -> dict:
        return {
            "d_arch": self.d_arch,
            "rule_distancia": self.rules["distancia"],
            "cka": self.cka,
            "rule_fidelidad": self.rules["fidelidad"],
            "response_ratio": self.response_ratio,
            "rule_no_dormant": self.rules["no_dormant"],
            "kl_proxy": self.kl_proxy,
            "veredicto": self.verdict,
        }


def dense_from_sparse(block: SparseBlock) -> np.ndarray:
    """Reconstruye la matriz `[d_out, d_in]` enmascarada desde el bit-tensor."""
    w = np.zeros((block.d_out, block.d_in), np.float32)
    w_idx = 0
    for i in range(block.d_in):
        for j in range(block.d_out):
            conn = i * block.d_out + j
            if int(block.adjacency[conn // 8]) & (1 << (conn % 8)):
                w[j, i] = block.weights[w_idx]
                w_idx += 1
    return w


def kl_proxy(h0: np.ndarray, h1: np.ndarray, eps: float = 1e-6) -> float:
    """Divergencia KL simétrica aproximada entre las salidas (proxy).

    Normaliza cada lote como distribución (softmax con temperatura) y calcula
    `0.5 * (KL(P||Q) + KL(Q||P))`. Con el modelo real se usa la KL real sobre
    logits (contrato `<= 0.05`).
    """
    p = np.exp(h0 - h0.max(axis=1, keepdims=True))
    q = np.exp(h1 - h1.max(axis=1, keepdims=True))
    p = p / (p.sum(axis=1, keepdims=True) + eps)
    q = q / (q.sum(axis=1, keepdims=True) + eps)
    kpq = float((p * np.log((p + eps) / (q + eps))).sum(axis=1).mean())
    kqp = float((q * np.log((q + eps) / (p + eps))).sum(axis=1).mean())
    return 0.5 * (kpq + kqp)


def evaluate_contract(
    block: SparseBlock,
    hook_x: np.ndarray,
    hook_h0: np.ndarray,
    cka_threshold: float = 0.90,
    response_floor: float = 0.01,
) -> ContractResult:
    """Evalúa las 3 reglas del contrato para el bloque consolidado."""
    hook_x = np.asarray(hook_x, np.float32)
    hook_h0 = np.asarray(hook_h0, np.float32)

    d_arch = float(block.sparsity())
    w = dense_from_sparse(block)
    h1 = (hook_x @ w.T).astype(np.float32)
    cka = centered_cka(gram_matrix(hook_h0), gram_matrix(h1))
    norm_h0 = float(np.linalg.norm(hook_h0))
    response_ratio = float(np.linalg.norm(h1) / norm_h0) if norm_h0 > 0 else 0.0
    kl = kl_proxy(hook_h0, h1)

    rules = {
        "distancia": d_arch >= 0.4,
        "fidelidad": cka >= cka_threshold,
        "no_dormant": response_ratio >= response_floor,
    }
    verdict = all(rules.values())
    return ContractResult(
        d_arch=d_arch,
        cka=cka,
        response_ratio=response_ratio,
        kl_proxy=kl,
        rules=rules,
        verdict=verdict,
    )
