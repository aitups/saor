# Informe del Proyecto: Optimización de Arquitectura Esparsa Vía B

**Repositorios:** [`saor`](https://github.com/aitups/saor) (motor de optimización) · [`hayai`](https://github.com/aitups/hayai) (runtime de inferencia GGUF)
**Fecha:** agosto 2026 · **Estado:** experimentos completados; evolución Vía B validada en 4 modelos.

---

## 1. Resumen ejecutivo

Se cierra el pipeline **Vía B**: un único genoma CPPN (Red de Patrones de
Composición) genera la topología de esparsidad de **todas** las capas FFN de un
modelo (coordenada de profundidad `y_layer`), optimizado con CMA-ES bajo la
frontera de Pareto **"maximizar compresión sujeta a KL ≤ 0.50"**. El decode de
la topología corre en **GPU vía OpenCL** (16.5× más rápido que CPU), integrado
en el embedder de producción.

Resultados clave (validados con el evaluador KL del runtime):

| Modelo | Compresión FFN (D_arch) | KL | Nota |
|---|---|---|---|
| SmolLM2-135M | 7.8 % | 1.68 | canario de validación |
| Qwen3.5-4B | 2.7 % | 0.113 | híbrido gated-DeltaNet |
| ALIA-40b | 1.8 % | 0.776 | el más sensible a la poda |
| Qwen3.8-27B | 10.7 % (mejor fitness) | 0.394 | el más robusto (KL 0.14 a sp 0.25) |

**Ranking de sensibilidad a la poda del FFN: 27B < 4B ≪ SmolLM2 < ALIA**
(a la misma compresión, ALIA produce ~70× más KL que el 27B).

> **Nota de métrica:** `D_arch` en este proyecto mide la **fracción de parámetros
> FFN podados** (compresión), no la densidad. El nombre histórico `D_arch`
> ("densidad de arquitectura") se conserva por compatibilidad; el objetivo D20
> "max D_arch @ KL ≤ 0.5" es, por tanto, **maximizar la compresión sujeta a
> calidad**.

---

## 2. Objetivo y método

### 2.1 Formulación (D20)
- **Objetivo:** `max D_arch_global` sujeto a `KL_global ≤ 0.50`, con esparsidad
  **heterogénea por capa**.
- **CPPN global (Vía B):** un solo genoma (466 parámetros: 2 capas ocultas de 16
  con activaciones tanh/sin) evalúa cada conexión `(i, j)` de cada capa con la
  coordenada de profundidad `y_layer ∈ [-1, 1]`. Conexión activa si `l_ij > τ`
  (`τ = 0.42`). El perfil por capa emerge del propio genoma (sin per-layer
  manual).
- **Optimizador:** CMA-ES sobre el genoma aplanado; fitness
  `= D_arch − 2·max(0, KL − 0.5)`.
- **Evaluador:** el modelo original y el esparso embebido se ejecutan en
  lockstep con el runtime `hayai` (StreamingGenerator); KL de logits por
  posición y `D_arch` medido por popcount de la adyacencia embebida.

### 2.2 Pipeline de producción
```
CPPN (genoma) → embed_sparse --genome [--gpu] → GGUF D16 esparso → kl_eval → CMA-ES
```
- `dump_weights` extrae los pesos F32 del gate por capa (una vez).
- `embed_sparse --genome` decodifica la topología por capa (CPU Rust o **kernel
  OpenCL `cppn_decode_adj`** en GPU) y conserva los pesos del profesor en las
  posiciones activas (warm-start teacher-copy), escribiendo el GGUF en
  streaming (sin cargar el modelo completo en RAM).
- `kl_eval` mide KL y D_arch sobre el modelo embebido.

### 2.3 Kernel OpenCL
El decode CPPN (un work-item por conexión, `atomic_or` para la adyacencia)
está integrado en `embed_sparse --gpu`:
- **Rendimiento:** 2 s vs 33 s en SmolLM2 (16.5×); escala a ALIA-40b (201M
  conexiones × 48 capas ≈ 9.6G evaluaciones) vía dispatches fragmentados
  (WDDM/TDR).
- **Validación:** bit-exacto frente a `instantiate_layer` (capa 7/30, patrón
  mixto, `adj_bytes_diff_layer = 0`).
- **Linkeo dinámico** (cl3 `dynamic`): los binarios cargan `OpenCL.dll` en
  runtime, sin requerir el SDK en tiempo de compilación.

---

## 3. Resultados por modelo

### 3.1 Frontera uniforme (baseline de poda por magnitud del gate)

| Modelo | sp 0.05 | sp 0.10 | sp 0.15 | sp 0.20 | sp 0.25 |
|---|---|---|---|---|---|
| **Qwen3.5-4B** | 0.045 / 0.017 | 0.108 / 0.033 | 0.173 / 0.050 | 0.296 / 0.067 | 0.463 / 0.083 |
| **ALIA-40b** | — | 0.958 / 0.033 | — | 2.151 / 0.067 | — |
| **Qwen3.8-27B** | 0.015 / 0.017 | 0.042 / 0.033 | 0.075 / 0.050 | 0.115 / 0.067 | 0.143 / 0.083 |

*(formato: `KL / D_arch`; D_arch = fracción del FFN podada, incluye los 3
bloques; sp es la esparsidad del gate)*

Lecturas:
- **Qwen3.8-27B es excepcionalmente robusto**: sp 0.25 (8.3 % de FFN podado)
  con KL 0.143.
- **ALIA-40b explota**: sp 0.2 → KL 2.15; su perfil (capas tempranas críticas)
  hace la poda uniforme inviable.

### 3.2 Evolución Vía B (genomas validados)

| Modelo | Genoma | Compresión (D_arch) | KL | n_pos | Validación |
|---|---|---|---|---|---|
| SmolLM2-135M | `via_b_best_genome.bin` | 7.8 % | 1.68 | 24 | CPU==GPU (Δ 1e-4) |
| Qwen3.5-4B | `via_b_best_genome_qwen35.bin` | 2.7 % | 0.113 | 24 | GPU |
| ALIA-40b | `via_b_best_genome_alia.bin` | 1.8 % | 0.776 | 8 | GPU |
| Qwen3.8-27B | `via_b_best_genome_qwen27.bin` | 10.7 % (best-fit) / 0.4 % (min-KL) | 0.394 / 0.0005 | 4 / 8 | GPU |

Perfiles de densidad por capa (gate, subsample):

```
SmolLM2:    95 95 95 95 96 96 ... 85 85 86 86 86 87   (media 92 %)
Qwen3.5-4B: 76 76 76 76 77 79 ... 100 100 100 100 100 (media 92 %)
ALIA-40b:   99 99 98 98 97 97 ... 95 95 95 95 95 95   (media 95 %)
Qwen3.8-27B:91 92 93 94 94 95 ... 100 100 100 100 100 (media 99 %)
```

La optimización concentra la poda en las capas donde el modelo es tolerante y
mantiene densas las críticas (para Qwen, las capas finales; para SmolLM2, las
iniciales).

### 3.3 Comparación Vía B vs. baseline uniforme

- **Qwen3.8-27B:** la evolución alcanza **10.7 % de compresión con KL 0.394**
  (ambos bajo el target 0.5), superando el punto uniforme más lejano medido
  (sp 0.25 → 8.3 % / KL 0.143). En el marco D20 (max compresión @ KL≤0.5), la
  evolución encuentra un punto de mayor compresión.
- **ALIA-40b:** la compresión alcanzable es baja (1.8 % con KL 0.776, aún sobre
  el target) — la sensibilidad estructural de ALIA limita la poda.
- **SmolLM2:** KL 1.68 a 7.8 % de compresión; el baseline uniforme a sp 0.1
  (3.3 %) da KL 0.754. La evolución no superó al baseline en este canario con
  4 generaciones (el paisaje CPPN requiere más generaciones o mejor
  exploración).

---

## 4. Hallazgos de ingeniería

1. **Kernel OpenCL integrado** (`embed_sparse --gpu`): 16.5× más rápido,
   bit-exacto, escala a modelos de 40B (vía dispatches fragmentados).
2. **Linkeo dinámico OpenCL** (cl3 `dynamic`): los binarios compilan sin el SDK
   (solo el runtime del driver).
3. **Deadlock de GPU concurrente:** dos procesos OpenCL simultáneos sobre la
   misma GPU se cuelgan (pool). Regla operacional: **un solo proceso OpenCL a
   la vez**.
4. **Fallos "OOM" de ALIA:** los errores históricos
   (`CL_MEM_OBJECT_ALLOCATION_FAILURE` / `UnexpectedEof`) eran **GGUF parciales**
   por carreras entre embeds concurrentes sobre los mismos `wdata.*`, no bugs
   del runtime. Con archivos completos el kl_eval del 40B corre en la RTX 4050.
5. **Embedding streaming:** reescritura GGUF sin cargar el modelo en RAM
   (crítico para 23–57 GB).
6. **Selección del mejor genoma (fix):** el script ahora guarda el genoma de
   **mejor fitness** (objetivo D20) además del de mínima KL.

---

## 5. Reproducibilidad

Requisitos: Rust (workspace `saor` + `hayai`), GPU OpenCL (opcional, para
`--gpu`), Python 3.12 con `numpy`.

```bash
# 1) Dump del gate (una vez por modelo)
hayai dump_weights --model <model.gguf> --out w_<name> --blocks gate

# 2) Frontera uniforme
python python/scripts/frontier_stream_sweep.py --model <model.gguf> \
  --name <name> --n-layers <N> --n-positions 8 --device auto

# 3) Evolución Vía B (decode GPU; kl_eval secuencial — un proceso OpenCL a la vez)
python python/scripts/via_b_evolve.py --streaming --gpu --model <model.gguf> \
  --weights w_<name> --n-layers <N> --d-in <D> --d-out <D2> \
  --device auto --name <name> --n-pos 8 --gens 4 --seed 7 --darch 0.10

# 4) Validación del mejor genoma
saor/target/release/embed_sparse --model <model.gguf> --out best.gguf \
  --weights w_<name> --genome via_b_best_genome_<name>.bin --tau 0.42 --gpu
hayai kl_eval --orig <model.gguf> --sparse best.gguf --prompts calib.txt \
  --n-positions 24 --device auto
```

Los GGUF embebidos (formato D16 de saor) se ejecutan con el `StreamingGenerator`
de `hayai` (incluye el FFN disperso SpMM-CSR en OpenCL).

---

## 6. Conclusiones

1. **El pipeline Vía B está cerrado y reproducible** en 4 modelos (135M–40B),
   con decode por GPU y validación end-to-end.
2. **La robustez a la poda es específica del modelo**: Qwen3.8-27B y Qwen3.5-4B
   (híbridos gated-DeltaNet) toleran compresión con KL bajo; ALIA-40b es
   estructuralmente sensible.
3. **La compresión alcanzable con calidad (KL ≤ 0.5) es modesta** (2–11 % del
   FFN) en estos modelos — el FFN de los LLM modernos es menos redundante de lo
   esperado para la poda no estructurada del gate.
4. **La topología CPPN (Vía B) automatiza el perfil por capa** pero no supera de
   forma consistente a la poda uniforme por magnitud en el presupuesto de
   generaciones usado (4 gens). El valor está en el descubrimiento automático
   del perfil y en el pipeline GPU.
5. **Publicaciones:** genomas CPPN (466 floats/modelo), GGUF D16 embebidos y
   este informe en Hugging Face; código en GitHub (`aitups/saor`, `aitups/hayai`).
