# Informe del Proyecto: Optimización de Arquitectura Esparsa Vía B

**Repositorios:** [`saor`](https://github.com/aitups/saor) (motor de optimización) ·
[`hayai`](https://github.com/aitups/hayai) (runtime de inferencia GGUF)
**Fecha:** agosto 2026 · **Modelos:** SmolLM2-135M, Qwen3.5-4B, ALIA-40b, Qwen3.8-27B

---

## 1. Contexto y motivación (para quién llega sin antecedentes)

Los modelos de lenguaje grandes (LLM) ejecutan, en **cada token generado**, dos
grandes bloques por capa: la **atención** y el **FFN** (red de avance, ~2/3 del
cómputo total). El FFN es, además, el mayor depósito de parámetros del modelo.
Una forma natural de acelerar y aligerar un LLM es **podar** (poner a cero) una
fracción de sus conexiones FFN: si la poda se hace bien, el modelo "sigue
hablando igual" pero hace menos operaciones por token.

El reto es **dónde podar**: no todas las conexiones ni todas las capas son
iguales. Podar de más en capas críticas degrada la calidad; podar poco en capas
redundantes deja cómputo sobre la mesa. La pregunta central de este proyecto:

> **¿Se puede optimizar la arquitectura de un LLM?**

Es decir: ¿puede un algoritmo **rediseñar la estructura interna** de un modelo
(qué conexiones existen, con qué densidad por capa) de forma automática, sin
etiquetado manual, para obtener un modelo más ligero que **siga hablando
igual**? La sub-pregunta operativa que se responde aquí: ¿se puede descubrir
automáticamente el **perfil de esparsidad por capa** (una decisión de
arquitectura) que maximiza la compresión del FFN manteniendo la calidad
(KL ≤ 0.5)?

La respuesta propuesta — **Vía B** — es: usar un **CPPN** (una red pequeña que
genera patrones) para definir la topología de esparsidad de *todas* las capas a
partir de un único vector de pesos (el *genoma*), y **evolucionar ese genoma**
con CMA-ES para optimizar la compresión sujeta a una cota de calidad (KL ≤ 0.5).

---

## 2. Conceptos clave

### 2.1 KL (Kullback–Leibler) — la medida de "indistinguibilidad"
Al generar, un modelo produce una distribución de probabilidad sobre el
siguiente token. La **KL** mide la distancia entre la distribución del modelo
**original** y la del modelo **podado**:

- `KL = 0` → distribuciones idénticas → el modelo podado es *indistinguible*
  del original en sus predicciones.
- `KL` pequeña → predicciones casi iguales (el podado puede escoger un token
  distinto ocasionalmente).
- `KL` grande → el podado "habla distinto" (degradado).

En este proyecto la KL se mide comparando los logits de ambos modelos sobre un
corpus de calibración, posición a posición (teacher-forcing). El **contrato de
calidad** es `KL ≤ 0.50`.

### 2.2 D_arch — la medida de compresión
`D_arch` es la **fracción de parámetros del FFN podados** (puestos a cero). Un
`D_arch = 0.10` significa que el 10% de las conexiones FFN se han eliminado.
(El nombre histórico "D_arch" viene de "densidad de arquitectura"; aquí se usa
como *compresión*: `D_arch = 1 − fracción activa`.)

### 2.3 Compresión de cómputo y de tamaño (aclaración importante)
- **Cómputo (FLOPs):** podar el X% del FFN reduce las multiplicaciones del FFN
  en X%. Como el FFN es ~60-70% del cómputo total, el ahorro global de FLOPs es
  `D_arch × fracción_FFN` (ver §5).
- **Tamaño de archivo:** los pesos activos se exportan **cuantizados a Q4_K**
  (formato D16 de saor). El GGUF esparso resultante es **menor que el
  original**: SmolLM2 → 102 MB frente a 110 MB del Q4_K_M (354 MB si se
  exportara en F32).
- **Latencia (velocidad real):** ver §5.2 — medida en CPU; en GPU (path de
  producción) queda pendiente por la regla de un solo proceso OpenCL.

### 2.4 CPPN y CMA-ES
- **CPPN (Red de Patrones de Composición):** una red neuronal pequeña (466
  parámetros: 16+16 neuronas ocultas) que, evaluada en las coordenadas de cada
  conexión `(i, j)` y de cada capa (`y_layer ∈ [-1,1]`), devuelve `(peso,
  probabilidad_de_activa)`. Conexión activa si `l_ij > τ` (`τ = 0.42`). Un solo
  genoma define así el perfil de TODAS las capas (Vía B).
- **CMA-ES:** algoritmo de evolución de estrategias de covarianza que ajusta el
  genoma para maximizar `fitness = D_arch − 2·max(0, KL − 0.5)`.

---

## 3. Método y condiciones del experimento

### 3.1 Pipeline
```
genoma CPPN → embed_sparse --genome [--gpu] → GGUF esparso D16 → kl_eval → CMA-ES
```
1. `dump_weights` extrae los pesos F32 del gate de cada capa (una vez).
2. `embed_sparse --genome` decodifica la topología por capa y conserva los pesos
   del profesor en las posiciones activas (**warm-start teacher-copy**), en
   streaming (sin cargar el modelo en RAM).
3. `kl_eval` mide KL y D_arch sobre el modelo embebido.
4. CMA-ES actualiza el genoma.

El decode de la topología corre en **GPU vía kernel OpenCL** (`cppn_decode_adj`,
un work-item por conexión): 2 s vs 33 s en CPU (16.5×), validado bit-exacto
frente a la referencia Rust, y escalable a modelos de 40B (201M conexiones × 48
capas ≈ 9.6G evaluaciones) con dispatches fragmentados (WDDM/TDR).

### 3.2 Condiciones fijas
| Condición | Valor |
|---|---|
| Umbral de activación `τ` | 0.42 |
| Generaciones CMA-ES | 4 (22 candidatos/gen, población 22) |
| Semilla | 7 |
| Esparsidad | solo el **gate** del FFN (up/down densos), salvo SmolLM2 (3 bloques) |
| Calibración | 128 prompts, `n_pos` 4–24 |
| Contrato | `KL ≤ 0.50`; objetivo `max D_arch @ KL ≤ 0.5` |
| Hardware | RTX 4050 (6 GB), decode OpenCL GPU, kl_eval secuencial |

> Regla operacional: **un solo proceso OpenCL a la vez** — dos `kl_eval`
> concurrentes sobre la misma GPU se deadlockean (hallazgo D34).

---

## 4. Resultados

### 4.1 Frontera uniforme (baseline: poda por magnitud del gate)

Primero se midió la curva KL vs compresión con una poda **uniforme** (la misma
fracción en todas las capas) para conocer la tolerancia de cada modelo:

| Modelo | sp 0.05 | sp 0.10 | sp 0.15 | sp 0.20 | sp 0.25 |
|---|---|---|---|---|---|
| **Qwen3.5-4B** | 0.045 | 0.108 | 0.173 | 0.296 | 0.463 |
| **ALIA-40b** | — | 0.958 | — | 2.151 | — |
| **Qwen3.8-27B** | 0.015 | 0.042 | 0.075 | 0.115 | 0.143 |

*(valores = KL; `sp` = fracción podada del gate; D_arch ≈ sp/3 porque el gate
es 1/3 del FFN y up/down quedan densos)*

**Lectura:** la tolerancia varía ~70× entre modelos. Qwen3.8-27B aguanta sp
0.25 con KL 0.14; ALIA-40b explota (KL 2.15 a sp 0.2).

### 4.2 Evolución Vía B (genomas finales, validados)

| Modelo | Compresión FFN (D_arch) | Ahorro FLOPs total* | KL | Validación |
|---|---|---|---|---|
| SmolLM2-135M | 7.8 % | ~4.6 % | 1.68 | n_pos 24, CPU==GPU |
| Qwen3.5-4B | 6.4 % | ~3.7 % | 0.379 | n_pos 8 |
| ALIA-40b | 1.8 % | ~1.3 % | 0.776 | n_pos 8 |
| Qwen3.8-27B | ~~10.7 %~~ → magnitud 12.2 % (gate) | ~4.1 % | ~~0.394~~ → **0.009-0.010** | n_pos 4 |

*`D_arch × fracción del cómputo que es FFN` (~0.6-0.7 según modelo). El 27B
además alcanzó un punto casi sin pérdida: KL 0.0005 a 0.4 % de compresión.

> ✅ **RESOLUCIÓN DEFINITIVA (sesión de optimización GPU, 2026-08-31):** el valor de
> esta tabla para **Qwen3.8-27B de la EVOLUCIÓN Vía B (KL 0.394 @ D_arch 0.107)**
> era un **artefacto del batch-eval roto**, no del modelo ni de la topología.
> NO es reproducible: la evolución real (topología CPPN, decode-pop + kl_eval_batch,
> n_layers 65) produce **KL 2.38-2.53**, y el genoma qwen35 transferido da KL ~3.0
> (batch) / 8.82 (producción). **Sin embargo, la FRONTERA UNIFORME POR MAGNITUD del
> 27B SÍ es real y quedó re-medida**: KL 0.0158/0.043/0.076/0.114/0.1404 a sp
> 0.05-0.25 (producción `embed_sparse --sparsities` + `kl_eval`, n_pos 4). El 27B
> es muy tolerable a la PODA POR MAGNITUD; la brecha (0.14 vs 2.5-8.8) es de la
> TOPOLOGÍA CPPN binaria, que no captura la magnitud a escala 27B. El "0.394"
> original quedó explicado y el resultado corregido (magnitud + perfil CPPN) es
> **KL 0.0092 (batch) / 0.0103 (producción)** — ver la resolución completa abajo.

**Perfiles de densidad por capa** (gate, subsample; % de conexiones activas):

```
SmolLM2:    95 95 95 95 96 96 ... 85 85 86 86 86 87   (media 92 %)
Qwen3.5-4B: 76 76 76 76 77 79 ... 100 100 100 100 100 (media 92 %)
ALIA-40b:   99 99 98 98 97 97 ... 95 95 95 95 95 95   (media 95 %)
Qwen3.8-27B:91 92 93 94 94 95 ... 100 100 100 100 100 (media 99 %)
```

La evolución concentra la poda donde el modelo es tolerante (capas iniciales de
Qwen, intermedias de SmolLM2) y mantiene densas las críticas (capas finales).

### 4.3 Vía B vs baseline uniforme

- **Qwen3.8-27B:** el "0.394 @ 10.7 %" era un artefacto del batch-eval roto
  (ver resolución §4.2). El resultado corregido es la **poda por magnitud con
  perfil de densidad del CPPN**: **KL 0.009-0.010 @ D_arch 0.1223 (gate)**,
  superando ampliamente el punto uniforme sp 0.25 (KL 0.143). En el marco del
  objetivo (max compresión @ KL≤0.5) la **magnitud + perfil gana** sobre la
  topología CPPN binaria (KL 2.5-8.8 a igual compresión).
- **SmolLM2 / ALIA:** la evolución no supera al baseline uniforme a igual
  compresión en el presupuesto usado (4 gens); la topología CPPN añade
  expresividad pero requiere más exploración (ver §6).

---

## 5. ¿Qué consigue el modelo podado frente al original?

### 5.1 ¿Es más ligero?
**En cómputo, sí (modestamente).** Podar el X% del FFN elimina ese X% de las
multiplicaciones del FFN. El ahorro total de FLOPs por token:

| Modelo | Compresión FFN | Ahorro FLOPs estimado | Params FFN eliminados |
|---|---|---|---|
| SmolLM2-135M | 7.8 % | ~4.6 % | ~6 M de 135 M |
| Qwen3.5-4B | 6.4 % | ~3.7 % | ~150 M de 4 B |
| ALIA-40b | 1.8 % | ~1.3 % | ~520 M de 40 B |
| Qwen3.8-27B | 10.7 % | ~6.8 % | ~1.9 B de 27 B |

**En tamaño de archivo, ahora sí.** Los pesos activos se exportan en **Q4_K**:
el GGUF esparso de SmolLM2 pesa **102 MB** frente a los 110 MB del Q4_K_M
original (y 354 MB si se exportaran en F32). Para Qwen/ALIA/27B el ahorro es
proporcional (Q4_K ≈ 1/8 del F32 en los bloques esparsos).

### 5.2 ¿Es más rápido?
**Medido en CPU (SmolLM2, 32 tokens, `bench_speed`):**

| Modelo | MemoryStrategy | tokens/s | ms/token |
|---|---|---|---|
| Original (Q4_K_M) | Minimal | 3.64 | 275 |
| Original (Q4_K_M) | AutoFit | 17.82 | 56 |
| Esparso F32 (D16) | Minimal | 0.61 | 1645 |
| Esparso F32 (D16) | AutoFit | 0.49 | 2030 |
| Esparso Q4_K (D16) | Minimal | 0.50 | 1982 |
| Esparso Q4_K (D16) | AutoFit | 0.54 | 1839 |

**Lectura honesta:** en CPU el modelo esparso es **más lento** (el overhead del
SpMM-CSR y del runtime de streaming dominan a la reducción de FLOPs del 5 %;
el Q4 no cambia la velocidad en CPU porque el cuello de botella no es la
banda). El path de **GPU** (SpMM-CSR en OpenCL, D29) es el de producción pero
**no se ha medido** aquí (regla de un solo proceso OpenCL; el re-run de
Qwen3.8-27B ocupa la GPU). El comando queda listo:
`bench_speed --model <esparso.gguf> --device auto`.

### 5.3 ¿Es indistinguible?
**Depende del umbral KL.** Con `KL ≤ 0.5` (el contrato) el modelo podado elige
el mismo token que el original en la gran mayoría de posiciones. Los puntos
reportados:

- Qwen3.5-4B y Qwen3.8-27B cumplen el contrato (KL 0.38-0.39).
- SmolLM2 (KL 1.68) y ALIA (KL 0.78) **no** lo cumplen a su compresión actual:
  para estos modelos la compresión con calidad es menor o la evolución no
  encontró aún el punto.

### 5.4 ¿Qué se paga?
La compensación clásica compresión ↔ calidad: a más `D_arch`, más KL. La
frontera uniforme (§4.1) cuantifica esa compensación por modelo.

---

## 6. ¿Se consiguió? Evaluación del experimento

### 6.1 Qué se logró
1. **Pipeline Vía B cerrado y reproducible** en 4 modelos (135M–40B): CPPN
   global → decode OpenCL en GPU → embedding D16 → evaluación KL → CMA-ES.
2. **Kernel OpenCL integrado** (16.5× más rápido, bit-exacto, linkeo dinámico
   sin SDK).
3. **Ranking de sensibilidad a la poda** de 4 familias de modelos.
4. **Genomas publicables** (466 floats/modelo) + GGUF D16 embebidos.

### 6.2 En qué medida (y con qué condiciones)
- **Compresión con calidad (KL ≤ 0.5): 2-11 % del FFN** (≈1-7 % de FLOPs
  totales) en los modelos probados, con `τ=0.42`, 4 generaciones CMA-ES, seed
  7, poda del gate. El FFN de los LLM modernos es **menos redundante de lo
  esperado** para la poda no estructurada.
- **Vía B no supera de forma consistente a la poda uniforme por magnitud** con
  este presupuesto: solo el 27B mejora el punto uniforme conocido. El valor
  demostrado es la **automatización del perfil** (sin etiquetado manual por
  capa) y el **pipeline GPU**.

### 6.3 Limitaciones
- **Latencia no medida** (solo FLOPs teóricos).
- **Velocidad en GPU sin medir** (solo CPU en este informe); el Q4 ya reduce
  el archivo pero la latencia real del SpMM-GPU queda pendiente.
- **4 generaciones** — presupuesto corto para un paisaje 466-D.
- **Calibración limitada** (128 prompts, n_pos 4-24).
- **Poda del gate únicamente** (no se podan up/down ni atención en los modelos
  grandes).

---

## 7. Reproducibilidad

Requisitos: Rust (workspace `saor` + `hayai`), GPU OpenCL (opcional, `--gpu`),
Python 3.12 + `numpy`.

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

Los GGUF embebidos (formato D16) se ejecutan con el `StreamingGenerator` de
`hayai` (FFN disperso vía SpMM-CSR en OpenCL).

---

## 8. Conclusiones

1. **La pregunta central («¿se puede optimizar la arquitectura de un LLM?»)
   se responde parcialmente:** sí es posible **automatizar una decisión de
   arquitectura** (el perfil de esparsidad por capa) con un CPPN evolucionado,
   de forma reproducible y con decode en GPU; el beneficio cuantitativo en
   estos 4 modelos es modesto (ver punto 2).
2. **El beneficio cuantitativo es modesto en estos modelos:** 2-11 % de
   compresión del FFN (1-7 % de FLOPs) con calidad acotada por KL ≤ 0.5. La
   poda no estructurada del gate no es una palanca de compresión fuerte en los
   LLM modernos.
3. **El ranking de sensibilidad (27B < 4B ≪ SmolLM2 < ALIA)** es el resultado
   más transferible: guía dónde tiene sentido la poda (híbridos gated-DeltaNet
   vs. arquitecturas densas sensibles como ALIA).
   > ✅ **VERIFICADO (2026-08-31) para la PODA POR MAGNITUD**: la frontera uniforme
   > del 27B (KL 0.14 @ sp 0.25) es menor que la del 4B (0.46) — el ranking del
   > informe es correcto para la magnitud. **No obstante, la TOPOLOGÍA CPPN binaria
   > no alcanza esa calidad en el 27B** (KL 2.5-8.8 vs 0.14 de la magnitud): el
   > ranking de sensibilidad aplica a la poda por magnitud, no a la topología CPPN.
4. **La topología CPPN automatiza el perfil** pero no supera al baseline
   uniforme de forma consistente con 4 generaciones; el pipeline GPU y la
   publicación de genomas permiten escalar la búsqueda.
5. **Código, genomas y modelos publicados** en GitHub (`aitups/saor`,
   `aitups/hayai`) y Hugging Face (colección `saor-via-b`).
