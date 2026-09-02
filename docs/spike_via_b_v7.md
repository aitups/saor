# Spike Vía B-v7 — Baseline de factibilidad (2026-09-01)

Preparado por ingeniería tras la directiva v7 (búsqueda no dirigida en sustrato
6-D, KL ≤0.15). Objetivo del spike: determinar si el objetivo científico es
demostrable y en qué testbed, ANTES de implementar el sustrato 6-D completo.

## 1. Hallazgo principal: el smol NO sirve de testbed para KL ≤0.15

La frontera de **poda por magnitud** (el óptimo teórico para densidad fija con
pesos del profesor) del SmolLM2-135M:

| sp | n_pos 4 | n_pos 24 |
|----|---------|----------|
| 0.05 | — | KL 0.419 |
| 0.10 | KL 2.00 | KL 0.771 |
| 0.25 | KL 6.58 | KL 2.481 |

Ni la magnitud (cota superior) alcanza **KL ≤0.15** en el smol, ni siquiera a la
poda más ligera (sp 0.05). El denso verifica KL 0.0 (evaluador correcto). **El
smol es KL-hostil al sparsificado del FFN** (135M: cada conexión cuenta mucho).
Consecuencia: el smol no puede demostrar la v7 → el testbed debe ser el 4B.

## 2. Baseline por modelo (¿quién puede demostrar KL ≤0.15?)

| Modelo | Magnitud (óptimo) | CPPN libre 9-D (v1, topología) | ¿≤0.15 alcanzable? |
|---|---|---|---|
| SmolLM2-135M | 0.42 @ D_arch 0.017 | 1.68 @ D_arch 0.078 | ❌ ni el óptimo |
| **Qwen3.5-4B** | 0.045-0.108 @ D_arch 0.017-0.033 | **0.0805 @ D_arch 0.027** | ✅ ambos |
| Qwen3.8-27B | 0.009-0.14 @ D_arch 0.017-0.083 | 2.5-8.8 | ✅ solo magnitud |

**Lectura:** el 4B es el único testbed rápido que demuestra ≤0.15 **tanto con el
óptimo (magnitud) como con la búsqueda libre (CPPN 9-D)**. La brecha magnitud vs
CPPN libre en el 4B es ~1.5-2× de compresión a igual KL — es la brecha que la
v7 quiere cerrar con el sustrato 6-D.

## 3. Diseño del spike 6-D (siguiente paso)

- **Testbed:** Qwen3.5-4B (33 capas FFN [2560→9216]), n_pos 4, protocolo del kl_eval
  de producción (el mismo de los modelos publicados).
- **Genoma:** CPPN 6-D `f(x1,y1,z1,x2,y2,z2) → [weight, link]` (ref Python), dims
  de entrada 6 (+ derivadas), 2×16 ocultos, salida 2.
- **Pregunta A (per-capa, primero):** con el grafo FFN actual (z1=z2=z_capa),
  ¿la búsqueda libre 6-D (enlace `l>τ`, sin |w|, pesos del profesor en los activos)
  alcanza KL ≤0.15 a D_arch ≥ la CPPN 9-D (0.027)? Esto aísla el valor del sustrato
  6-D frente al 9-D.
- **Pregunta B (cross-capa, después):** requiere el `ExecPlan` de grafo (skipping
  de bloques + puentes asimétricos) — la libertad real de la v7.

## 4. Estados

- ✅ Tags: `saor v1.0.0-via-b`, `hayai v0.3.0` (frontera de la variante dirigida).
- ✅ Baseline del spike (este documento).
- ⏳ Pregunta A (6-D per-capa): implementación en curso.
- ⏳ Pregunta B (cross-capa): requiere `ExecPlan` de grafo en hayai.
