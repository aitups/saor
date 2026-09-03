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
| **Qwen3.5-4B** | 0.045-0.108 @ D_arch 0.017-0.033 | **0.447 @ D_arch 0.064** (corregido) | ✅ solo magnitud |
| Qwen3.8-27B | 0.009-0.14 @ D_arch 0.017-0.083 | 2.5-8.8 | ✅ solo magnitud |

> **CORRECCIÓN (2026-09-01):** los valores históricos de la CPPN-topología del 4B
> (KL 0.379 y 0.0805 @ D_arch 0.027 del `via_b_history_qwen35`) se midieron con
> el **batch con el bug del kernel batcheado Q4_K** (hidden 2560 > 2048 → el bug
> del tile `t0>0` los afectaba). Re-medido con la **producción** (gemv simple,
> evaluador correcto): el genoma CPPN libre del 4B da **KL 0.447 @ D_arch 0.064**.
> La brecha magnitud vs CPPN-libre en el 4B es mayor de lo que se creía.
>
> Nota v7: estos valores son de la CPPN **por capas** de la v1 y **no aplican** al
> sustrato **9-D global** de la v7 (nunca medido). Solo son la referencia del
> punto de partida dirigido.

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

## 5. Hallazgo M3b (2026-09-01): la CPPN no puede warm-startar los pesos reales

La v7-b especifica que el genoma de la gen-0 se inicializa por **regresión** para
que la CPPN pinte una réplica casi exacta de los pesos del profesor (KL inicial
≈ 0). Medición sobre un slice real del gate de Qwen3.5-4B (w.0.ffn_gate,
`[512×512]`, regresión ELM de 256 ocultos — la mejor réplica para esa arquitectura):

| Métrica | Valor | Implicación |
|---|---|---|
| Correlación filas adyacentes (pesos reales) | **-0.001** | sin estructura local en (i,j) |
| Correlación columnas adyacentes | **0.013** | sin estructura local |
| CKA(CPPN-recon, W real) | **0.0002** | la CPPN no regresa los pesos |
| CKA(CPPN-recon, W suavizado) | 0.0037 | ni el suavizado es expresable |

**Conclusión:** los pesos FFN entrenados son ~aleatorios en el orden de canales
(sin estructura suave que una CPPN geométrica pueda capturar). El warm-start por
regresión de la v7-b **no alcanzará KL inicial ≈ 0**: la gen-0 partirá de un KL
alto (comportamiento ~aleatorio de los pesos aproximados). Esto coincide con el
hallazgo D15 previo.

**Consecuencia operativa:** o (a) diseño redefine el warm-start (p. ej. geometría
de canal aprendida/permutada, no el índice crudo), o (b) la evolución parte de un
genoma aleatorio (gen-0 sin warm-start) midiendo si CMA-ES descubre estructura
que reduzca la KL desde el caos — que es la prueba pura de la hipótesis no
dirigida de la v7, con el suelo alto como posible "hallazgo del límite".

### 5b. Medición del KL gen-0 del warm-up (2026-09-01, la prueba real)

La premisa del warm-up ("la regresión deja KL ≈ 0 desde el que evolucionar") se
midió de forma **comportamental** (no proxy): regresión ELM del genoma al gate
real del smol (la mejor réplica que una CPPN geométrica puede pintar) → inyección
densa D16 all-active (gen-0, `link=1`, pesos de la CPPN) → `kl_eval` (n_pos 8):

| Medición | Resultado |
|---|---|
| Fidelidad de la réplica (1 = perfecta) | **0.0007** |
| **KL gen-0 real (solo la gate 0 reemplazada)** | **7.94** |

**Conclusión:** reemplazar UNA sola gate (capa 0 de 30) con la aproximación del
warm-up colapsa el modelo (KL 7.94 ≈ modelo aleatorio). El warm-up con la
geometría de canal **cruda** (índice escalado) no deja KL ≈ 0: la CPPN no puede
expresar los pesos reales (no-suaves en el orden de canales). **La premisa del
warm-up requiere geometría de canal aprendida/permutada** (que las coordenadas se
ordenen para que W sea suave en ellas) o la gen-0 parte del caos.

### 5c. Medición al nivel del BLOQUE COMPLETO (corrección del proceso)

Corrigiendo el error de descomposición (medir una gate aislada): el warm-up se
ejecutó sobre el **FFN completo** (gate+up+down) de las **30 capas** del smol
(90 matrices aproximadas por la regresión CPPN e inyectadas en el D16 all-active,
`--all-blocks` añadido al embed):

| Medición | Resultado |
|---|---|
| Matrices del bloque completo aproximadas | 90 (30 capas × gate/up/down) |
| **KL gen-0 (modelo completo, FFN completo reescrito)** | **14.65** |

**Resultado al nivel completo:** el modelo queda ~aleatorio (KL 14.65). El warm-up
con la geometría cruda no deja KL ≈ 0 tampoco al nivel de bloque completo. La
corrección de rumbo NO es abandonar la v7, sino **añadir geometría de canal
aprendida** al sustrato (coordenadas por canal evolucionadas/regresadas, no el
índice crudo) para que la CPPN pueda expresar los pesos y el warm-up cumpla su
función.



## 6. Estados

- ✅ Tags: `saor v1.0.0-via-b`, `hayai v0.3.0` (frontera de la variante dirigida).
- ✅ Baseline del spike (este documento).
- ⏳ Pregunta A (6-D per-capa): implementación en curso.
- ⏳ Pregunta B (cross-capa): requiere `ExecPlan` de grafo en hayai.
