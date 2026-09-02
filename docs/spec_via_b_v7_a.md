# Especificación Vía B-v7, Interpretación A — Modelo compuesto training-free por CPPN (2026-09-01)

**Estado:** propuesta técnica de ingeniería para confirmación del equipo de diseño.
**Base:** directiva v7 (búsqueda no dirigida, sustrato 6-D global, KL ≤0.15, training-free).

## 1. Contrato científico (confirmado con diseño)

- El profesor (Qwen3.8-27B / ALIA-40B) es **solo el target de comportamiento**: se
  cachean sus logits sobre el corpus de calibración. No hay conversión de pesos
  entre modelos.
- El entregable es un **modelo nuevo** (grafo de nodos reales en el espacio 6-D
  global), **no** una poda/re-esparsificación del original.
- **Training-free**: los pesos del modelo nuevo son una **función determinista del
  genoma + la geometría** (CPPN como hypernetwork). CMA-ES optimiza el genoma;
  el fitness es la **KL global** frente al profesor. No hay gradiente ni
  destilación.
- El sustrato es **6-D global** `f(x1,y1,z1,x2,y2,z2) → [w, link]` como punto de
  partida; si la expresividad no alcanza, se amplía la entrada (p. ej. 9-D con
  derivadas `dx,dy,dz`, `sin/cos`).

## 2. Contrato de ejecución (runtime)

### I/O fija (frontera del modelo nuevo)
- **Tokenizer + vocabulario** del profesor (la KL es sobre logits del vocabulario).
- **Embedding y lm_head del profesor, congelados**: son la interfaz de entrada y
  salida (no son "capas a evolucionar"). El cómputo evolucionable vive entre ellos.
- El modelo nuevo procesa los tokens → `h = Embed(x)` → **grafo nuevo** → `logits
  = lm_head(h_final)`.

### El nodo real y el grafo nuevo
- Un nodo es una **unidad de cómputo vectorial** en la posición global
  `(x,y,z)` del modelo nuevo (ancho `d` libre; profundidad `z ∈ [-1,1]`).
- Cada nodo recibe contribuciones de los nodos-origen que la CPPN enlaza
  (`z_origen < z_destino`, salto libre), y emite su activación al flujo.
- La **CPPN genera el peso de cada conexión** desde la geometría:
  `w(conexión) = salida_peso de la CPPN`, evaluada sobre las coordenadas de los
  nodos origen/destino. El modelo resultante es **composicional/suave por
  construcción** — su capacidad de equivalencia global es lo que el experimento
  mide.
- El `link` de la CPPN decide qué conexiones existen (`l > τ`); el `weight` su
  ganancia/transformación.

### Bucle evolutivo
1. CMA-ES sobre el genoma CPPN (dim. ~500).
2. Por candidato: decodificar el grafo nuevo (nodos + conexiones + pesos) →
   forward sobre el corpus de calibración → logits.
3. Fitness = `KL(modelo_nuevo || profesor)` sobre el corpus cacheado.
4. Criterio: KL ≤0.15 (meta ≤0.05); el tamaño es variable (resultado, no
   restricción).

## 3. Hitos de implementación (saor + hayai)

| Hito | Contenido | Motor |
|---|---|---|
| M1 | Referencia Python del hypernetwork: `W[j,i] = f_CPPN(x_j,y_j,x_i,y_i)` por bloque; forward numpy; teacher cacheado; CMA-ES; KL. | saor (Python) |
| M2 | Validar M1 contra un **profesor sintético suave** (la CPPN debe poder expresarlo) → confirma el decode. | saor (Python) |
| M3 | Validar M1 contra un profesor real pequeño (Smol/4B): ¿la KL baja con las generaciones? ¿hasta dónde? | saor (Python) + hayai (cache) |
| M4 | Sustrato 6-D completo (nodos con z libre, saltos entre nodos) en la referencia Python. | saor |
| M5 | Portar el ejecutor del grafo nuevo a Rust/OpenCL (hayai ExecPlan de grafo) cuando M3/M4 lo justifiquen. | hayai |

## 4. Riesgos abiertos (a medir, no a asumir)

- **Expresividad del sustrato:** un modelo "suave por construcción" puede no captar
  la estructura no-suave del profesor con pocos nodos. Se mide en M3 (la KL baja
  o se estanca) y se amplía la entrada si hace falta (punto acordado).
- **Capacidad:** el genoma (~500 floats) compone el modelo; si la equivalencia
  exige más parámetros libres de los que un genoma fijo puede expresar, el
  sustrato/genoma se amplía (seguir el criterio empírico).
- **Coste del fitness:** cada forward del modelo nuevo sobre el corpus es el
  coste dominante; el diseño de M1 debe mantener `d` y el corpus pequeños para
  iterar rápido.
