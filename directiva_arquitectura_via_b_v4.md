# Directiva Técnica de Redirección v4: Búsqueda Macro-Topológica Global y Evaluación de KL en Streaming (Vía B-v4)

**Para:** Equipo de Ingeniería y Desarrollo (`saor` & `hayai`)  
**Estatus:** DIRECTIVA MANDATORIA DE DISEÑO Y PRODUCCIÓN (Reemplaza la v3 y consolida la Vía B-v4)  
**Objetivo Científico:** Romper definitivamente la inercia de la optimización compartimentada capa a capa. El objetivo de este experimento no es aplicar una poda determinista a un esqueleto rígido de capas, sino **utilizar la neuroevolución para descubrir de forma orgánica, libre y no dirigida el grafo computacional mínimo e inteligente (morfología óptima)** que conserve el conocimiento semántico de modelos reales a escala industrial (`ALIA-40b` y `Qwen3.8-27b`) reduciendo el consumo físico de VRAM y el camino crítico de activación en inferencia.

---

## 1. El Paradigma de la Búsqueda Macro-Topológica No Dirigida

La optimización tradicional de modelos se limita a "curar las cicatrices" de una poda estática mediante un reentrenamiento pesado. En la **Vía B-v4**, el algoritmo evolutivo (CMA-ES y el espacio de representación geométrico de la CPPN) actúa como un arquitecto de caja negra que rediseña la macro-estructura del modelo a nivel global.

Para que la topología emerja verdaderamente de manera no dirigida, se establecen las siguientes directrices de libertad absoluta en el diseño del genotipo:

```
                  [ MODELO ORIGINAL: 4,000 CAPAS (36B) ]
                                    │
               (Proyección al Sustrato Tridimensional Global)
                                    ▼
               [ ESPACIO CONTINUO DE 6 DIMENSIONES (CPPN) ]
                f_CPPN(x1, y1, z1, x2, y2, z2) ──► [w, link]
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
[ LAYER SKIPPING FÍSICO ]  [ CONEXIONES ASIMÉTRICAS ]  [ FUSIÓN NO LINEAL ]
(Bypass total en streaming  (Salto Layer 5 ──► 200)    (Proyección continua de
 y cómputo de la GPU)                                   capas redundantes)
         │                          │                          │
         └──────────────────────────┼──────────────────────────┘
                                    ▼
                  [ PLAN DE EJECUCIÓN DINÁMICO (GGUF) ]
                (Compilado nativamente por hayai-core ExecPlan)
                                    ▼
                  [ MODELO ÓPTIMO EMERGENTE: 200 CAPAS ]
                  (100% compatible y físicamente ligero)
```

### A. El Sustrato Continuo Global de 6 Dimensiones
Se abandona el modelado secuencial o el uso de sustratos bidimensionales aislados por capa. Todo el LLM se proyecta en un único espacio continuo tridimensional:
*   Las coordenadas $x, y \in [-1, 1]$ representan la geometría del canal y las proyecciones neuronales de entrada/salida.
*   La tercera coordenada $z \in [-1, 1]$ representa la **profundidad relativa global de todo el modelo**, desde los embeddings de entrada ($z = -1.0$) hasta la cabeza de proyección de vocabulario (`lm_head`) en ($z = 1.0$).
*   La CPPN se evalúa como una función de acoplamiento espacial de 6 dimensiones:
    $$f_{\text{CPPN}}(x_1, y_1, z_1, x_2, y_2, z_2) \to [weight, link]$$
    Esta función determina la existencia y el peso de una conexión sináptica entre cualquier nodo en la posición $(x_1, y_1, z_1)$ y el nodo en $(x_2, y_2, z_2)$, sin importar su separación o el bloque al que pertenezcan originalmente.

### B. Mutaciones Estructurales Libres Permitidas
El loop evolutivo offline (`saor-engine`), inspirado en el diseño de agrupamiento de topologías complejas de **TensorNEAT** y **EvoX** [15, 25, 51], tendrá la capacidad y libertad de realizar:
1.  **Omisión Física de Capas (Layer Skipping / Block Dropping):** Si la densidad de conexiones que transitan por una profundidad $z$ (que corresponde a una capa MLP, de Atención o un bloque SSM de Mamba) cae por debajo de un umbral crítico $\tau_{\text{skip}}$, esa capa se declara inactiva. 
    *   *Bypass de Cómputo:* El compilador `ExecPlan` de `hayai` reescribe el grafo de inferencia para saltarse físicamente el bloque, haciendo fluir las activaciones a través del canal residual libre de FLOPs.
    *   *Bypass de Memoria:* El cargador asíncrono en Rust (`RustStreamer`) **ignora el bloque y no lo carga en la VRAM**, reduciendo físicamente el tamaño del modelo en memoria durante la inferencia.
2.  **Conexiones Asimétricas y No Secuenciales:** La evolución puede aprender a puentear grandes porciones del modelo, enrutando, por ejemplo, las activaciones de salida de la Capa 5 directamente a la Capa 120 de forma paralela al flujo principal, o creando ramificaciones asincrónicas.
3.  **Fusión No Lineal Continua:** Para grupos de capas intermedias con una altísima similitud semántica (redundancia de profundidad), la CPPN puede actuar como una función de proyección compacta. Condensa el mapeo funcional de múltiples capas secuenciales densas en un número reducido de operadores dispersos equivalentes de menor costo dimensional, eliminando las cicatrices que degradan la inteligencia y evitando la necesidad de entrenar de nuevo.

---

## 2. Evaluación de KL Global en Streaming (Layer-Streaming Batched KL)

Para resolver el bloqueo de ingeniería entre la necesidad científica de medir el comportamiento del modelo completo (utilizando logits reales en lugar de proxies locales que ocultan la acumulación de errores de profundidad) y la restricción de hardware de la RTX 4050, se implementa el paradigma de **Evaluación de KL en Streaming Batcheado**.

Se descarta el costoso proceso de evaluar a cada candidato de forma secuencial de principio a fin (lo que requeriría minutos por generación). En su lugar, el motor de ejecución aplica la regla: **"Movemos los candidatos a través del modelo, no el modelo a través de los candidatos"**.

```
[ PASO SECUENCIAL TRADICIONAL (INVIABLE) ]
Para cada candidato (1..22):
  Cargar Modelo Completo (23 GB) ──► Computar Forward ──► Medir KL (Minutos por Gen)

[ BARRIDO DE CAPAS EN LOTE (MANDATORIO VÍA B-V4) ]
Cargar Capa 1 en VRAM
  └── GPU procesa Capa 1 para los 22 candidatos en paralelo [VRAM: 46 MB]
Liberar Capa 1 ──► Cargar Capa 2
  └── GPU procesa Capa 2 para los 22 candidatos en paralelo...
Al llegar a lm_head (Capa final):
  └── GPU calcula Divergencia KL de los 22 candidatos a la vez frente al Profesor
```

### A. El Análisis Físico de Memoria en GPU
Mantener los estados de activación de toda la población en la VRAM es extremadamente barato y eficiente:
*   Para un lote de calibración de alta entropía de $B = 128$ tokens y una dimensión oculta de modelo de $d = 8192$ (como en `ALIA-40b`), la matriz de activación en FP16 para un único candidato en una capa dada ocupa:
    $$\text{Tamaño}_{\text{act}} = 128 \times 8192 \times 2 \text{ bytes} \approx 2.09 \text{ MB}$$
*   Para una población completa de $N = 22$ candidatos, el búfer de activación acumulado de la población entera ocupa:
    $$\text{VRAM}_{\text{población}} = 22 \times 2.09 \text{ MB} \approx 46 \text{ MB}$$
*   Este volumen de memoria es insignificante para los 6 GB de la RTX 4050, liberando casi toda la VRAM para cargar secuencialmente los pesos de la capa activa bajo streaming dinámico.

### B. El Algoritmo de Barrido por Capas (Layer-by-Layer Streaming)
Durante el bucle evolutivo, el orquestador asíncrono de Rust (`RustStreamer`) ejecuta un único pase de streaming de capas por generación, manteniendo a la población agrupada en memoria de la GPU:
1.  **Carga de Capa:** Se carga la Capa $L$ del modelo original en la GPU (pesando aproximadamente entre 300 y 500 MB).
2.  **Cómputo Batcheado:** El kernel de multiplicación de matrices dispersas toma el lote continuo gigante de activación de entrada de tamaño $(B \cdot N) \times d_{\text{in}}$ (los 22 candidatos concatenados) y proyecta las activaciones de la Capa $L$ de una sola vez.
3.  **Inyección de Topologías Variadas:** Al pasar por los bloques modificados por la CPPN, el kernel de OpenCL `cppn_decode_batched` aplica simultáneamente las 22 diferentes topologías e intensidades de peso decodificadas de la población sobre sus respectivas secciones del tensor batcheado.
4.  **Liberación de Memoria:** Se liberan los tensores de la Capa $L$ de la VRAM y se transfiere la Capa $L+1$ desde la memoria RAM del sistema a través de la interfaz PCIe Gen 4.
5.  **Cálculo de KL Directo:** En la última capa del modelo (`lm_head`), la GPU recibe los 22 tensores de logits resultantes, calcula la divergencia KL de la población completa de forma simultánea contra los logits del modelo profesor en microsegundos, y devuelve al host un array de $N$ flotantes con el fitness real del modelo completo. CMA-ES actualiza la matriz de covarianza de la población de inmediato.

*   **Rendimiento Físico Esperado:** El streaming completo del archivo GGUF toma ~3.8 segundos a velocidades estándar de PCIe Gen 4 x4, sumado a un coste de procesamiento batcheado despreciable de ~0.4 segundos. **El tiempo total de generación se consolida en ~4.2 segundos**, lo que permite ejecutar evoluciones de 100 generaciones en menos de 7 minutos utilizando la métrica de KL real de modelo completo como fitness directo de búsqueda.

---

## 3. Compatibilidad con Modelos Híbridos de Nueva Publicación

El motor evolutivo y el compilador `ExecPlan` deben ser 100% agnósticos y compatibles con las model cards de los tres nuevos modelos publicados por el equipo, garantizando el soporte nativo para:

*   **Arquitecturas Híbridas (Qwen3.8-27b):** Soporte explícito en el despachador de activaciones para bloques que fusionan atención de rango completo (`FullAttn`), transformadores recurrentes lineales (`DeltaNet`) y capas FFN con compresión SwiGLU (*gate*, *up* y *down*).
*   **Bloques de Espacio de Estados (Mamba SSM):** Las proyecciones estáticas masivas `in_proj` y `out_proj` se integran como nodos en el Grafo de Adyacencia Global de la CPPN para permitir su esparsificación estructural, mientras que los coeficientes del núcleo de escaneo temporal selectivo (S6) se conservan intactos en su formato de peso original para proteger el sesgo inductivo secuencial.

---

## 4. Requisitos de Implementación para los Equipos de Desarrollo

Para consolidar la Vía B-v4, se ordena suspender los parches deterministas capa a capa y programar los siguientes cambios en los repositorios de `saor` y `hayai`:

### Módulo `saor-engine` (Python / Rust):
1.  **Rediseñar el sustrato de coordenadas continuous:** Migrar de `coordinates.bin` bidimensionales por capa a un vector de geometría global continua de 6 dimensiones $(x_1, y_1, z_1, x_2, y_2, z_2)$ para representar conexiones macro-estructurales de extremo a extremo del LLM.
2.  **Integrar el evaluador de streaming de capas batcheado:** Implementar el tensor de activación de entrada combinado de la población $(B \cdot N) \times d_{\text{in}}$ y la secuencia de despacho coordinada con `RustStreamer` para extraer la KL global de la última capa en un solo pase por generación.
3.  **Exportación y Recuantización Q4 Estándar:** Modificar la Fase 7 de empaquetado para escribir los pesos optimizados supervivientes de las subredes y DAGs resultantes directamente bajo el formato estándar de cuantización `GGML_TYPE_Q4_K_M`, garantizando que el archivo GGUF resultante sea físicamente de un **30% a 50% más pequeño** y compatible con los runtimes del estado del arte sin necesidad de software exclusivo.

### Módulo `hayai-core` (Rust / OpenCL):
1.  **Compilador de Grafo Dinámico (`ExecPlan`):** Actualizar el planificador para que pueda interpretar archivos GGUF dispersos que contengan esquemas de enrutamiento asimétricos, conexiones no secuenciales de salto (capas puente) y operadores de omisión física de bloques sin causar pánicos de ejecución.
2.  **Kernels OpenCL de Ejecución de Grafo:** Asegurar que los kernels de procesamiento en GPU toleren y despachen de forma eficiente las llamadas asimétricas del plan de inferencia irregular utilizando la SRAM local para optimizar las activaciones.

---

## 5. Criterios de Aceptación Científica de la Vía B-v4

El éxito del proyecto y la entrega del modelo final consolidado se certificarán únicamente si se cumplen simultáneamente los siguientes cuatro requisitos en la máquina de pruebas de Karuo:

| Criterio de Control | Umbral de Éxito | Método de Medición |
| :--- | :--- | :--- |
| **Conservación de Inteligencia** | $KL_{\text{global}} \\le 0.50$ (o caída de precisión $\le 1.0\%$ en ARC-Challenge) | Evaluación de inferencia de modelo completo en `hayai` tras el intercambio físico del bloque disperso. |
| **Optimización de Latencia** | Aceleración de velocidad de inferencia $\ge 25\\%$ (en tokens/s) | Ejecución nativa del modelo optimizado final versus el modelo denso original de referencia. |
| **Reducción de VRAM** | Ahorro físico en memoria activa de GPU $\ge 30\\%$ | Monitoreo directo de la asignación de buffers de inferencia estáticos y dinámicos en VRAM. |
| **Novedad y Parsimonia** | Emergencia probada de DAG asimétrico con Layer Skipping | Visualización y conteo de bloques descartados o puenteados de forma autónoma por la evolución. |
