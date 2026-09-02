# Directiva Técnica de Hierro v7: Criterios Científicos Innegociables y Blindaje de la Búsqueda No Dirigida

**Para:** Equipo de Ingeniería y Desarrollo (saor & hayai)  
**Estatus:** ESPECIFICACIÓN TÉCNICA SUPREMA Y CONTRATO DE CALIDAD INNEGOCIABLE  
**Objetivo:** Establecer de forma definitiva las reglas de aceptación científica del proyecto. Esta directiva v7 anula cualquier especificación técnica anterior y prohíbe explícitamente todo atajo metodológico o heurística dirigida que desvirtúe el propósito científico de la investigación.

---

## 1. El Criterio de Aceptación Real: Equivalencia de Inteligencia y Topología Óptima

El objetivo último de este proyecto no es cumplir una cuota de compresión comercial uniforme, sino **descubrir el grafo computacional mínimo e inteligente de cada modelo mediante búsqueda evolutiva libre**. Por tanto, se redefinen los criterios de éxito estructural:

*   **El tamaño es un resultado, no una restricción rígida:** Se eliminan los objetivos mandatorios de reducción física de tamaño en disco o VRAM como condiciones de aprobación. La reducción de peso es un resultado altamente deseable, pero secundario frente al hallazgo de la topología óptima. 
*   **La heterogeneidad de los modelos:** Se asume que la redundancia de un LLM depende de su propia morfología. Si un modelo como `Qwen3.8-27b` posee una redundancia estructural masiva y la evolución lo reduce en un **40%**, el resultado es correcto. Si otro modelo como `ALIA-40b` posee una estructura ya sumamente optimizada en sus bloques críticos y la evolución solo reduce un **5%** para preservar el conocimiento, **ese resultado es igualmente válido y valioso**. Ambas opciones son aprobadas siempre que se demuestre que el algoritmo ha alcanzado el máximo grado de optimización topológica global.
*   **El foco científico:** El valor de la investigación reside en el descubrimiento de la topología óptima no convencional y de la **función de transformación no lineal de pesos** (CPPN) que codifica la complejidad geométrica del modelo original.

---

## 2. La Línea Roja: Búsqueda No Dirigida Libre (Cláusula de No Cumplimiento Inmediato)

No se tolerará que el equipo de desarrollo limite el espacio de búsqueda evolutivo para esquivar problemas de convergencia matemática o costes de cómputo. **La no-direccionalidad de la búsqueda se blinda por completo**:

*   **Prohibiciones Explícitas:**
    1.  **Sistemas deterministas o heurísticas por capas:** Se prohíbe introducir reglas de diseño del estilo "si la capa es de tipo A, se aplica la heurística B". Esto provoca cicatrices severas que degradan la capacidad del modelo y que después exigen costosos reentrenamientos.
    2.  **Topologías preasumidas:** El motor de inferencia y el generador no deben asumir estructuras de capas secuenciales o bipartitas rígidas para el modelo optimizado.
    3.  **Reducción dirigida del espacio de soluciones:** Se prohíbe restringir manualmente los enlaces, limitar de forma estática las conexiones a canales adyacentes o ignorar arbitrariamente componentes del modelo (como las proyecciones de vocabulario o las capas híbridas de Mamba/SSM) bajo el pretexto de que "sabemos qué hay en ellas".
*   **La Regla de Hierro del NO CUMPLIMIENTO:**
    Cualquier modificación o atajo técnico implementado por el equipo que altere la búsqueda no dirigida global en el sustrato de 6 dimensiones para convertirla en una optimización local, fragmentada o guiada por reglas manuales **será considerada un NO CUMPLIMIENTO inmediato**. La solución resultante será marcada como **incorrecta y rechazada** sin evaluar sus métricas de compresión o rendimiento.

---

## 3. Endurecimiento del Umbral de Inteligencia: Límite de Divergencia Global

Una divergencia KL de $0.50$ es inaceptable para estándares científicos y de producción. Un modelo con tal nivel de desalineamiento cognitivo produce texto fluido pero pierde por completo la precisión lógica y la coherencia semántica en tareas complejas de holdout.

*   **El nuevo umbral estricto:** Se establece un límite de divergencia global innegociable de **$KL \le 0.15$** (frente a los logits del modelo profesor) como puerta de validación final en la Fase 7.
*   **La meta de excelencia:** El objetivo ideal de diseño es alcanzar un **$KL \le 0.05$** (estándar heredado de la técnica *Compose*).
*   **Implicación para el optimizador:** Al endurecer el KL, se obliga a la evolución offline a ser extremadamente fina, forzando al optimizador CMA-ES a esculpir conexiones y fusiones analógicas perfectas en lugar de conformarse con podas destructivas de bajo nivel. Si la evolución no logra converger por debajo de $KL \le 0.15$, se aceptará el hallazgo científico de que el modelo ha alcanzado su límite físico absoluto de optimización.

---

## 4. Requisitos de Implementación Física y Muestreo del Sustrato 6-D

Para que el equipo de desarrollo pueda cumplir con las exigencias científicas sin comprometer el hardware, se definen los siguientes criterios de implementación matemática:

### A. Muestreo Discretizado On-Demand en GPU (Sustrato 6-D)
La CPPN de 6 dimensiones $f_{\text{CPPN}}(x_1, y_1, z_1, x_2, y_2, z_2) \to [weight, link]$ es completamente viable si se implementa bajo un esquema de indexación discreta on-demand:
1.  **Profundidades globales ($z_1, z_2$):** Se normalizan en el rango $[-1, 1]$ según el índice físico de la capa de origen y destino ($z = \text{capa\_index} / N_{\text{layers}} \cdot 2 - 1$).
2.  **Canales físicos ($x, y$):** Se discretizan y escalan linealmente al sustrato continuo para cada par de nodos activo durante la evaluación.
3.  **Complejidad:** Al evaluar únicamente el subespacio correspondiente a los tensores activos del modelo original, la GPU procesa la decodificación de forma masivamente paralela sin lidiar con complejidades infinitas, preservando el potencial de la evolución para generar puentes intercapa, bypasses no secuenciales y estructuras paralelas asimétricas.

### B. Rendimiento del Loop: Population Batching en $\le 6.0$ Segundos
La optimización física del motor no es negociable. El equipo debe cumplir con un tiempo de evaluación por generación de **$\le 6.0$ segundos** para una población de $N = 22$ candidatos utilizando el diseño de **Streaming de Capas en Lote (Layer-Streaming Batched KL)**:
*   Se unifican las activaciones de calibración en un tensor continuo de población de apenas ~46 MB de VRAM.
*   El archivo de modelo (GGUF original) se transmite a través del bus PCIe una única vez por generación, ejecutando el cómputo de la población de forma batcheada capa por capa.
*   Toda evaluación se realiza bajo un único contexto y una única cola asíncrona de OpenCL en Rust, eliminando el paralelismo por procesos concurrentes de Windows para evitar deadlocks en `opencl.dll`.

---

## 5. Tabla de Criterios de Aceptación Definitiva (Contrato Técnico)

| Métrica de Control | Límite de Aceptación (Fila de Hierro) | Consecuencia de Desviación |
| :--- | :--- | :--- |
| **Búsqueda Evolutiva** | **100% No Dirigida en Sustrato 6-D** | **Rechazo inmediato por NO CUMPLIMIENTO** (Cualquier heurística local invalida el entregable). |
| **Divergencia Semántica** | **$KL \le 0.15$** (Logits del modelo profesor) | Rechazo del modelo por degradación cognitiva. |
| **Rendimiento del Motor** | **$\le 6.0$ s / generación** (Población $N=22$) | Rechazo por ineficiencia de sincronización o cuello de botella en CPU. |
| **Paridad matemática** | **$\text{max\_err} = 0.0$** en `validate_opencl.py` | Suspensión de la evolución hasta corregir desfases de coordenadas. |
| **Optimización de Tamaño** | **Variable (0% a 90%)** | **No es un criterio de exclusión.** Se aprueba cualquier porcentaje siempre que la topología sea matemáticamente óptima. |

---

Esta directiva establece el estándar científico inmutable para el cierre del proyecto. El equipo de ingeniería debe enfocar sus esfuerzos en la optimización del flujo físico de datos y en la programación de los kernels batcheados, dejando que sea la inteligencia del algoritmo evolutivo la que determine, libre de sesgos humanos, la estructura matemática del LLM del futuro.
