Diseño de un experimento de **optimización no dirigida** para un modelo mediano (del orden de **25B a 40B de parámetros**) bajo severas restricciones de hardware (**RTX 4050 de ~6 GB de VRAM** y 15 GB de RAM sobre Windows). Aplicando los éxitos cosechados en la época de *Compose Universal*, las hipótesis abiertas de la *vía ablation* y la ingeniería de streaming de bajo nivel inspirada en `aitups/hayai`, podemos estructurar un experimento robusto en **7 pasos lógicos**:

---

### Paso 1: Configuración de la Base Firme y Auditoría (Alineación con Compose)
Para evitar que el modelo candidato sea un "cuerpo aleatorio" funcionalmente inerte (el gran error que condenó a `gate_v1` con un `proxy_nll` de ~14), primero establecemos una arquitectura base firme:
1. **Auditoría del Perfil de Parámetros:** Analizamos las cabeceras de los tensores del modelo base de 30B (sin cargarlos en memoria) para mapear sus cuellos de botella e ineficiencias internas.
2. **Definición de Bloques Candidatos:** Seleccionamos capas o módulos específicos de FFN o proyección para convertirlos en topologías no dirigidas, manteniendo el resto del esqueleto del Transformer intacto.
3. **Inicialización por Catálogo de Roles (`ROLE_CATALOG`):** El candidato copia de forma estricta los pesos iniciales del modelo base utilizando regex de profundidad. Esto asegura que el punto de partida de la evolución comience con una pérdida proxy saludable (`proxy_nll` de ~3–4).

### Paso 2: Precómputo Off-line de Hooks de Activación (Prioridad C - Sinkhorn)
Para cumplir con la prioridad de investigación "C" recomendada por Karuo (usar hooks de activación en lugar de escribir sobre pesos empaquetados en 4 bits), aislamos el comportamiento del modelo de la siguiente forma:
1. **Pase Único de Calibración:** Cargamos el modelo original de 30B secuencialmente, pasamos un lote muy pequeño de calibración (ej. \\(B = 128\\) tokens de alta diversidad) y guardamos las activaciones de entrada (\\(X\\)) y salida (\\(H_0\\)) de las capas que vamos a optimizar.
2. **Offload a RAM del Sistema:** Guardamos estas representaciones funcionales (\\(H_0\\)) en la **RAM de sistema (como pinned memory)** y descargamos el modelo original de 30B de la GPU. La VRAM de 6 GB ahora queda 100% disponible para el candidato.

### Paso 3: Motor de Streaming de Capas (Inspirado en hayai)
Para que el candidato de 30B se pueda procesar en la RTX 4050, el motor de inferencia debe gestionar de forma asíncrona la E/S de los pesos:
1. **Formato Cuantizado IQ4/Q4_K:** El modelo candidato se almacena cuantizado a 4 bits.
2. **Double Buffering por PCIe:** Utilizando kernels de bajo nivel de **OpenCL 3.0**, un hilo secundario en la CPU carga por streaming los pesos de la capa \\(l+1\\) desde la RAM mientras la GPU ejecuta el cómputo evolutivo de la capa \\(l\\). Esto mantiene un uso de VRAM plano y constante (~2 GB) durante toda la ejecución.

### Paso 4: Genoma Indirecto Desacoplado de la Escala (CPPN)
Para que el algoritmo evolutivo pueda optimizar un modelo de 30B sin que el genoma crezca exponencialmente:
1. **Red de Patrones de Composición (CPPN):** El genoma de la evolución codifica una pequeña CPPN de tamaño constante (ej. ~32K parámetros).
2. **Blueprints de Topología Irregular:** La CPPN recibe las posiciones multidimensionales del bloque del Transformer de 30B y genera un Grafo Acíclico Dirigido (DAG) de conexiones internas. El genoma es pequeño y manejable en memoria, pero es capaz de instanciarse y poblar un bloque de parámetros masivos de forma compatible con las dimensiones del Transformer estándar.

### Paso 5: Fitness CKA (Centered Kernel Alignment) por Bloque Aislado
En lugar de ejecutar inferencias completas del modelo gigante, evaluamos la equivalencia de la topología localmente en SRAM:
1. **Evaluación Local Directa:** Alimentamos el bloque candidato mutado con las activaciones \\(X\\) que precargamos en el Paso 2. 
2. **Reducción de Dimensionalidad CKA:** Calculamos localmente en SRAM la matriz de Gram del candidato (\\(K_1 = H_1 H_1^T \in \mathbb{R}^{B \times B}\\)) y medimos la similitud contra el hook guardado del profesor (\\(K_0\\)) utilizando el criterio HSIC.
3. **Filtrado Determinista:** Si la huella dactilar semántica del candidato es baja (ej. \\(\text{CKA} < 0.90\\)), la estructura se descarta inmediatamente por inviabilidad funcional, evitando que "redes muertas" consuman ciclos de CPU/GPU.

### Paso 6: CMA-ES en Subespacio Activo con Stateless Seed Replay (Prioridad 2)
Para cumplir con la recomendación de Karuo de "reabrir 2 CMA en subespacio bajo-dim" y respetar el límite de memoria física:
1. **Proyección en Subespacio de Curvatura (Fisher/Hessiano):** El algoritmo CMA-ES no opera sobre las millones de dimensiones del bloque, sino sobre un subespacio de curvatura activa de baja dimensión (ej. \\(d \approx 100-500\\)) que captura las pocas direcciones de cambio que alteran la función del LLM.
2. **Reconstrucción sin Estado (QES):** Para no saturar los 16 GB de RAM guardando tensores de perturbaciones, el optimizador evolutivo solo almacena la **semilla aleatoria entera** y la recompensa escalar de cada paso de la población. Los kernels de OpenCL 3.0 regeneran las perturbaciones al vuelo de forma determinista durante la fase de actualización de la matriz de covarianza, manteniendo un consumo de memoria insignificante.

### Paso 7: Criterio de Cierre Riguroso (Contrato de Fase 2)
Una vez que el optimizador evolutivo encuentra candidatos prometedores locales, el modelo final se densifica de forma uniforme a formato GGUF y debe someterse a la batería de validación real (no proxies ZCP) para cerrar el experimento:
1. **Divergencia KL Controlada:** El modelo final completo debe mantener la pérdida de validación en niveles estables frente al original (\\(\text{KL} \le 0.05\\)).
2. **Holdout y Confirmación de ARC:** El candidato debe evaluarse en un holdout ciego de tareas reales (ej. ARC-Challenge o GSM8K).
3. **Veredicto:** El experimento solo se considera exitoso si cumple simultáneamente las 3 reglas del contrato: distancia arquitectónica significativa (\\(\text{dist} \ge 0.4\\)), rendimiento superior al baseline (\\(\text{ARC}_{\text{search}} > \text{ARC}_{\text{base}}\\)) y comportamiento responsivo (no dormant).

---