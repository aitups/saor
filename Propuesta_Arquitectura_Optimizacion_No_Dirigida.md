# Diseño de Arquitectura y Sistema: Optimización No Dirigida de LLMs

## 1. Visión General del Experimento
Este documento detalla la arquitectura de software e infraestructura para ejecutar un experimento de optimización no dirigida sobre modelos de lenguaje grandes (25B a 40B de parámetros). El diseño está constreñido a hardware de consumo restrictivo: una GPU RTX 4050 (~6 GB de VRAM) y 15 GB de RAM del sistema sobre Windows. 

## 2. Pila Tecnológica (Núcleo Híbrido)
Para garantizar el control determinista de la memoria sin sacrificar la madurez del ecosistema de ML, el sistema adopta una arquitectura de núcleo híbrido:

* **Rust (Motor de Streaming y Memoria):** Gestiona la memoria asíncrona, el hilo principal de CPU, la memoria *pinned* y orquesta la carga de tensores hacia la GPU. Proporciona seguridad frente a fugas en tiempos de ejecución largos (semanas).
* **OpenCL 3.0 (Cómputo en VRAM):** Vía la caja `opencl3` de Rust. Ejecuta los kernels matemáticos directamente sobre la RTX 4050, operando de forma determinista y altamente paralela.
* **Python (Orquestación):** Vía `PyO3`. Coordina el flujo global, instanciando la red CPPN, aplicando las expresiones regulares para inicialización de pesos y calculando las divergencias para los contratos de fase final.

## 3. Arquitectura de Aplicación (Data-Oriented Hexagonal)
En lugar de una arquitectura orientada a objetos (que fragmentaría la memoria), se emplea un modelo orientado a datos que aísla el I/O del cómputo puro:

### Capa de Infraestructura (Hardware)
* `RustStreamer`: Planificador asíncrono que precarga en VRAM la capa $l+1$ mientras la capa $l$ se computa.
* `PinnedMemoryAllocator`: Reserva contigua de RAM para mapeo directo vía PCIe sin pasar por memoria paginada de Windows.
* `OpenCL_Kernels`: Multiplicación de matrices dispersas (SpMM) y proyecciones.

### Capa de Dominio (Matemáticas)
* `CKA_Evaluator`: Calcula la matriz de Gram y la similitud HSIC localmente en SRAM.
* `SubspaceCMAES`: Reconstrucción sin estado basada en semillas enteras de las perturbaciones sobre la curvatura de Fisher/Hessiano.
* `DimensionalReconciler`: Aplica reglas matemáticas para conectar sub-grafos irregulares.

### Capa de Aplicación (Evolución y Grafo)
* `CPPN_Decoder`: Transforma el genoma de tamaño fijo (~32K) en el espacio euclidiano continuo.
* `TopologyInstantiator`: Ensambla el Grafo Acíclico Dirigido (DAG) validando la máscara causal algorítmica.

## 4. Diseño del Flujo en 7 Pasos Lógicos

**Paso 1: Configuración de la Base Firme y Auditoría**
* Mapeo in-memory de tensores del modelo base de 30B.
* Selección de bloques candidatos (FFN/Proyección) inicializados estrictamente copiando los pesos originales vía regex de profundidad, estableciendo un catálogo de roles (`ROLE_CATALOG`). 

**Paso 2: Precómputo Off-line de Hooks de Activación**
* Se pasa un lote de calibración de alta diversidad. 
* Las activaciones (Entrada $X$ y Salida $H_0$) se guardan en la RAM del sistema (pinned memory).
* El modelo original se descarga para liberar el 100% de la VRAM al candidato.

**Paso 3: Motor de Streaming de Capas (Rust + Double Buffering)**
* Modelo almacenado en formato cuantizado a 4 bits (IQ4/Q4_K).
* Un hilo secundario de Rust precarga tensores por PCIe mediante buffers en anillo, manteniendo el pico de VRAM constante en ~2 GB.

**Paso 4: Genoma Indirecto Desacoplado (CPPN)**
* En lugar de mutar billones de parámetros, el genoma es una CPPN (Red de Patrones de Composición) estática evaluando coordenadas espaciales $(x_i, y_i) 	o (x_j, y_j)$, incrustaciones posicionales trigonométricas y una señal de profundidad relativa $z$.
* Es completamente **libre de convenciones**: Puede ignorar estructuras clásicas de Transformers siempre que no viole el enrutamiento acíclico.

**Paso 5: Fitness CKA (Centered Kernel Alignment)**
* Alimentación del bloque candidato en SRAM y evaluación de alineación semántica.
* Si el candidato obtiene un $\text{CKA} < 0.90$ frente al hook precomputado ($H_0$), es descartado inmediatamente (prevención de redes muertas).

**Paso 6: CMA-ES en Subespacio Activo con Reconstrucción Sin Estado**
* El motor evolutivo solo guarda en memoria la semilla aleatoria y la recompensa de cada individuo de la población.
* Las mutaciones se regeneran dinámicamente al vuelo mediante OpenCL, reduciendo a cero la presión sobre la memoria RAM (Stateless Seed Replay).
* El algoritmo evoluciona no solo los pesos, sino un **umbral dinámico $\tau$** que rige el grado de dispersión (sparsity) de las conexiones generadas por la CPPN.

**Paso 7: Criterio de Cierre Riguroso (Fase 2)**
* El modelo se consolida en formato denso (GGUF).
* Validación final: Divergencia de Kullback-Leibler ($\text{KL} \le 0.05$) y pruebas holdout en tareas reales (ARC-Challenge, GSM8K).
* Requiere cumplir distancia arquitectónica $\ge 0.4$ y superación del rendimiento base.

## 5. Reconciliación Dimensional en Topologías No Dirigidas
Dado que la red CPPN puede conectar tensores arbitrarios de dimensiones dispares ($d_A$ y $d_B$), el motor en Rust aplica dos primitivas de reconciliación algorítmica al vuelo, insertando nodos-puente en el DAG sin penalizar VRAM:

1. **Submuestreo por Truncamiento Rápido ($d_A > d_B$):** Descarte funcional en tiempo constante $\mathcal{O}(1)$ vía kernels de indexación en OpenCL.
2. **Proyección Lineal Adaptativa ($d_A < d_B$):** Instanciación de una matriz $W_{proj}$ que emula una proyección identidad para mantener intacta la varianza de las activaciones previas, con pesos modulados implícitamente por el subespacio latente de la CPPN.
