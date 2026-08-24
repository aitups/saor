# Diseño de Arquitectura y Sistema: Optimización No Dirigida de LLMs

## 1. Visión General del Experimento
Este documento detalla la arquitectura de software e infraestructura para ejecutar un experimento de optimización no dirigida sobre modelos de lenguaje grandes (25B a 40B de parámetros). El diseño está constreñido a hardware de consumo restrictivo: una GPU RTX 4050 (~6 GB de VRAM) y 15 GB de RAM del sistema sobre Windows. 

## 2. Pila Tecnológica (Núcleo Híbrido)
* **Rust (Motor de Streaming y Memoria):** Gestiona la memoria asíncrona, el hilo principal de CPU, la memoria *pinned* y orquesta la carga de tensores hacia la GPU. Proporciona seguridad frente a fugas en tiempos de ejecución largos (semanas).
* **OpenCL 3.0 (Cómputo en VRAM):** Vía la caja `opencl3` de Rust. Ejecuta los kernels matemáticos directamente sobre la RTX 4050.
* **Python (Orquestación):** Vía `PyO3`. Coordina el flujo global.

## 3. Arquitectura de Aplicación (Data-Oriented Hexagonal)
Se emplea un modelo orientado a datos que aísla el I/O del cómputo puro:
* **Infraestructura:** `RustStreamer`, `PinnedMemoryAllocator`, `OpenCL_Kernels`.
* **Dominio:** `CKA_Evaluator`, `SubspaceCMAES`, `DimensionalReconciler`.
* **Aplicación:** `CPPN_Decoder`, `TopologyInstantiator`.

## 4. Diseño del Flujo en 7 Pasos Lógicos

**Paso 1: Configuración de la Base Firme y Auditoría**
* Mapeo in-memory de tensores del modelo base de 30B y selección de bloques inicializados vía regex (`ROLE_CATALOG`). 

**Paso 2: Precómputo Off-line de Hooks de Activación**
* Se pasa un lote de calibración fijado en $B = 128$ tokens de alta entropía semántica (mezclando código, matemáticas y prosa literaria).
* Las activaciones (Entrada $X$ y Salida $H_0$) se guardan en la RAM del sistema (pinned memory).
* Cálculo estático de Varianza de Activación (diagonal de la matriz de Fisher) para identificar los canales más activos.

**Paso 3: Motor de Streaming de Capas (Rust + Double Buffering)**
* Procesamiento estándar aplicado para mantener el pico de VRAM en ~2 GB.

**Paso 4: Genoma Indirecto Desacoplado (CPPN)**
* Red de tamaño estático evaluando coordenadas espaciales, generando un DAG libre de convenciones.

**Paso 5: Fitness CKA (Centered Kernel Alignment)**
* Alimentación del bloque candidato en SRAM y evaluación de alineación semántica usando el lote $B = 128$ para evitar sobreajuste y saturación de memoria.
* Filtro estricto: $\text{CKA} < 0.90$ implica descarte inmediato.

**Paso 6: CMA-ES en Subespacio Activo con Reconstrucción Sin Estado**
* Regeneración dinámica al vuelo de mutaciones topológicas (incluyendo umbral $\tau$) mediante OpenCL y semillas enteras.

**Paso 7: Criterio de Cierre Riguroso (Fase 2)**
* Validación final ($\text{KL} \le 0.05$, ARC-Challenge, GSM8K, distancia $\ge 0.4$).

## 5. Reconciliación Dimensional en Topologías No Dirigidas
El motor en Rust aplica dos primitivas de reconciliación algorítmica al vuelo, insertando nodos-puente en el DAG:

1. **Submuestreo Inteligente por Índices Calientes ($d_A > d_B$):** En lugar de un truncamiento ingenuo, el kernel OpenCL selecciona en tiempo constante $\mathcal{O}(1)$ los $d_B$ canales con mayor varianza, precalculados estáticamente a partir del profesor ($H_0$). Esto conserva la entropía de la representación semántica.
2. **Proyección Lineal Adaptativa ($d_A < d_B$):** Instanciación de una matriz $W_{proj}$ que emula una proyección identidad para mantener intacta la varianza de las activaciones previas.
