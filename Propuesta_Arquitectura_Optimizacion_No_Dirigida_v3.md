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
* **Aplicación (Evolución y Grafo):** 
  * `CPPN_Decoder`: Transforma el genoma de tamaño fijo (~32K) en el espacio euclidiano continuo, procesando coordenadas espaciales, distancias relativas e incrustaciones trigonométricas para generar topologías complejas.
  * `TopologyInstantiator`: Ensambla el Grafo Acíclico Dirigido (DAG) validando la máscara causal algorítmica y aplicando el umbral de esparsidad dinámico.

## 4. Diseño del Flujo en 7 Pasos Lógicos

**Paso 1: Configuración de la Base Firme y Auditoría**
* Mapeo in-memory de tensores del modelo base de 30B y selección de bloques inicializados vía regex (`ROLE_CATALOG`). 

**Paso 2: Precómputo Off-line de Hooks de Activación**
* Se pasa un lote de calibración fijado en $B = 128$ tokens de alta entropía semántica (mezclando código, matemáticas y prosa literaria).
* Las activaciones (Entrada $X$ y Salida $H_0$) se guardan en la RAM del sistema (pinned memory).
* Cálculo estático de Varianza de Activación (diagonal de la matriz de Fisher) para identificar los canales más activos.

**Paso 3: Motor de Streaming de Capas (Rust + Double Buffering)**
* Procesamiento estándar aplicado para mantener el pico de VRAM en ~2 GB.

**Paso 4: Genoma Indirecto Desacoplado (Especificación Técnica CPPN)**
La red de patrones de composición (CPPN) se define mediante el siguiente formalismo matemático para interactuar con las capas del Transformer de 30B:

* **1. Sistema de Coordenadas del Sustrato (Transformer-to-2D Mapping):**
  Para optimizar un bloque de proyección lineal o FFN de entrada $d_{\text{in}}$ y salida $d_{\text{out}}$, definimos un sustrato bidimensional donde cada neurona recibe una coordenada fija en $[-1, 1]$:
  * **Capa de Entrada (A):** $x_i = -1.0, \quad y_i = -1.0 + \frac{2i - 2}{d_{\text{in}} - 1}$ para $i \in \{1, \dots, d_{\text{in}}\}$
  * **Capa de Salida (B):** $x_j = 1.0, \quad y_j = -1.0 + \frac{2j - 2}{d_{\text{out}} - 1}$ para $j \in \{1, \dots, d_{\text{out}}\}$
  * **Nodos Internos Irregulares (DAG):** Coordenadas dinámicas evaluadas en el rango $x_k \in (-1.0, 1.0)$.

* **2. Ecuación de la CPPN y Variables del Genoma:**
  La CPPN es una función continua $f_{\text{CPPN}}$ parametrizada por un genoma de $\approx 32\text{K}$ parámetros. Procesa un vector $\mathbf{v}_{\text{in}}$ de 8 dimensiones:
  $$\mathbf{v}_{\text{in}} = \left[ x_i, y_i, x_j, y_j, \Delta x, \Delta y, \sin(\pi y_i), \cos(\pi y_j) \right]$$
  Donde $\Delta x = x_j - x_i$ y $\Delta y = y_j - y_i$. Los términos trigonométricos actúan como incrustaciones posicionales fijas para inducir patrones modulares periódicos.

* **3. Salidas de la CPPN y Generación de la Topología:**
  $$f_{\text{CPPN}}(\mathbf{v}_{\text{in}}) \to \left[ w_{ij}, l_{ij} \right]$$
  * **Peso de Conexión ($w_{ij}$):** Valor real linearizado que define la magnitud del peso sináptico.
  * **Probabilidad de Enlace ($l_{ij} \in (0, 1)$):** Filtrado por Sigmoide, define la fuerza estructural de la conexión.

* **4. Aplicación de la Máscara de Esparsidad Dinámica ($\tau$):**
  El genoma evolutivo optimiza concurrentemente un umbral de esparsidad adaptativo $\tau \in (0, 1)$. La matriz de adyacencia $\mathbf{A}$ en SRAM se construye como:
  $$\mathbf{A}_{ij} = 1 \text{ si } l_{ij} > \tau \text{, de lo contrario } 0$$
  Si $\mathbf{A}_{ij} = 1$, el peso en el DAG es $W_{ij} = w_{ij}$; de lo contrario, se omite.

* **Impacto en el Motor (Rust/OpenCL 3.0):**
  * **Representación GGUF Agnóstico:** El archivo almacena solo el genoma CPPN (~32K pesos) y el umbral escalar $\tau$.
  * **Decodificación al vuelo en GPU (OpenCL):** Para evitar cuellos de botella de VRAM y RAM, el kernel OpenCL ejecuta $f_{\text{CPPN}}$ paralelamente en GPU, poblando la matriz dispersa (SpMM) de forma local en SRAM, enviando solo el vector de 32K por PCIe.

**Paso 5: Fitness CKA (Centered Kernel Alignment)**
* Alimentación del bloque candidato en SRAM y evaluación usando el lote $B = 128$.
* Filtro estricto: $\text{CKA} < 0.90$ implica descarte inmediato.

**Paso 6: CMA-ES en Subespacio Activo con Reconstrucción Sin Estado**
* Regeneración dinámica al vuelo de mutaciones topológicas (incluyendo umbral $\tau$) mediante OpenCL y semillas enteras.

**Paso 7: Criterio de Cierre Riguroso (Fase 2)**
* Validación final ($\text{KL} \le 0.05$, ARC-Challenge, GSM8K, distancia $\ge 0.4$).

## 5. Reconciliación Dimensional en Topologías No Dirigidas
El motor en Rust aplica dos primitivas algorítmicas al vuelo:

1. **Submuestreo Inteligente por Índices Calientes ($d_A > d_B$):** El kernel OpenCL selecciona en tiempo constante $\mathcal{O}(1)$ los $d_B$ canales con mayor varianza, precalculados estáticamente del profesor ($H_0$), conservando la entropía semántica.
2. **Proyección Lineal Adaptativa ($d_A < d_B$):** Instanciación de una matriz $W_{proj}$ que emula una proyección identidad.
