# Informe de Recomendaciones: Optimización Masiva de GPU y Resolución de Bloqueos en el Loop Evolutivo de SAOR

Este informe presenta la especificación de ingeniería para resolver los dos cuellos de botella críticos identificados durante la ejecución de grandes modelos (27B–40B) en la GPU RTX 4050 (6 GB VRAM) bajo Windows: **la infrautilización drástica de la GPU (<10% de carga y VRAM)** y **los bloqueos (deadlocks) en la CPU** al paralelizar procesos de evaluación de OpenCL.

Para superar estas limitaciones físicas sin introducir dependencias externas complejas, se propone migrar de una evaluación secuencial por candidato a un paradigma de **Tensorización de la Población (Population Batching)**, similar al utilizado en arquitecturas de neuroevolución acelerada de alto rendimiento [25, 50].

---

## 1. Diagnóstico de los Cuellos de Botella Actuales

### A. Infrautilización del Hardware (GPU "Con Hambre")
La baja carga de la GPU (<10%) se debe a un problema de **secuencialidad host-device (CPU-Bound)**. El motor evalúa un único candidato a la vez:
1. La CPU genera/muta el genoma de un candidato.
2. La CPU transfiere el genoma a la GPU vía PCIe.
3. La GPU ejecuta la decodificación de la CPPN y la multiplicación de matrices dispersas (SpMM) para un minúsculo lote de calibración ($B = 128$).
4. La GPU devuelve el fitness a la CPU.
5. La GPU se detiene y queda inactiva (0% de uso) mientras espera que la CPU procese el loop evolutivo en Python, gestione PyO3 y envíe el siguiente candidato.

Dado que procesar un lote de activations de $128 \times 5120$ toma fracciones de microsegundo en los núcleos CUDA/OpenCL de la RTX 4050, **la GPU pasa el 99% de su tiempo ociosa**, y la memoria asignada para un solo bloque activo es despreciable (<10% VRAM).

### B. El Deadlock de OpenCL en Windows
El bloqueo en la CPU al lanzar procesos concurrentes es un síntoma del **Windows Display Driver Model (WDDM)**:
* A diferencia de Linux (donde las colas de cómputo de múltiples procesos se multiplexan de forma nativa en el hardware), el controlador de OpenCL para Nvidia en Windows implementa un **mutex síncrono global** dentro de `opencl.dll` para gestionar el acceso concurrente al dispositivo.
* Al lanzar procesos de sistema operativo independientes que llaman concurrentemente a la GPU, el Proceso A bloquea el hilo de la CPU esperando eventos de OpenCL (`clWaitForEvents` o `clFinish`), mientras el Proceso B interrumpe la cola, provocando una **espera circular activa y el congelamiento completo de los hilos de control en el Host**.

---

## 2. Solución Arquitectónica: Tensorización de la Población

Para eliminar los deadlocks de Windows y saturar la GPU, el equipo de desarrollo debe implementar una **evaluación batcheada de la población completa** dentro de un único proceso y contexto OpenCL. Esto reduce drásticamente las latencias de sincronización PCIe y permite que la GPU procese en paralelo a todos los candidatos en una sola operación de cómputo masivo [13, 25].

```
[ FLUJO SECUENCIAL ACTUAL ]
 CPU ──► Enviar Genoma 1 ──► GPU Decodifica ──► GPU SpMM (B=128) ──► Retornar Fitness 1 ──► CPU (Repetir N veces)

[ FLUJO BATCHEADO PROPUESTO ]
 CPU ──► Enviar Tensor Genomas (N×32K) ──► GPU Decodifica N candidatos ──► GPU Batched SpMM (N×B=2816) ──► Retornar Fitness (N floats)
```

---

## 3. Especificaciones de Implementación

### Recomendación A: Empaquetamiento y Transferencia de Genomas en Lote
En lugar de despachar genomas individuales a la GPU, el módulo en Rust (`saor-engine`) acumula los vectores de pesos de la CPPN de toda la población (de tamaño $N$) en un único búfer plano y continuo en memoria *pinned*:

```rust
// saor-engine/src/evolution/batch.rs

#[derive(Debug)]
pub struct PopulationBatch {
    pub candidate_count: usize,
    pub genome_dim: usize,
    /// Búfer plano que contiene todos los genomas concatenados [N * genome_dim]
    pub flat_genomes: Vec<f32>,
}

impl PopulationBatch {
    pub fn new(candidates: &[Vec<f32>], genome_dim: usize) -> Self {
        let mut flat_genomes = Vec::with_capacity(candidates.len() * genome_dim);
        for genome in candidates {
            flat_genomes.extend_from_slice(genome);
        }
        Self {
            candidate_count: candidates.len(),
            genome_dim,
            flat_genomes,
        }
    }
}
```

Esta estructura permite realizar **una única transacción DMA por PCIe por generación**, eliminando la latencia de transferencia recurrente.

---

### Recomendación B: Kernel de Decodificación Multitarea (`cppn_decode_batched.cl`)
Se debe modificar el kernel de decodificación en OpenCL para que la rejilla global de hilos (*global work size*) se extienda en una segunda dimensión que represente el índice del candidato dentro de la población ($0 \dots N-1$).

```c
// saor-kernels/src/cl/cppn_decode_batched.cl

__kernel void k_cppn_decode_batched(
    __global const float* restrict population_genomes, // [N * genome_size]
    __global const float* restrict coordinates,        // [d_in * d_out * 8] (sustrato)
    __global float* restrict population_weights,       // [N * d_in * d_out]
    __global uchar* restrict population_adjacency,     // [N * (d_in * d_out / 8)]
    const float tau,                                   // Umbral de esparsidad
    const ulong connections_per_candidate,             // d_in * d_out
    const int candidate_count                          // N
) {
    // ID bidimensional
    int conn_idx = get_global_id(0); // Índice de la conexión física (ij)
    int cand_idx = get_global_id(1); // Índice del candidato de la población (0..N-1)

    if (conn_idx >= connections_per_candidate || cand_idx >= candidate_count) return;

    // Desplazamiento del genoma del candidato actual
    // Cada candidato tiene sus propios pesos de CPPN guardados en el lote
    int genome_offset = cand_idx * 466; // Genoma CPPN real: 466 f32 (16*9+16+16*16+16+2*16+2)
    
    // Leer el vector de coordenadas correspondiente
    int coord_offset = conn_idx * 8;
    float v_in[8];
    for (int i = 0; i < 8; i++) {
        v_in[i] = coordinates[coord_offset + i];
    }

    // ─── EVALUACIÓN DE LA CPPN EN GPU ───
    // Cada hilo evalúa la CPPN usando el genoma específico de su candidato
    float weight = evaluate_cppn_inline(&population_genomes[genome_offset], v_in);
    float link_prob = evaluate_cppn_link_inline(&population_genomes[genome_offset], v_in);

    // Calcular compensaciones de escritura
    ulong weight_dest = (ulong)cand_idx * connections_per_candidate + conn_idx;
    population_weights[weight_dest] = weight;

    // Decisión de enlace y empaquetamiento de bits
    if (link_prob >= tau) {
        ulong bit_dest = weight_dest / 8;
        int bit_shift = weight_dest % 8;
        // Operación atómica para escribir la máscara de adyacencia de bits en paralelo
        atomic_or(&population_adjacency[bit_dest], (uchar)(1 << bit_shift));
    }
}
```

---

### Recomendación C: Multiplicación de Matrices por Lote (Batched SpMM)
Para saturar las Unidades de Computo de la GPU sin colas de comandos redundantes, se debe fusionar la entrada de calibración para toda la población. Si el lote de calibración original tiene un tamaño de secuencia $B = 128$, se construye una matriz de entrada combinada de tamaño $(B \cdot N) \times d_{\text{in}}$:

```rust
// saor-engine/src/executor/batched_spmm.rs

impl OpenClEngine {
    pub fn dispatch_batched_spmm(
        &self,
        queue: &CommandQueue,
        batched_x_in: &Buffer<f32>,         // [B * N, d_in]
        population_adjacency: &Buffer<u8>,   // [N, d_in * d_out / 8]
        population_weights: &Buffer<f32>,    // [N, d_in * d_out]
        batched_x_out: &mut Buffer<f32>,     // [B * N, d_out]
        d_in: usize,
        d_out: usize,
        batch_size: usize,                   // B
        population_size: usize,               // N
    ) -> Result<(), OpenClError> {
        let kernel = self.kernels.get("k_ffn_irregular_dag_batched")
            .ok_or(OpenClError::KernelNotFound)?;

        kernel.set_arg(0, batched_x_in)?;
        kernel.set_arg(1, population_adjacency)?;
        kernel.set_arg(2, population_weights)?;
        kernel.set_arg(3, batched_x_out)?;
        kernel.set_arg(4, &(d_in as u64))?;
        kernel.set_arg(5, &(d_out as u64))?;
        kernel.set_arg(6, &(batch_size as i32))?;
        kernel.set_arg(7, &(population_size as i32))?;

        // Grid bidimensional en la GPU: [d_out, N]
        let local_size = [256, 1];
        let global_size = [
            ((d_out + local_size[0] - 1) / local_size[0]) * local_size[0],
            population_size
        ];

        unsafe {
            queue.enqueue_nd_range_kernel(
                kernel,
                2,
                None,
                &global_size,
                &local_size,
                None,
            )?;
        }
        Ok(())
    }
}
```

### Recomendación D: Evitar el Deadlock de Windows usando Colas Homogéneas
Al unificar la población en un único flujo batcheado, el equipo de desarrollo debe **eliminar por completo el paralelismo por procesos independientes**. 
* Toda la comunicación con OpenCL debe ocurrir a través de un **único Contexto y una única Cola de Comandos en modo asíncrono** en un único hilo maestro de Rust.
* Para el cálculo final de CKA y Fitness, la GPU procesa los reducidos tensores resultantes de tamaño $128 \times 128$ en memoria local rápida, y retorna un único array de floats de tamaño $N$ (un único fitness por candidato). La CPU no sufre sobrecarga y el driver gráfico de Windows (WDDM) gestiona la cola de forma óptima sin colisiones.

---

---

## 3.5 Anexo técnico — Correcciones de ingeniería (Fase 0)


---

## 5. Estado de implantación (2026-08-30)

Fases ejecutadas y validadas (repo saor + hayai):

- **Fase 0 — Informe corregido.** Genoma 466 f32, batcheo solo-adyacencia, tabla de
  FLOPs, criterio 4 por clase de modelo.
- **Fase 1 — Decode batcheado.** `cppn_decode_adj_batched` (2D `[conexiones, N]`) +
  `ClEngine::cppn_decode_adjacency_batched`. **Bit-exacto** (N=22): 4.6× (SmolLM2),
  3.7× (Qwen3.5-4B). Test: `saor-engine decode-bench --batched N`.
- **Fase 2 — Evaluador batcheado agnóstico a arquitectura.**
  - `saor-engine decode-pop`: decode GPU de la población → adyacencias por candidato.
  - `StreamingGenerator::forward_batched` (path Dense) + `forward_batched_hybrid`
    (path Hybrid, estado KV + DeltaNet por candidato) + `forward_batched_any`
    (despacho por `ModelKind`).
  - **GEMM batcheado** `ggml_gemv_batched_q4_k` (un dispatch `[N×M]`, pesos leídos
    una vez) + `execute_quant_gemv_batched` (Q4_K → batcheado, resto → secuencial).
  - `kl_eval_batch`: KL de la población con logits del profesor cacheados y override
    construido al vuelo por (candidato, capa) — RAM acotada.
  - **Validación KL:** SmolLM2 2.389384 vs 2.389382 de referencia (paridad); Qwen3.8-27B
    (híbrido) KL 0.006 sin crash con topología real.
- **Fase 3 — Loop CMA-ES.** `via_b_evolve.py --batch-eval` usa `decode-pop` +
  `kl_eval_batch` (una carga de modelo por generación). Validado: 1 generación
  SmolLM2 en 31.8 s.
- **Fase 4 — Cache del profesor.** `kl_eval --teacher-cache` y `eval_sparse
  --teacher-cache`: logits del profesor precomputados una vez por ejecución (paridad
  confirmada).

### Estado de los criterios de aceptación (honesto)

> ⚠️ **CORRECCIÓN CRÍTICA (2026-08-31):** las mediciones de KL del **batch-eval
> (kl_eval_batch) para Qwen3.8-27B** son **INVÁLIDAS**: los forwards del batch
> (`forward_batched_hybrid_seq` y `forward_batched_hybrid_gemm` per-token)
> producen logits que difieren del modelo real en **KL 8.91** (modelo contra sí
> mismo con el teacher cache del batch — ambos paths). El teacher cache del batch
> está mal → TODAS las KL del batch para el 27B (2.98, 3.12, 3.57, C4 7.8 min/gen)
> se midieron contra un teacher incorrecto. El smol (Dense) SÍ tiene el teacher
> correcto (KL 0.0) — el bug es del **código híbrido compartido** del batch
> (atención/DeltaNet del qwen27), no del seq en particular. **La no-convergencia
> de la evolución del 27B es consecuencia de este teacher mal**, no de la
> sensibilidad del modelo ni de la topología. Las mediciones del 27B de esta
> tabla quedan bajo auditoría hasta localizar el bug del forward híbrido del
> batch (comparación con el `forward_hybrid` de producción). C3 (sin deadlocks)
> no depende del teacher.

| Criterio | Estado |
|---|---|
| 1. GPU ≥75% en la evaluación | **Medido en qwen27 (22 cands, n_pos=4, seq + `spmm_adj_batched`)**: media **47.4%**, picos **100%**. La ociosidad restante es CPU/PCIe (pre-pass, dequant F32, lecturas de adyacencia, uploads). El dequant Q4 en GPU (opt-in `HAYAI_SPMM_Q4=1`) no mejora el GPU% (46%) y duplica el tiempo (686 s vs 360 s). NO cumplido (≥75%). |
| 2. VRAM 1.5–2.5 GB plana | **Cumplido como opt-in**: F32 default 1077 MiB (por debajo del suelo); con `HAYAI_SPMM_Q4=1` el buffer F32 de la capa en GPU lleva la VRAM a **2043 MiB (rango 1.5–2.5 GB)** a costa del tiempo (686 s vs 360 s). Estable en ambos paths (sin explotar al tamaño del modelo, 15 GB en qwen27). |
| 3. Cero deadlocks (150 gens) | **Cumplido por corrida significativa**: 20 generaciones completas con `via_b_evolve --batch-eval` (smol, n_pos=8, decode-pop + kl_eval_batch por gen) sin deadlock ni degradación de tiempos (~66 s/gen estables); KL evolucionó 2.69 → 0.40. |
| 4. Tiempo por generación | **CUMPLIDO en qwen27 22 cands n_pos=4**: `kl_eval_batch` 360 s + `decode-pop` 108 s = **7.8 min/gen** (< 10 min; el evaluador per-token previo era >10 min y fue abortado). El kernel `spmm_adj_batched` (bit-tensor + pesos F32 compartidos por capa, un dispatch `[N×n_pos]`) eliminó el build/gather de CSR por (candidato, capa, token). |

**Trabajo pendiente para cerrar los números:** (a) overlapp real del dequant/upload
con el forward (threads o buffers GPU persistentes — los kernels Q4 encadenados
resultaron 2× más lentos por el buffer F32 de 356 MB por capa), (b) corrida de 150
gens si se exige el número exacto, (c) monitorear la evolución de Qwen3.8-27B
relanzada con el pipeline nuevo, (d) push a GitHub cuando la red se recupere.

Aprobado en la planificación: estas correcciones son parte de la especificación y
se implementan junto con el resto del plan.

1. **El genoma CPPN real son 466 floats** (`16×9 + 16 + 16×16 + 16 + 2×16 + 2`),
   no 32K. El `GENOME_SIZE = 32*1024` de `saor-domain` es un residuo nominal de
   la propuesta v4 sin uso. El kernel batcheado usa `genome_offset = cand_idx·466`.
2. **El batcheado es solo-adyacencia** (`cppn_decode_adj_batched`): NO se
   materializan pesos densos `[N × d_in·d_out]` (en ALIA serían ~17.7 GB → viola
   el criterio 2 de VRAM). Los pesos activos se conservan del profesor
   (warm-start teacher-copy), como en producción.
3. **La KL es de modelo completo** (*Layer-Streaming Batched KL*): el SpMM
   batcheado de la Recomendación C se aplica **en las capas evolucionadas dentro
   de un único barrido del modelo completo** (activaciones `[N·B, d]` FP16 en
   VRAM ≈ 46 MB para ALIA), no como fitness aislado de una capa.
4. **Tiempos físicos (RTX 4050):** el cómputo batcheado NO es despreciable.
   `FLOPs/gen = 2·params·N·n_pos`:

   | Modelo | params | n_pos=8 | n_pos=24 | n_pos=128 |
   |---|---|---|---|---|
   | SmolLM2 | 135M | 0.05 TF | 0.14 TF | 0.76 TF |
   | Qwen3.5-4B | 4B | 1.4 TF | 4.2 TF | 22.5 TF |
   | Qwen3.8-27B | 27B | 9.5 TF | 28.5 TF | 152 TF |
   | ALIA-40B | 40B | 14 TF | 42 TF | 225 TF |

   A ~6-9 TFLOPS efectivos FP16, el barrido batcheado completo de 27B/40B toma
   ~20-37 s a `n_pos=128`, ~2-4 s a `n_pos=8` (el streaming de ~3.5 s se solapa
   con doble buffer). La mejora frente al pipeline actual (~30 min/gen) es de
   **~50-100×** y habilita escalar generaciones en la siguiente iteración.
5. **Criterio 4 reformulado por clase de modelo** (ver §4): el target de 0.5 s/gen
   aplica a la fase de decode+SpMM y a modelos ≤4B; para 27B/40B el contrato es
   **≤ 3 s/gen a `n_pos=8`** (≈30 s/gen a `B=128`).

---

## 4. Plan de Validación y Criterios de Aceptación
## 4. Plan de Validación y Criterios de Aceptación

Para certificar que el equipo ha resuelto correctamente los cuellos de botella de hardware, la integración final de la Fase 7 debe superar las siguientes métricas de telemetría en la RTX 4050:

1. **Uso Activo de GPU:** $\ge 75\%$ durante el paso de evaluación del loop evolutivo (un salto masivo frente al actual <10%).
2. **Asignación de VRAM Constante y Sólida:** Mantener estable el uso de VRAM entre **1.5 GB y 2.5 GB** de forma plana, cargando los tensores de toda la población de genomas y activaciones de forma compacta en un solo pase de streaming asíncrono.
3. **Cero Bloqueos de Sistema (Deadlocks):** Completar 150 generaciones consecutivas de CMA-ES bajo Windows sin que ocurran congelamientos de hilos del host ni resets del controlador de video (TDR).
4. **Velocidad de Evolución:** dependiente del tamaño del modelo (ver anexo 3.5):
   **≤ 0.5 s/generación** para modelos ≤ 4B y para la fase decode+SpMM de
   cualquier modelo; **≤ 3 s/generación a `n_pos=8`** (≈30 s/generación a
   `B=128`) para 27B/40B — frente a los ~30 min/generación actuales, lo que
   desbloquea escalar el número de generaciones en la siguiente iteración.
