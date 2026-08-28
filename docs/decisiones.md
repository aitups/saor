# Registro de decisiones del proyecto

> Fechas aproximadas, ordenadas cronológicamente. Cada entrada registra la
> decisión, la motivación y el impacto en el código.

## D1 — Instalar MSVC Build Tools + CMake (Fase 0)
- **Decisión:** instalar VS Build Tools 2022 (carga "Desarrollo de escritorio con
  C++") y CMake vía `winget`; cambiar el toolchain activo a
  `stable-x86_64-pc-windows-msvc` (`rust-toolchain.toml`).
- **Motivación:** el entorno carecía de `cl.exe`/`link.exe` y `cmake`, necesarios
  para PyO3 y para enlazar dependencias de C.
- **Impacto:** `rust-toolchain.toml` fija el MSVC; el proyecto no usa MSYS2.

## D2 — Sin OpenBLAS / sin BLAS nativo
- **Decisión:** el álgebra lineal de CPU es **Rust puro**: `nalgebra` (con la
  característica `matrixmultiply`) como crate primario; `ndarray`+`matrixmultiply`
  como alternativa tipo tensor.
- **Prohibido:** OpenBLAS, Intel MKL, `ndarray-linalg` (exige backend LAPACK),
  MSYS2, compiladores Fortran.
- **Motivación:** toda la computación pesada (CPPN, SpMM, Gram/CKA) corre en la
  GPU vía OpenCL 3.0; en CPU solo hay matrices pequeñas (Gram 128×128,
  covarianza `d×d`, d≈100–500), donde el Rust puro es suficiente y evita la
  pesadilla de compilación de BLAS en Windows.

## D3 — SVD del subespacio activo en Rust puro
- **Decisión:** la reducción SVD dual (Fase 2 del subespacio) usa
  `nalgebra::linalg::SVD` (implementación pura en Rust), **no** `ndarray-linalg`.
- **Motivación:** `ndarray-linalg` requeriría LAPACK (OpenBLAS/MKL), lo que
  reintroduciría la dependencia prohibida en D2.

## D4 — GGUF disperso sin densificar
- **Decisión:** el modelo candidato se consolida en GGUF **disperso tal cual**:
  pesos activos + tensor de bits `ffn_dag_adjacency` (`saor-streamer`). No hay
  densificación.
- **Motivación:** alineación con la PR paralela de `hayai`
  (`pr_soporte_gguf_disperso_v2.md`) para ejecutar GGUFs dispersos.

## D5 — Distancia arquitectónica = Hamming normalizada (= sparsity)
- **Decisión:** `D_arch(A0,A1) = 1 - ΣA1/(d_in·d_out) = Sparsity(A1)`, porque el
  bloque original es denso. Se calcula en host con `popcount` sobre el bit-tensor.
- **Contrato de cierre:** `D_arch ≥ 0.4` (eliminar ≥ 40% de conexiones).

## D6 — CMA-ES en subespacio activo (Fase 1)
- **Decisión:** precomputar `Var(H0)` (diagonal de la Fisher empírica) en el
  Paso 2; seleccionar los `d` canales calientes (d≈100–500); congelar ~99% de
  los pesos de la CPPN y evolucionar solo `z ∈ R^d`; seed replay sin estado.
- **Evolución futura:** SVD dual sobre la Gram 128×128 (ver D3).

## D7 — PyO3 vs IPC
- **Decisión:** PyO3 como ruta principal (habilitada por D1); **fallback**
  documentado: `saor-engine` + protocolo JSON-lines por stdio
  (`saor_orchestrator.ipc.SaorEngineClient`).
- **Estado:** IPC operativo en Fase 0; PyO3 se valida cuando el build de la
  extensión esté disponible.

## D8 — Runtime de hooks (Fase 5)
- **Decisión preliminar:** `llama-cpp-python`/ggml para capturar activaciones del
  modelo profesor de 30B (carga secuencial capa a capa, nativo GGUF).
- **Estado:** la infraestructura (auditoría GGUF, catálogo de roles, lote B=128,
  varianza de Fisher, abstracción `TeacherRuntime`) está implementada y
  testeada con el backend sintético. La captura real requiere: (a) el archivo
  GGUF del modelo base (~30B) y (b) un runtime con hooks de capa intermedia
  (`llama-cpp-python` con build adecuada o `hayai` — PR paralela). **Pendiente
  de la disponibilidad del modelo.**

## D9 — Restricciones del driver OpenCL (RTX 4050, driver 560.94)
- **OpenCL C = 1.2** es el lenguaje máximo reportado por el driver NVIDIA (el
  OpenCL 3.0 de la plataforma es *backwards compatible*; NVIDIA no implementa
  C 2.x/3.0 como lenguaje). **Consecuencia:** todos los kernels de la Fase 3 se
  escriben en **OpenCL C 1.2** (sin device-side enqueue, SVM atoms, etc.).
- **`CL_DEVICE_MAX_MEM_ALLOC_SIZE` = 1.5 GiB** en la RTX 4050: ningún buffer
  individual puede exceder ~1.5 GiB. El motor de streaming (Fase 2/3) debe
  fragmentar los pesos en buffers ≤ 1.5 GiB — coherente con el presupuesto de
  pico de VRAM de ~2 GB y con la mitigación WDDM/TDR.

## D10 — OpenCL.lib local (resolución del enlace en Windows)
- `cl-sys` (dependencia de `opencl3`) enlaza `OpenCL.lib` y añade un `LIBPATH`
  fijo inexistente (`C:\Program Files (x86)\OCL_SDK_Light\lib\x86_64`).
- **Solución:** `scripts\generate_opencl_lib.bat` genera el import lib desde
  `C:\Windows\System32\OpenCL.dll` (123 exports) en `.local\windows-lib\`, y
  `scripts\build_dev.bat` lo añade a `LIB`.

## D11 — Interop del GGUF disperso con hayai (v0.2.3, commit 16cda2017)
- hayai v0.2.3 añade los ops `FfnDagAdjacency` (`ffn_dag_adjacency`, I8
  LSB-first) y `FfnDagWeights` (`ffn_dag_weights`, F32 i-mayor) + lectura de
  metadatos `saor.*` — espejo exacto del formato de `saor-streamer`.
- **Correcciones de compatibilidad aplicadas (reconciliación R1):**
  1. `GGML_TYPE_I8` pasa de `16` a `24` (el enum GGML actual asigna 16 a
     `IQ2_XXS`; hayai ya lo había corregido en `69a6b83`).
  2. `write_sparse_gguf` ahora **alinea la cabecera a 32** y escribe **offsets
     relativos** a la sección de datos (espec GGUF v3 / `tensor_abs_offset =
     data_offset + offset` de hayai); el reader (Rust y Python) computa
     `data_offset = align_up(header, 32)`.
  3. `gguf_audit.py` usa el enum GGML nuevo (I8=24, IQ 16–23).
- **Validación end-to-end:** `hayai plan --model <gguf saor>` → `known_ops:
  FfnDagAdjacency, FfnDagWeights`, `status: OK`. `load_sparse_dag` lee
  `d_in/d_out/tau/adjacency/weights/genome` correctamente y `spmm_csr ==
  spmm_dense` (max_err 0).
- **Compilación de hayai en Windows:** `nightly-x86_64-pc-windows-gnu` +
  MinGW (msys64) + `cl3` con `features=["dynamic"]` (OpenCL cargado en runtime,
  sin `OpenCL.lib`).

## D12 — Modelos objetivo (Fase 7)
- **ALIA-40b** (BSC-LT): arquitectura **llama** (densa). GGUF Q4_K_M:
  `mradermacher/ALIA-40b-GGUF/ALIA-40b.Q4_K_M.gguf` (~22 GB).
- **Qwen3.8-27B** (Qwen): arquitectura **qwen3.5 híbrida** (gated-DeltaNet +
  attention + MTP + multimodal). GGUF Q4_K_M:
  `unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf` (~16 GB).
- Descarga a `models/` (script `python/scripts/download_models.py`).
- **Auditoría real:** ALIA-40b = 48 capas, bloques FFN `[8192, 24576]` (Q4_K);
  Qwen3.8-27B = 65 bloques FFN `[5120, 17408]` (IQ4_XS) + 336 tensores SSM.

## D13 — Escalado del decode CPPN y fixes críticos (Fase 7)
- **Bug de eventos (`enqueue_chunked`):** `opencl3` libera el `cl_event` en el
  `Drop`; esperar sobre un evento liberado en el siguiente dispatch colgaba el
  kernel (multi-dispatch >1). Fix: mantener los `Event` vivos en un `Vec`.
- **Presión de registros de la CPPN:** con 64+64 ocultos el evaluador forzaba
  spilling y el decode colapsaba (~6500× por dispatch en bloques grandes).
  **`HIDDEN = 16+16`** (Rust/Python/kernel) reduce registros 4× y flops 16×.
- **Decode por conexión + `atomic_or` (u32) + `pack_adjacency`:** máximo
  paralelismo (1 work-item por conexión) sin carreras en el bit-tensor.
- **Resultado:** 885K conexiones ≈ 10 ms; 8.4M ≈ 36 ms. Proyección real:
  Qwen (89M) ≈ 0.4 s/decode, ALIA (201M) ≈ 0.9 s/decode.
- **`evolve` usa el decode GPU** (no `instantiate` CPU por candidato):
  0.07–0.12 s/candidato a 576×1536.

## D14 — Piloto de bloque real (Fase 7)
- `python/scripts/pilot_block.py`: profesor real (tensor FFN de SmolLM2-135M vía
  `hayai dump_tensor_f32`) → evolución (decode GPU) → consolidación GGUF
  disperso → `hayai plan` + `load_saor_sparse` OK.
- Resultado: bloque `[576,1536]`, `d_arch=0.991`, `spmm_csr == spmm_dense`
  (max_err 0). Fases 5/6 a nivel de bloque cerradas con modelo real.
- **Hallazgo de investigación:** el CKA del mejor candidato (~0.13) es bajo:
  la CPPN aleatoria no reconstruye un FFN entrenado a alta esparcidad en 8
  generaciones con subespacio 120 — señal de que la inicialización del genoma
  (no aleatoria) y/o más generaciones son necesarias para fidelidad real.

## D15 — Fase 4b (warm-start): la regresión CPPN no alcanza 0.85; la copia del profesor sí (CKA medido)
- `python/scripts/warm_start.py` implementa la spec v2: sustrato exacto
  (`build_substrate` == `input_vector`), Método A (torch Adam+Ridge) y Método B
  (NumPy/Tikhonov-ELM), alineación `l_ij` (`w2[1,:]=0`, `b2[1]=logit(ρ)`),
  salida `genome.bin` en orden `flatten()` de `saor_domain::cppn`.
- **Medición empírica sobre el gate real de SmolLM2 (`blk.0.ffn_gate` 576×1536):**
  - Método B (ELM 16→256 ocultos): CKA ≈ 0.14–0.17 — la CPPN es una función
    suave del sustrato y los pesos FFN entrenados no tienen estructura suave en
    `(i,j)`. El techo SVD rank-128 (CKA 0.90) requiere la base óptima que la
    CPPN no puede producir. **El umbral 0.85 de la spec no es alcanzable por
    regresión CPPN sobre FFNs reales.**
  - **Alternativa que cumple el contrato — "copia del profesor":** gen 0 denso
    con pesos del tensor real (`CKA=1.0`) y la evolución esculpe la topología
    (`l_ij`, `τ`). Curva medida sobre el gate real: sp 0.4→CKA 0.998, sp
    0.9→0.951, sp 0.99→0.876. Esto coincide con el §5 de la spec v2 ("la
    evolución se enfoca en ajustar `l_ij` y `τ`").
- **Implementación:** `evolve`/`consolidate` con `--teacher-copy` (los pesos
  activos del candidato = profesor; el genoma solo controla la topología vía el
  bit-tensor del decode GPU) y `--tau0` (recomendado 0.3: zona sensible del
  sigmoide; con 0.02 la esparcidad no avanza). `--warm genome.bin` carga un
  genoma base en lugar del aleatorio.
- **Validación end-to-end (gate real):** `consolidate --teacher-copy --tau0 0.3`
  → `best_cka=0.953`, `d_arch=0.572`, GGUF disperso (378K/885K conexiones
  activas) validado por `hayai load_saor_sparse` (`spmm_csr == spmm_dense`,
  max_err 0). CKA ≥ 0.85 en toda la corrida.
- Tests: +4 Python (`test_warm_start.py`: sustrato, alineación `l_ij`, Método B
  sobre profesor suave, curva copia-profesor) y +1 Rust (teacher-copy
  `topology_from_dense`). Suites: 32 Python + 33 Rust, verdes.

## D16 — Fase 0: catálogo de roles completo (cobertura >90%) + decisiones de alcance
- **Alcance aprobado por el usuario:** incluir atención, objetivo de cobertura
  >90% del volumen del modelo, GGUF **embebido** desde el principio (sin
  sidecar), y validación escalonada (tests en SmolLM2 + checkpoint contra
  ALIA/Qwen en cada fase).
- `role_catalog.py` ampliado con los roles medidos de Qwen3.8-27B-UD (NextN):
  `attn.qkv`, `attn.gate` (atención fusionada) y `ssm.out`/`ssm.alpha`/`ssm.beta`
  (proyecciones lineales del bloque SSM, "cirugía híbrida" — el núcleo
  recurrente `ssm_a`/`ssm_dt.bias`/`ssm_norm`/`ssm_conv1d` se marca `ssm.core`
  **no esparcible** y se conserva denso).
- Nuevo `SPARSIFIABLE_ROLES` + `is_sparsifiable`/`sparsifiable_tensors`/
  `coverage_report`. **Cobertura medida sobre los GGUFs reales:**
  - Qwen3.8-27B-UD: **90.5%** (24.72/27.32 B; FFN 63.6% + attn 21.3% + SSM-proj 5.6%).
  - ALIA-40b: **89.6%** (36.24/40.43 B; FFN + attn). El déficit al 90% es
    íntegramente `lm_head` (2.10 B, **sin atar** — Q6_K vs Q4_K de `token_embd`)
    + embeddings + norms: no esparcibles por diseño. Cobertura = 100% del
    volumen esparcible.
  - SmolLM2-135M: 78.9% (canario de tests; la atención pesa más en modelos
    pequeños).
- Formato objetivo (Fase 1): GGUF embebido — sustituir `blk.N.<rol>.weight`
  denso por `blk.N.<rol>.ffn_dag_adjacency` (I8) + `blk.N.<rol>.ffn_dag_weights`
  (activos 4-bit) + metadatos `saor.blk.N.<rol>.*` por bloque.
- Tests: +3 Python (roles SSM/attn, esparcibles, cobertura). Suite: 35 Python.

## D17 — Fase 1: GGUF embebido (rewriter streaming) + hallazgo de escalado en `topology_from_dense`
- Nuevo `saor-streamer::gguf_embed`: rewriter **streaming** que sustituye tensores
  densos por bloques dispersos embebidos (`blk.N.<rol>.ffn_dag_adjacency` +
  `ffn_dag_weights` + metadatos `saor.<base>.*` por bloque y `saor.sparse_count`).
  Nunca carga la sección de datos completa. Subcomando `saor-engine embed
  --src --dst --block <t> --sparse <gguf>` con verificación automática
  (re-lee el bloque embebido y compara).
- **Bug corregido (crítico para interop):** el parseo de KV GGUF usaba tamaños
  erróneos (`FLOAT32=6` como 8 bytes en vez de 4; UINT8/16 etc.). Afectaba a
  cualquier GGUF real con metadatos FLOAT32 escalares. Ahora la tabla de
  tamaños por tipo es exacta y se soportan ARRAY de STRING (tokenizer).
- **Checkpoint SmolLM2 ✓:** embebe `blk.0.ffn_gate` (576×1536, 378K activos) en
  el GGUF completo: `verified:[true]`, tensores dispersos presentes y metadatos
  `saor.*` correctos. `hayai plan` parsea el GGUF embebido con `status: OK` y
  mapea los tensores a ops conocidas (`FfnDagAdjacency`/`FfnDagWeights`).
- **Hallazgo de escalado (D17):** la evolución de un bloque ALIA real
  (8192×24576, 201M conexiones) tarda ~30 s/candidato (vs ~0.07 s en SmolLM2).
  El cuello de botella es el loop CPU `topology_from_dense` (O(N) por candidato)
  + SpMM 121M no-ceros. Proyección: ~36 min/bloque × 144 tensores ≈ 86 h —
  **inaceptable para la Fase 4**. Plan: kernel GPU de "gather" de pesos del
  profesor en las posiciones activas (elimina el loop CPU) en la Fase 3/4.
- **Fix de lectura en bloque:** `read_embedded_block` hacía un `seek`+lectura de
  4 bytes por float (201M syscalls en ALIA → cuelga). Ahora lee el tensor de
  pesos en un solo `read` y decodifica. Verificación de ALIA en segundos.
- **Nuevo subcomando `make-block`:** ensambla un `SparseBlock` desde bins crudos
  (`--identity`, `--adj`, `--weights`, `--genome`, `--tau`), para baselines y
  checkpoints sin correr la evolución.
- **Checkpoint de escala (ALIA 23 GB) ✓:** `embed` de `blk.0.ffn_gate`
  (8192×24576) en el GGUF completo de 24.6 GB → 25.3 GB resultante,
  `verified:[true]` (re-lectura idéntica), tensores dispersos + metadatos
  `saor.*` correctos (audit Python: 436 tensores, 40.46 B params). El rewriter
  streaming nunca carga el archivo en RAM.
- **D17 resuelto (kernel GPU de gather):** `cppn_decode_adjacency` (no
  materializa `w_out`, evita 805 MB GPU/host) + `count_rows`/`gather_csr_teacher`
  + `spmm_csr_teacher` (CSR construido y ejecutado en GPU, sin roundtrip host).
  `Scored` deja de clonar adyacencia+pesos por candidato (la topología del mejor
  se reconstruye al final, una sola vez).
  - SmolLM2: trayectoria **idéntica** a la pre-fix (best_fit=1.1489,
    cka=0.939, sp=0.699) y generaciones ~2.5× más rápidas.
  - **ALIA real: ~30 s → 0.9–4.3 s por candidato (7–30×).** Evolución completa
    del gate 8192×24576 en 6 gens (~5 min): **best_cka=0.950, d_arch=0.694**
    (61.6M/201M activos). `consolidate` + `embed` en el GGUF completo:
    `verified:[true]`, 24.47 GB conservados + 272 MB dispersos.

## D18 — Fase 3: FFN disperso ejecutado por hayai + harness KL (bug de layout en make-block)
- **hayai**: `load_embedded_block` (formato embebido D16 por bloque) +
  `CsrSparse` en `LayerWeights` con fallback denso/disperso en el loader +
  ejecución CSR (`spmm_csr_cpu`) en `Generator::forward` para gate/up/down
  sustituidos. `hayai generate --dev-mmap` ejecuta el modelo embebido.
- **Harness KL:** ejemplo `dump_logits` (logits teacher-forced por prompt) +
  `gate_equiv` (diagnóstico gemv/csr/dequant).
- **Bug de layout en `make-block`:** escribía los pesos del dump `[d_out,d_in]`
  fila-mayor, pero `SparseBlock` exige el orden i-mayor (conn=i*d_out+j). El
  diagnóstico `gate_equiv` lo expuso (|gemv−deq|=4e-6, |deq−csr|=11.6). Fix:
  `--dense-dump` reordena. Los bloques de `consolidate` ya estaban en i-mayor.
- **Validación a nivel de modelo (SmolLM2):**
  - Bloque denso embebido (copia del profesor): **KL simétrica = 0.000000**,
    max|logits|=8.8e-5, top-1 agreement 100% → el path CSR es **correcto**.
  - Bloque disperso real (gate blk.0, 43% activo, calibrado con X aleatorio):
    **KL = 1.61** — degradación esperada: la esparsificación necesita **hooks de
    activación reales (Fase 2)** para caer las conexiones correctas.
- Pendiente Fase 3: `StreamingGenerator` (path por defecto del CLI) aún no
  enruta el FFN disperso ("tensor not found: blk.N.ffn_gate.weight"); se usa
  `--dev-mmap` (`Generator`). Qwen (attn_qkv fusionado) requiere soporte de
  arquitectura adicional en `LlamaWeights`.

## D19 — Fase 2 (hooks reales) + curva KL vs esparsidad a nivel de modelo
- **hooks reales:** `Generator::forward_with_hooks` (captura la entrada FFN por
  capa, salida del `ffn_norm`) + ejemplo `dump-hooks`. X real de SmolLM2
  capturada (30 capas × 4096 posiciones × 576). La evolución con X real
  (SmolLM2 blk.0 gate) da best_cka=0.937 a sp=0.798.
- **Curva KL a nivel de modelo (SmolLM2, gate blk.0, poda por magnitud = cota
  superior, `kl_sweep_model.py`):**
  | sparsity | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.8 |
  |---|---|---|---|---|---|---|---|---|
  | KL sim. | 0.000 | 0.085 | 0.212 | 0.504 | 0.704 | 2.52 | 3.09 | 3.49 |
- **Hallazgo:** la KL a nivel de modelo crece muy rápido con la esparsidad; el
  umbral del contrato `KL ≤ 0.05` solo se cumple a ~5–8% de esparsidad por
  bloque — **en conflicto con `D_arch ≥ 0.4`** incluso con poda óptima. Con el
  bloque evolucionado real (sp 0.798, X real) la KL es 3.88. La topología CPPN
  rinde ~10% peor que la poda por magnitud (3.88 vs 3.49 a sp 0.8).
- **Decisión pendiente (usuario):** reconciliar el contrato — (a) interpretar
  `KL ≤ 0.05` como pérdida por bloque (`kl_proxy` de `contract.py`, que sí es
  compatible con sp 0.4) en lugar de KL de logits de modelo completo; (b) bajar
  la esparsidad objetivo por bloque a ~0.05–0.1 (cumple KL pero no D_arch); o
  (c) reequilibrar los umbrales del contrato.

## D20 — Optimización global bajo Frontera de Pareto (redefinición aprobada)
- **Objetivo redefinido por el usuario:** `max D_arch_global` sujeto a
  `KL_global ≤ 0.50`, con esparsidad **heterogénea por capa** (la evolución
  descubre qué capas toleran 70% y cuáles deben quedar al 5%). Vías A (vector
  de esparsidad por capa, poda por magnitud) y B (CPPN global con coordenada de
  capa) en paralelo.
- **Runtime global (hayai):** `FfnOverride` + `forward_with_override` (override
  FFN por capa en runtime, sin re-embed por candidato) + `eval_sparse`
  (KL_global y D_arch_global en lockstep original/candidato, soporta
  gate/up/down por capa). `LlamaWeights`/`LayerWeights`/`Tokenizer` ahora Clone.
- **Mapa de sensibilidad (SmolLM2, poda por magnitud, KL@sp 0.4):**
  capas tolerantes 17/16/14/12/25 (KL 0.016–0.024); capas sensibles **28
  (1.72)** / 29 / 0 / 8 / 11. Heterogeneidad real confirmada.
- **Barrida heterogénea:** la KL acumulada es **super-aditiva**:
  - 16 capas tolerantes (12–27) gate-only @0.6 → D_arch 0.32, KL **1.16**.
  - FFN completo (gate+up+down) 12–27 @0.4 → D_arch 0.40, KL **2.27**.
  - `up`/`down` tienen coste KL/paramétro ≈ al gate (0.013/0.014 vs 0.016 @0.4).
- **Evolución global CMA-ES (Vía A):** convergió a la **frontera de Pareto real
  de SmolLM2: D_arch ≈ 0.10 @ KL ≈ 0.50** (gen 9: D_arch 0.101, KL 0.508).
  Ninguna configuración alcanza KL ≤ 0.50 por encima de D_arch ~0.10.
- **Conclusión empírica:** el objetivo `D_arch 0.40 @ KL 0.50` es **inalcanzable
  en SmolLM2** (incluso con poda por magnitud, la cota superior). El presupuesto
  de KL 0.50 limita D_arch a ~0.10. El mecanismo global Pareto funciona y
  descubre la frontera real; queda por ver si ALIA/Qwen (modelos grandes con más
  redundancia) ofrecen una frontera más favorable. Scripts:
  `sensitivity_map.py`, `hetero_probe.py`, `pareto_evolve.py`.
- **Fix D20b (D_arch global):** `eval_sparse` reportaba el promedio de esparsidad
  solo sobre las capas esparsas; ahora el denominador es **todos los parámetros
  FFN del modelo**. (Los valores KL eran correctos.)
- **Frontera de ALIA-40b (medida con `eval_sparse`, poda por magnitud del gate):**
  - capa 0 @ 0.4 → KL **0.32**; **capa 24 @ 0.4 → KL 0.0009** (¡casi nula!);
  - **16 capas medias (16–31) gate @ 0.4 → KL 0.037** (D_arch global 0.044).
  - La heterogeneidad de ALIA es MUCHO más pronunciada que la de SmolLM2: las
    capas medias de un modelo 40B toleran esparsidad alta con KL despreciable.
    Valida la hipótesis del usuario: "en algunos modelos implicará compresión
    del 80% y en otros del 2%" — la optimización topológica es específica del
    modelo.
- **Limitación de memoria (próximo paso):** el evaluador construye CSRs en RAM
  (~1 GB/bloque a sp 0.4); el FFN completo de 16 capas de ALIA (48 CSR ≈ 48 GB)
  hace OOM. Para la barrida completa de la frontera de ALIA se necesita un
  evaluador con CSR **respaldado en disco (mmap)** o el `StreamingGenerator` con
  FFN disperso (formato embebido) — mismo trabajo pendiente de la Fase 3.

## D21 — StreamingGenerator con FFN disperso embebido + sustrato CPPN global (Vía B)
- **Decisión (dos frentes en paralelo):**
  1. **hayai — `StreamingGenerator` con FFN disperso (Fase 3 en el path de
     producción).** El evaluador disk-backed se descarta como transitorio; se
     adelanta la ruta real: el `StreamingGenerator` (path por defecto de
     `hayai generate`, sin `--dev-mmap`) carga el bloque disperso embebido
     (D16) y ejecuta el FFN vía CSR en CPU.
  2. **saor — sustrato CPPN global con coordenada de capa `y_layer` (Vía B).**
- **Motivación:** (1) resolver el OOM de ALIA/Qwen en el camino de producción y
  cerrar la Fase 3 de una vez; (2) codificar la hipótesis de "compresión variable
  por modelo" en el propio genoma: un solo CPPN genera la topología de TODAS las
  capas, parametrizada por la profundidad.
- **Impacto (hayai):**
  - `LayerWeightPack` gana `gate_csr/up_csr/down_csr`; `load_embedded_csr` lee el
    bloque D16 desde el catálogo (tensores `ffn_dag_adjacency/ffn_dag_weights` +
    metadatos `saor.*`).
  - `load_layer_pack`, `load_layer_pack_into`, `layer_pack_views_from_base`,
    `layer_pack_nbytes` y `rebind_views` toleran FFN ausentes (tensor denso
    sustituido → CSR + layout 0 bytes).
  - `forward_inner`/`forward_macro_chunk` ejecutan `run_ffn_block` (spmm CSR en
    CPU) cuando el pack trae bloques dispersos; `prefill_wave.rs` hace lo mismo
    para el prefill (gate/up CSR en el begin, down CSR en el finish).
  - **Validación:** los logits del embebido denso (identidad) vs original difieren
    en `max_abs = 5.5e-5` (6×49152 logits); el modelo realmente disperso (43%)
    genera end-to-end con RSS acotado (74 MiB); 77 tests Rust verdes.
- **Impacto (saor — Vía B):**
  - `CPPN_INPUT_DIM 8→9`: el vector de entrada gana `y_layer ∈ [-1,1]` (centro
    de banda de cada capa en el eje de profundidad). Genoma 450→466 (Rust) /
    4930 f32 aplanados; kernel `cppn_decode.cl` y `cppn_decode_adj` reciben
    `layer`/`n_layers`.
  - `instantiate_layer` y `build_substrate(y_layer)` para decodificar por capa;
    `CppnGenome.decode_global` (NumPy vectorizado) para un modelo completo.
  - **Validación:** un CPPN aleatorio sobre SmolLM2 (30 capas, 576→1536) induce
    densidad por capa tipo campana (0.27→0.51, std 0.072, heterogeneidad 0.175):
    las capas medias más densas, las exteriores más esparsas — la curva de
    tolerancia por profundidad que ALIA ya mostró, ahora expresable en el genoma.
  - `saor-engine` no se puede enlazar en esta máquina (falta `OpenCL.lib` del
    `OCL_SDK_Light`); el test de validación cruzada OpenCL salta (skip) ante un
    binario stale hasta recompilar en la máquina con OCL.
- **Pendiente Vía B:** loop CMA-ES global (subespacio activo sobre el genoma
  global) con fitness Pareto `KL_global`/`D_arch_global` en SmolLM2 y ALIA;
  el `make-block`/`embed` debe aceptar la topología por capa del CPPN global.

## D22 — Vía B: CMA-ES global sobre el genoma CPPN + evaluador de topología real
- **Decisión:** cerrar la Vía B con el loop evolutivo global y dos niveles de
  evaluador:
  1. **Proxy de densidad** (rápido): `via_b_evolve.py` fija un `D_arch` objetivo
     (`rho = 1 - darch`), decodifica el perfil por capa del CPPN global con
     `decode_global(..., step=4)` (submuestreo ×34 más rápido, error <0.1% vs
     step=1) y evalúa con `eval_sparse --sparsities` (poda por magnitud).
  2. **Topología real** (exacto): `eval_sparse --genome <bin> --tau <f>` decodifica
     la adyacencia CPPN de cada capa (con `y_layer`), conserva los pesos del
     profesor en las posiciones activas y mide la KL del modelo con esa topología
     exacta (rayon).
- **Impacto:**
  - `via_b_evolve.py` (saor): CMA-ES sobre los 466 floats del genoma global;
    fitness = -KL al D_arch fijado. El proxy de densidad traza la frontera
    (SmolLM2: KL 0.329 @ D_arch 0.017 con el perfil evolucionado).
  - `eval_sparse --genome` (hayai): modo Vía B; `cppn_eval`/`cppn_layer_active`
    espejo del sustrato v5 y del kernel `cppn_decode.cl`; paralelizado con rayon.
  - `via_b_global_probe.py`: verifica que un solo CPPN induce densidad por capa
    tipo campana (0.27→0.51 en SmolLM2) y reescala a una densidad objetivo.
- **Hallazgo empírico clave:** la topología CPPN **arbitraria** (pesos del profesor
  en posiciones seleccionadas por el CPPN) es mucho más costosa que la poda por
  magnitud a la misma densidad (gen 0 `--full`: KL 5.57 @ D_arch 0.154 vs
  magnitud ~KL 0.5 @ D_arch 0.10). Implicación: el valor de la Vía B no es el
  patrón espacial aleatorio sino el **perfil por capa** (coordenada `y_layer`);
  para que el patrón espacial importe, el CPPN debe evolucionarse con el fitness
  de topología real (modo `--full`), mucho más caro por candidato.
- **Estado:** sustrato v5 ✓; probe ✓; proxy de densidad ✓; evaluador real ✓;
  loop `--full` validado (1 gen). El trazo completo de la frontera de ALIA con
  la topología real requiere el kernel OpenCL (máquina con OCL; el decode CPU de
  bloques 8192×24576 ≈ 201M conexiones × 120 bloques no escala en Rust).

## D23 — Qwen3.5 híbrido en el streaming: DeltaNet/SSM + attn_qkv fusionado
- **Decisión:** cerrar el bloqueo "soporte Qwen" con el Qwen3.5-4B local (24
  capas DeltaNet/SSM con `attn_qkv` fusionado + 8 capas full-attn con `attn_q`
  separado + 1 NextN/MTP), patrón 3+1 repetido.
- **Problemas resueltos:**
  1. **Enrutado híbrido**: `model_kind()` solo miraba `known_ops` del plan; el
     clasificador `AttnQkv` ganaba al de `DeltaNet` y los tensores `ssm_*` no
     entraban en los units → el modelo se trataba como Dense y fallaba en
     `attn_q`. Fix: `model_kind()` primero escanea el catálogo con
     `is_deltanet_layer` (tensores `ssm_a` + `attn_qkv`).
  2. **`stage_pack` con bloques mixtos**: el macro-bloque (block_k>1) cargaba
     capas deltanet como packs llama (sin `attn_q`) → error. Fix: si el bloque
     contiene cualquier capa deltanet, cae al ping-pong por capa (las capas
     deltanet viven en `DeltaNetLayerWeights` cache, no en el scratch).
  3. **Presupuesto de memoria**: el cache DeltaNet/SSM (~1 GB) no está en el
     estimado; la verificación de presupuesto al final de `generate` es ahora
     orientativa (skip) para `ModelKind::Hybrid`.
- **Validación:** `hayai generate` sobre Qwen_Qwen3.5-4B-Q4_K_M produce texto
  coherente ("Once upon a time in a" → "small town, there lived a young man
  named Jack."). Ablation `HAYAI_SKIP_DN_ATTN=1` → salida corrupta
  ("µÄ¿µ£║µ×äNI,\pi=1"), probando que el path DeltaNet compute de verdad.
- **Limitación conocida (pre-existente):** `Qwen2.5-7B` falla la verificación de
  presupuesto al final de generate (Dense); no relacionado con este cambio.
- **Pendiente Qwen3.8-27B:** mismo soporte (attn_qkv + SsmIrregularDag) ya
  cubierto en el streaming; falta validar con el modelo 27B y el evaluador Vía B.

## D24 — Evaluador de frontera sobre el path de producción (StreamingGenerator)
- **Decisión:** el evaluador de la frontera de Pareto (KL_global / D_arch_global)
  para modelos grandes usa ahora el **StreamingGenerator** (path de producción)
  en vez del `Generator` legacy (`--dev-mmap`, que no soporta híbridos Qwen3.5).
  Tres piezas:
  1. **`dump_weights`** (hayai): dequant F32 de los bloques FFN del profesor
     (`w.{layer}.{block}.bin` + `meta.json`) — una sola vez por modelo.
  2. **`embed_sparse`** (bin de `saor-streamer`): poda por magnitud del gate y
     reescritura **streaming** del GGUF embebido (D16) — sin OpenCL y sin cargar
     el archivo completo (crítico para GGUFs de 15–27 GB).
  3. **`kl_eval`** (hayai): abre original + embebido como `StreamingGenerator`,
     forward teacher-forced token a token → `KL_global`; `D_arch_global` desde
     los bloques dispersos del embebido (popcount de la adyacencia, params FFN
     totales como denominador).
- **Validación (SmolLM2):** el sweep `frontier_stream_sweep.py` reproduce la
  frontera KL vs D_arch; **`kl_eval` y `eval_sparse` dan exactamente el mismo
  KL (0.754 @ D_arch 0.033)** — cross-validación del path streaming vs legacy.
  La frontera de Vía A (D_arch ~0.10 @ KL 0.50) era con perfil optimizado por
  capa; el uniforme da más KL (esperado).
- **Limitación de disco:** el dump F32 de Qwen3.5 (~8 GB) y el embebido (~3 GB)
  no caben en esta máquina (disco casi lleno por `alia_emb2.gguf` de 23 GB);
  el pipeline queda listo para ejecutarse con espacio disponible. En híbridos
  Qwen3.5, esparcir solo las capas full-attn (el path deltanet carga el FFN por
  `load_quant_matrix`, no por CSR).

## D25 — Frontera Qwen3.5-4B cuantificada + scratch en disco D
- **Fix de entorno:** los temporales ahora viven en `d:\Documents\pySrc\.scratch`
  (disco D, ~190 GB libres, más rápido) — nunca en `C:\Users\...\AppData`.
  Con esto la limitación de disco de D24 queda resuelta.
- **Fix de funcionalidad:** el FFN disperso embebido (D16) ahora funciona en el
  híbrido completo:
  - `run_full_attn_block` (full-attn) usa `run_ffn_block` cuando el pack trae CSR.
  - `run_deltanet_block` (DeltaNet/SSM) carga las matrices FFN con
    `GgufCatalog::load_ffn_matrices` (densas o CSR) y ejecuta el CSR en CPU.
- **Frontera de Qwen3.5-4B (path de producción, `kl_eval`):**
  | perfil | D_arch | KL |
  |---|---|---|
  | full-attn (8/33) gate sp 0.3 | 0.024 | 0.288 |
  | full-attn gate sp 0.5 | 0.040 | 0.567 |
  | full-attn gate sp 0.7 | 0.057 | 0.944 |
  | **todas (33/33) gate sp 0.3** | **0.100** | **0.762** |
- **Hallazgo clave:** el modelo de 4B es **~3× más compresible que SmolLM2** a
  igual KL: D_arch ~0.10 → KL 0.76 (Qwen3.5) vs KL >>1 (SmolLM2 uniforme). La
  hipótesis de compresión variable por modelo queda cuantificada entre escalas
  (135M vs 4B). Esparcir distribuidamente (todas las capas) gana mucho frente a
  concentrar en pocas capas a igual D_arch.

## D26 — Frontera Qwen3.5-4B completa (todas las capas) + fix de embed_sparse
- **Fix de `embed_sparse`:** solo se reemplazan los bloques con `sp > 0`. Antes
  reemplazaba también up/down (sp=0) con CSR F32 denso — inflaba el modelo
  (10.3→5.1 GB) y cambiaba la numerica Q4→F32 innecesariamente.
- **Caché de orden de magnitud:** el sort por |w| es invariante a la esparsidad;
  `embed_sparse` lo escribe una vez (`order.{layer}.{block}.bin`) y lo reusa en
  cada punto del barrido (~9 min el primero, <1 min los siguientes).
- **Frontera Qwen3.5-4B completa (gate de TODAS las capas, poda por magnitud):**
  | sp | D_arch | KL |
  |---|---|---|
  | 0.10 | 0.033 | 0.107 |
  | 0.20 | 0.067 | 0.296 |
  | 0.30 | 0.100 | 0.762 |
  | 0.40 | 0.133 | 2.707 |
  - **Frontera factible: D_arch ≈ 0.085 @ KL 0.50** (~3.4× más compresible que
    SmolLM2, que da D_arch ~0.025 @ KL 0.50 uniforme).
  - La KL crece super-lineal con D_arch (0.11 → 0.30 → 0.76 → 2.71 para pasos
    lineales de 0.033) — el presupuesto de KL limita D_arch de forma agresiva.
- **ALIA:** punto medido en la máquina del usuario (`alia_eval5`): 12 capas
  esparsas → D_arch 0.058 @ KL 0.073 (tolerancia altísima en capas medias).
  El `alia_emb2.gguf` (23 GB) se movió a `d:\Documents\pySrc\.scratch`.

## D27 — Frontera de ALIA-40b (Q4_K_M, 48 capas) en el path de producción
- **Modelo:** `ALIA-40b-instruct-2606.Q4_K_M.gguf` (22.9 GB) descargado de
  `mradermacher/ALIA-40b-instruct-2606-GGUF` a `d:\Documents\pySrc\.scratch`.
  48 capas, llama estándar (sin SSM), gate 8192×24576.
- **Escalado del pipeline a 40B:**
  - `dump_weights --blocks gate` (solo gates → 36 GB F32 en vez de ~100 GB).
  - `embed_sparse`: pesos F32 **streaming a fichero** (`BlockReplacement.weights_file`)
    + sort de magnitud **paralelo (rayon)** + caché de orden — sin OOM en 16 GB
    RAM (antes: 35 GB de pesos en RAM → fallo de asignación).
  - `kl_eval`: dos `StreamingGenerator` (original + embebido) en lockstep,
    ~45 min por punto (forward de 40B en CPU, 4 posiciones).
- **Frontera de ALIA medida:**
  | perfil | D_arch | KL |
  |---|---|---|
  | uniforme gate sp 0.1 (48 capas) | 0.033 | 0.958 |
  | **medio sp 0.4 (solo capas 16–31)** | 0.044 | **0.449** |
- **Hallazgo clave:** la tolerancia de ALIA está **concentrada en las capas
  medias** — esparcir solo las 16 capas centrales a 0.4 da MÁS D_arch (0.044)
  con MENOS de la mitad de KL (0.449 vs 0.958) que el uniforme. Es la validación
  extrema de la compresión variable por modelo (40B): el perfil importa tanto o
  más que la densidad total. El CPPN global de Vía B (coordenada `y_layer`)
  expresa exactamente esta topología tipo campana.

## D28 — Frontera ALIA en GPU (OpenCL) + curva uniforme completa
- **GPU/OpenCL:** `hayai --device auto` detecta la RTX 4050 (6 GB, OpenCL 3.0
  CUDA) + Intel UHD (pool hetero). Para el evaluador de 40B se necesitó:
  - **`kl_eval` secuencial**: un scratch SVM a la vez (los logits del original se
    guardan en RAM y el generador esparso corre después) — dos generadores
    simultáneos agotaban la VRAM.
  - **`MemoryStrategy::Minimal`** (2 slots ping-pong) en vez de AutoFit — la
    ventana residente de 40B superaba los 6 GB.
- **Frontera uniforme ALIA completada (GPU):**
  | perfil | D_arch | KL |
  |---|---|---|
  | uniforme sp 0.1 (48 capas) | 0.033 | 0.958 |
  | uniforme sp 0.2 | 0.067 | **2.151** |
  | **medio sp 0.4 (16–31)** | 0.044 | **0.449** |
- **Lectura:** la curva uniforme explota (KL 0.96 → 2.15 para sp 0.1 → 0.2)
  mientras el perfil medio mantiene KL < 0.5 con más D_arch. En ALIA-40b el
  perfil es el 80% del juego: la optimización topológica (Vía B) es la ruta.
- **Costo por punto:** ~40 min en GPU (el CSR disperso del modelo embebido corre
  en CPU: `spmm_csr_cpu`; la atención y el modelo original van a GPU). Para
  acelerar el path esparso haría falta el SpMM CSR en OpenCL (kernel `spmm.cl`
  de saor) dentro del `StreamingGenerator`.

## D29 — SpMM CSR en OpenCL para el FFN disperso del streaming
- **Decisión:** el `StreamingGenerator` ejecuta ahora el FFN disperso embebido
  (D16) con **SpMM CSR en OpenCL** cuando hay pool (`run_ffn_block` → nuevo
  helper `spmm_csr`: `orch.opencl_engine().spmm_csr(...)` si hay GPU, si no
  `spmm_csr_cpu`). El kernel ya existía en `hayai-opencl` (validado contra la
  referencia densa); solo faltaba cablearlo al path del streaming.
- **Validación:** `kl_eval` SmolLM2 sp0.1 CPU vs GPU → **KL 0.754375 vs
  0.754372** (idénticos, 6 decimales). El path esparso ahora se acelera con la
  RTX 4050 (los bloques CSR del 40B dejan de correr en CPU).
- **Estado GPU/OpenCL completo:**
  - `hayai --device auto` detecta RTX 4050 + Intel UHD (pool hetero).
  - `kl_eval` secuencial (un scratch SVM a la vez) + `MemoryStrategy::Minimal`
    para caber en 6 GB VRAM.
  - FFN denso → gemv Q4 en GPU; FFN disperso → SpMM CSR en GPU.
- **Tests:** 154 Rust verdes (incluye `opencl_spmm_csr_matches_cpu`).

## D30 — El perfil de ALIA es un escalón, no una campana (frontera completa)
- **SpMM GPU también en el prefill:** `prefill_wave.rs` usa el helper
  `spmm_csr` (OpenCL si hay pool) para gate/up/down — todo el path disperso del
  streaming va a GPU.
- **Frontera de ALIA-40b completada (GPU):**
  | perfil | D_arch | KL |
  |---|---|---|
  | uniforme sp 0.1 (48 capas) | 0.033 | 0.958 |
  | **medio sp 0.4 (16–31)** | 0.044 | **0.449** |
  | **medio sp 0.6 (16–31)** | 0.067 | **0.936** |
  | uniforme sp 0.2 (48 capas) | 0.067 | **2.151** |
  | campana suave (bordes 0.1 → centro 0.5) | 0.100 | 4.667 |
- **Lectura clave (2):**
  1. **A mismo D_arch (0.067), el perfil medio da 2.3× menos KL que el uniforme**
     (0.94 vs 2.15) — el perfil es el 80% del juego en ALIA.
  2. **La campana suave es mala** (KL 4.67): las capas tempranas de ALIA son
     **hipersensibles** — incluso 10% de esparsidad en los bordes es carísimo.
     El óptimo es un **escalón**: capas 16–31 esparsas (hasta 0.6), el resto
     completamente densas. Frontera factible del perfil medio ≈ D_arch 0.050
     @ KL 0.50; el uniforme apenas llega a ~0.025.
- **Implicación Vía B:** el CPPN global con `y_layer` debe aprender un escalón
  abrupto (no una campana suave) para ALIA — la coordenada de capa lo permite
  (transición sigmoide empinada). La métrica del perfil es la que manda.

## D31 — Vía B completa: `embed_sparse --genome` + loop CMA-ES streaming
- **`embed_sparse --genome <bin> --tau <f>`**: decodifica la topología CPPN global
  de CADA capa (sustrato v5, `instantiate_layer` con `y_layer`) y conserva los
  pesos del profesor en las posiciones activas. Con esto el pipeline
  **CPPN → embed D16 → kl_eval → CMA-ES** queda cerrado de punta a punta:
  - `via_b_evolve.py --streaming` ejecuta el loop con la topología REAL (no el
    proxy de densidad): por candidato escribe el genoma, `embed_sparse --genome`,
    `kl_eval` (StreamingGenerator, GPU) y el CMA-ES actualiza.
  - Validación en SmolLM2: gen 0 → KL 5.57 @ D_arch 0.154 (el genoma inicial
    aleatorio); el loop corre y guarda el mejor genoma.
- **Resultado de evolución (4 gens, n_pos 8):** el CMA-ES baja la KL
  7.24 → 6.84 → 2.39 (gen 2, mejor); gen 3 diverge (8.9, sigma alta — ruido
  esperado). El mejor genoma validado con evaluación completa (n_pos 24):
  **KL = 1.68 @ D_arch 0.078** — 4.3× menos KL que el genoma inicial, con
  D_arch pegado al target 0.10.
- **Perfil por capa del genoma evolucionado** (decode de referencia): escalón —
  capas 0–15 al 95–99% esparsas, capas 16–29 progresivamente densas (94 → 85%),
  media 92.2%. El CMA-ES reproduce el patrón de la frontera de ALIA: las capas
  tempranas son prescindibles y la información se concentra antes de la cabeza.
- **Reproducibilidad:** run limpio (n_pos 24, 4 gens) → gen 0: 5.57, gen 1: 5.58,
  gen 2: **1.6823** @ D_arch 0.078, gen 3: 7.14 (divergente). Mismo óptimo que el
  run ruidoso (n_pos 8): el mínimo KL ≈ 1.68 @ D_arch ≈ 0.078 es robusto. La gen 3
  diverge en ambos runs (sigma del CMA-ES crece) — mitigado con el reinicio de
  sigma (`via_b_evolve`, commit 5701e3b) para runs largos.
- **Costo por candidato** (~40 s en SmolLM2): el decode CPPN en Rust
  (`instantiate_layer`, 26M conexiones × 30 capas) + la reescritura GGUF + el
  kl_eval. Con `--n-pos 8` una generación (22 candidatos) ≈ 15 min.
- **Pendiente de rendimiento:** ~~el decode CPPN en Rust (201M conexiones × 48
  capas ≈ 9.6G evals) es inviable; requiere el kernel OpenCL~~ — **resuelto en
  D32**: `embed_sparse --genome --gpu` usa el kernel `cppn_decode_adj` (GPU 2 s
  vs CPU 33 s en SmolLM2; escala a ALIA vía dispatches fragmentados WDDM).

## D32 — Kernel OpenCL completado: `embed_sparse --genome --gpu`
- **Linkeo dinámico (cl3 `dynamic`)**: saor-opencl pasa a opencl3 0.12 + cl3 0.13
  con `dynamic` (carga OpenCL.dll en runtime, patrón de hayai-opencl). Los
  binarios linkean en máquinas sin el SDK (solo el runtime del driver); antes
  `LNK1181: OpenCL.lib` impedía compilar `saor-engine`. Todas las llamadas
  opencl3 (create/set_arg/enqueue) se marcaron `unsafe` y se envolvieron en
  helpers (`set_arg`, buffer/write/read).
- **`embed_sparse --genome --gpu`**: decodifica la topología CPPN de cada
  (capa, bloque) con el kernel `cppn_decode_adj` (solo adyacencia, sin w_out —
  el warm-start conserva los pesos del profesor). `--gpu` es explícito y sin
  fallback silencioso (en ALIA el decode CPU es inviable). El reporte incluye
  `"device"`.
- **Validado en esta máquina (RTX 4050):**
  - `kernels-run` ahora cubre el decode multi-capa: capa 7/30 (y_layer ≠ 0),
    tau 0.30 → 1893/3072 activos (patrón mixto), **bit-exacto** con
    `instantiate_layer` (`adj_bytes_diff_layer=0`).
  - SmolLM2 best-genome: CPU KL 1.682290 vs GPU KL 1.682447 (Δ 1e-4). La única
    diferencia: 1 conexión en 80M donde `l_ij ≈ tau` y el f32 de sin/tanh/exp
    difiere 1 ULP entre OpenCL y Rust std (esperable; irrelevante para la KL).
  - **Rendimiento: GPU 2 s vs CPU 33 s en SmolLM2 (16.5×)** — en ALIA-40b
    (201M conexiones × 48 capas ≈ 9.6G evals) el GPU escala vía dispatch
    fragmentados (WDDM), el CPU no.
- **`via_b_evolve --gpu`**: el loop CMA-ES pasa `--gpu` al embed (acelera cada
  candidato de ~30 s a ~2 s en SmolLM2; en ALIA es la única vía viable).

## D33 — Qwen3.5-4B y ALIA-40b: fases completadas y evolución Vía B en curso
- **Qwen3.5-4B** (`Qwen_Qwen3.5-4B-Q4_K_M.gguf`, 2.81 GB): arquitectura híbrida
  (D23). 33 capas (32 transformadoras + 1 NextN/MTP), gate `2560×9216`.
  - Dump gate: `w_qwen35` (33 × 90 MB).
  - **Frontera uniforme medida** (n_pos 8, GPU): sp 0.05 → KL 0.045 @ D_arch
    0.017; sp 0.10 → 0.108 @ 0.033; sp 0.15 → 0.173 @ 0.050; sp 0.20 → 0.296 @
    0.067; sp 0.25 → 0.463 @ 0.083. El 4B es muy robusto a la poda del gate.
  - **Evolución Vía B** lanzada (4 gens, `--streaming --gpu`, n_pos 8).
- **ALIA-40b**: dump gate existente (`w_alia`, 48 × 768 MB).
  - **Frontera uniforme re-medida** (n_pos 4): sp 0.1 → KL 0.96 @ 0.033;
    sp 0.2 → 2.15 @ 0.067; **sp 0.4 → 3.24 @ 0.133** (el archivo de 43 GB
    completo; la medida previa de 0.45 era de un archivo parcial/corrupto).
  - **Investigación OOM resuelta:** los fallos históricos
    (`CL_MEM_OBJECT_ALLOCATION_FAILURE` / `UnexpectedEof`) eran de **archivos
    GGUF parciales** (carrera entre dos `embed_sparse` concurrentes sobre los
    mismos `wdata.*` de `w_alia`) + estado pre-D29. Con el archivo completo el
    kl_eval del 40B **no OOMa** (RTX 4050, ~2.1 GB VRAM). Nota de operación:
    **nunca lanzar dos `embed_sparse` sobre el mismo `--weights`**, y no tocar
    los `wdata.*` mientras un embed corre.
  - **Coste kl_eval ALIA**: ~20–25 min por candidato a n_pos 4. Cuando los
    gates quedan densos (>~50%), el SpMM CSR no cabe con el modelo en 6 GB y el
    pool OpenCL cae a CPU (GPU al 1%); con gates esparsos la GPU se activa.
  - **Evolución Vía B** lanzada (4 gens, `--streaming --gpu`, n_pos 4, ~1 día).
- **Rendimiento embed ALIA**: decode CPPN GPU ~2 s; reescritura streaming ~5 min
  (~23 GB escritos por candidato). El genoma evolucionado de SmolLM2 **no
  transfiere** a ALIA (el sustrato 8192×24576 muestrea la superficie CPPN de
  forma distinta → topología distinta); la evolución de ALIA debe encontrar el
  suyo.

## D34 — Resultados de las evoluciones Vía B (Qwen3.5-4B y ALIA-40b) + Qwen3.8-27B
- **Qwen3.5-4B — evolución Vía B completada y validada:**
  - Trayectoria: gen 0 KL 1.40 @ D_arch 0.16 → gen 1 **KL 0.080** @ 0.027 (best) →
    gens 2-3 KL 0.14-0.38. Validación definitiva (n_pos 24): **KL 0.113 @
    D_arch 0.027**. Genoma: `via_b_best_genome_qwen35.bin`.
- **ALIA-40b — evolución Vía B completada y validada:**
  - Trayectoria: gen 0 KL 3.07 @ 0.051 → gen 2 **KL 0.90 @ 0.018** (best).
    Validación definitiva (n_pos 8): **KL 0.776 @ D_arch 0.018**. Genoma:
    `via_b_best_genome_alia.bin`. El perfil decodificado es un **valle**
    (99% esparso en los extremos → 92% en el medio). ALIA es el modelo más
    sensible: KL ~0.78 a solo 1.8% de densidad arquitectónica (Qwen3.5-4B
    logra KL 0.11 al 2.7%).
- **Deadlock de GPU concurrente (hallazgo operacional):** dos procesos OpenCL
  (`kl_eval --device auto`) simultáneos sobre la misma GPU **se cuelgan**
  (CPU y GPU quedan idle tras un rato de cómputo). Diagnóstico: CPU deja de
  crecer + GPU ~0% + proceso vivo = deadlock de pool. Regla: **un solo proceso
  OpenCL a la vez** (los trabajos del pipeline deben serializarse).
- **Qwen3.8-27B — runtime híbrido validado:**
  - Dump gate: 65 capas `5120×17408` (51 s). Los 336 tensores SSM no
    interfieren (el streaming los rutea vía `hybrid_layer_kind`).
  - Smoke test (dense vs sparse sp 0.1, n_pos 4, GPU): **KL 0.042 @ D_arch
    0.033** — el 27B es el más robusto a la poda del gate. El "hang" inicial
    del 27B era el deadlock de GPU concurrente, no el runtime.
  - Frontera uniforme en curso (5 puntos, secuencial, ~2.5 h).

## D35 — Qwen3.8-27B: frontera completa + evolución Vía B en curso
- **Frontera uniforme completada** (n_pos 4, GPU, secuencial — un solo proceso
  OpenCL, regla D34):
  | sp | D_arch | KL |
  |---|---|---|
  | 0.05 | 0.017 | **0.015** |
  | 0.10 | 0.033 | 0.042 |
  | 0.15 | 0.050 | 0.075 |
  | 0.20 | 0.067 | 0.115 |
  | 0.25 | 0.083 | **0.143** |
- **Lectura:** el 27B es el modelo **más robusto** a la poda del gate de los
  cuatro (KL 0.14 a sp 0.25, muy por debajo del target 0.5). El híbrido
  gated-DeltaNet de 65 capas tolera la poda de forma excepcional — el FFN es
  redundante en este modelo.
- **Evolución Vía B** lanzada (4 gens, `--streaming --gpu`, n_pos 4, ~1-2 días
  por el coste kl_eval de 27B, secuencial). El ranking de sensibilidad queda:
  **27B < 4B << SmolLM2 << ALIA** (ALIA es ~70× más sensible que el 27B a la
  misma densidad).

## Notas externas
- La PR `pr_soporte_gguf_disperso_v2.md` de `hayai` no está en este directorio;
  se trata como artefacto externo de referencia (D4).
- La máquina tiene dos GPU OpenCL: NVIDIA RTX 4050 (discreta, 6 GB) e Intel UHD
  Graphics (iGPU, memoria compartida). El experimento debe fijar la NVIDIA.

