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

## Notas externas
- La PR `pr_soporte_gguf_disperso_v2.md` de `hayai` no está en este directorio;
  se trata como artefacto externo de referencia (D4).
- La máquina tiene dos GPU OpenCL: NVIDIA RTX 4050 (discreta, 6 GB) e Intel UHD
  Graphics (iGPU, memoria compartida). El experimento debe fijar la NVIDIA.

