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



## Notas externas
- La PR `pr_soporte_gguf_disperso_v2.md` de `hayai` no está en este directorio;
  se trata como artefacto externo de referencia (D4).
- La máquina tiene dos GPU OpenCL: NVIDIA RTX 4050 (discreta, 6 GB) e Intel UHD
  Graphics (iGPU, memoria compartida). El experimento debe fijar la NVIDIA.

