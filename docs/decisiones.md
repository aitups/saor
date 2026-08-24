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
- **Estado:** a confirmar al llegar a la Fase 5.

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

## Notas externas
- La PR `pr_soporte_gguf_disperso_v2.md` de `hayai` no está en este directorio;
  se trata como artefacto externo de referencia (D4).
- La máquina tiene dos GPU OpenCL: NVIDIA RTX 4050 (discreta, 6 GB) e Intel UHD
  Graphics (iGPU, memoria compartida). El experimento debe fijar la NVIDIA.

