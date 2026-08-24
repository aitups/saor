# saor — Optimización No Dirigida de LLMs

Experimento de **optimización no dirigida** sobre un LLM de 25B–40B bajo hardware
de consumo restrictivo (RTX 4050 ~6 GB VRAM, 15 GB RAM, Windows). La base
conceptual está en [`DiseñoExperimento.md`](DiseñoExperimento.md) y la
arquitectura en
[`Propuesta_Arquitectura_Optimizacion_No_Dirigida_v4.md`](Propuesta_Arquitectura_Optimizacion_No_Dirigida_v4.md).

**Idea central:** un genoma indirecto (CPPN de ~32K parámetros) genera el DAG de
un bloque del Transformer; el fitness es CKA contra activaciones precómputadas
del profesor; la evolución es CMA-ES en un subespacio activo (d≈100–500). El
modelo candidato se almacena en **GGUF disperso** (sin densificar) y el cómputo
pesado corre en OpenCL 3.0 sobre la GPU.

## Estructura

```
crates/saor-domain      Matemática pura (Rust): CPPN, topología DAG, CKA,
                        CMA-ES, reconciliación dimensional, distancia arquitectónica
crates/saor-opencl      OpenCL 3.0: descubrimiento de dispositivo (context),
                        buffers y kernels (Fase 3)
crates/saor-streamer    Motor de streaming capa a capa, memoria pinned,
                        cuantización 4-bit, GGUF disperso (Fase 2)
engine/                 Binario saor-engine (CLI/IPC JSON-lines)
python/                 Orquestador Python: referencia NumPy (Fase 1) + hooks (Fase 5)
docs/decisiones.md      Registro de decisiones de diseño
scripts/                build_dev.bat (entorno MSVC), generate_opencl_lib.bat
```

## Requisitos

- Windows + **MSVC Build Tools 2022** (carga "Desarrollo de escritorio con C++")
  y **CMake** (instalados en Fase 0).
- Rust toolchain `stable-x86_64-pc-windows-msvc` (fijado por `rust-toolchain.toml`).
- Python 3.10+ (venv en `python/.venv`).
- Driver NVIDIA con OpenCL (la RTX 4050 reporta OpenCL C 1.2 y
  `CL_DEVICE_MAX_MEM_ALLOC_SIZE` = 1.5 GiB — ver `docs/decisiones.md` D9).

## Build y tests

```bat
:: Compilar todo el workspace
scripts\build_dev.bat build --workspace

:: Tests de Rust (18 en Fase 0)
scripts\build_dev.bat test --workspace

:: Info del dispositivo OpenCL (RTX 4050)
target\debug\saor-engine.exe device-info

:: Tests de Python
cd python
.venv\Scripts\python.exe -m pytest tests/ -v
```

> `build_dev.bat` configura `vcvars64` y añade el import lib local de OpenCL
> (`scripts\generate_opencl_lib.bat`, decisión D10).

## Roadmap

| Fase | Estado |
|---|---|
| 0 — Fundación y toolchain | ✅ completada (workspace, MSVC/CMake, OpenCL.lib, `device-info`, smoke tests) |
| 1 — Referencia NumPy | ✅ completada (cppn, topology, cka, cmaes, reconciler, arch_distance + 14 tests) |
| 2 — Núcleo Rust (streaming/memoria) | ⏳ siguiente |
| 3 — Kernels OpenCL 3.0 | ⏳ |
| 4 — Loop evolutivo integrado | ⏳ |
| 5 — Hooks del modelo real | ⏳ |
| 6 — Cierre (contrato de Fase 2) | ⏳ |
