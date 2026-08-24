//! # saor-domain
//!
//! Matemática pura del experimento de optimización no dirigida. Sin I/O, sin
//! dependencias nativas: todo el álgebra es Rust puro (`nalgebra` con la
//! característica `matrixmultiply`).
//!
//! Módulos (la matemática de referencia se valida primero en NumPy — Fase 1 —
//! y luego se porta aquí para el motor nativo):
//!
//! * `cppn`   — red de patrones de composición (genoma indirecto de ~32K params).
//! * `topology` — instanciación del DAG irregular + máscara de esparsidad τ.
//! * `cka`    — fitness CKA (matrices de Gram + HSIC) entre bloque y profesor.
//! * `cmaes`  — CMA-ES en subespacio activo con reconstrucción sin estado.
//! * `reconciler` — reconciliación dimensional (índices calientes / proyección).
//! * `arch_distance` — distancia arquitectónica (Hamming normalizada / sparsity).

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod arch_distance;
pub mod cka;
pub mod cmaes;
pub mod cppn;
pub mod reconciler;
pub mod topology;

/// Versión del dominio matemático.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
