//! # saor-opencl
//!
//! Cómputo pesado en VRAM vía OpenCL 3.0 (crate `opencl3`). Aisla el I/O del
//! cómputo puro: `context` descubre el dispositivo NVIDIA y los kernels (Fase 3)
//! implementan el decodificador CPPN, el SpMM del DAG y la matriz de Gram/CKA.

#![deny(missing_docs)]

pub mod compute;
pub mod context;

/// Re-export del crate subyacente para conveniencia.
pub use opencl3;
