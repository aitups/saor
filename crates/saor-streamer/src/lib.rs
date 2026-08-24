//! # saor-streamer
//!
//! Motor de streaming capa a capa (Paso 3 de la propuesta):
//!
//! * `PinnedMemoryAllocator` — reserva contigua de RAM para mapeo PCIe directo.
//! * `RustStreamer` — doble buffer: precarga la capa `l+1` mientras la GPU
//!   computa la capa `l`, manteniendo un pico de VRAM de ~2 GB.
//! * Cuantización 4-bit (IQ4/Q4_K) y almacenamiento **GGUF disperso** (pesos
//!   activos + bit-tensor `ffn_dag_adjacency`), sin densificar.
//!
//! El desarrollo detallado de esta crate es la Fase 2; aquí queda el esqueleto.

#![deny(missing_docs)]

/// Presupuesto objetivo de pico de VRAM del motor de streaming (~2 GB).
pub const VRAM_PEAK_BUDGET_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Nombre del tensor de adyacencia en el GGUF disperso (alineado con
/// `pr_soporte_gguf_disperso_v2.md` de hayai).
pub const ADJACENCY_TENSOR_NAME: &str = "ffn_dag_adjacency";

/// Versión del motor de streaming.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presupuesto_de_vram_es_de_2gb() {
        assert_eq!(VRAM_PEAK_BUDGET_BYTES, 2 * 1024 * 1024 * 1024);
    }
}
