//! # saor-streamer
//!
//! Motor de streaming capa a capa (Paso 3 de la propuesta):
//!
//! * `pinned` — `PinnedMemoryAllocator`: RAM page-locked para mapeo PCIe directo.
//! * `quant` — cuantización 4-bit por bloques de 32 con escala (escalón previo
//!   a los esquemas exactos IQ4/Q4_K de GGML, que se integran con hayai).
//! * `streamer` — `RustStreamer`: doble buffer que precarga la capa `l+1`
//!   mientras la GPU computa la capa `l`, con presupuesto de pico de VRAM.
//! * `gguf_sparse` — almacenamiento **GGUF disperso** (pesos activos +
//!   bit-tensor `ffn_dag_adjacency`), sin densificar, alineado con la PR
//!   `pr_soporte_gguf_disperso_v2.md` de hayai.

#![deny(missing_docs)]

pub mod gguf_sparse;
pub mod pinned;
pub mod quant;
pub mod streamer;

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

