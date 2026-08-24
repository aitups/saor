//! Cuantización 4-bit por bloques de 32 con escala por bloque.
//!
//! Formato propio de streaming (escalón previo a los esquemas exactos
//! IQ4/Q4_K de GGML, que se integrarán con `hayai` en la PR de GGUF disperso).
//! Simétrico: `q = clamp(round(v/scale), -8, 7)` con `scale = max_abs/7`.
//! Cada bloque ocupa `1 f32 + 16 bytes = 20 bytes` por 32 valores.

use serde::{Deserialize, Serialize};

/// Valores por bloque cuantizado.
pub const BLOCK_SIZE: usize = 32;
/// Bytes de nibbles por bloque (4 bits por valor).
pub const NIBBLES_PER_BLOCK: usize = BLOCK_SIZE / 2;

/// Un bloque cuantizado de 32 valores.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantizedBlock {
    /// Escala global del bloque (`max_abs / 7`).
    pub scale: f32,
    /// Nibbles empaquetados (16 bytes): valor `i` en el nibble bajo/alto.
    pub nibbles: [u8; NIBBLES_PER_BLOCK],
}

impl QuantizedBlock {
    /// Bloque vacío (escala 0, sin información).
    pub fn empty() -> Self {
        Self {
            scale: 0.0,
            nibbles: [0; NIBBLES_PER_BLOCK],
        }
    }

    /// Bytes ocupados por un bloque.
    pub fn byte_size() -> usize {
        std::mem::size_of::<f32>() + NIBBLES_PER_BLOCK
    }
}

/// Convierte un nibble con signo en complemento a 2 (rango `-8..=7`).
pub fn signed_nibble(nib: u8) -> i8 {
    let v = nib & 0x0F;
    if v >= 8 {
        v as i8 - 16
    } else {
        v as i8
    }
}

/// Cuantiza un bloque de exactamente [`BLOCK_SIZE`] valores.
pub fn quantize_block(values: &[f32]) -> QuantizedBlock {
    debug_assert_eq!(values.len(), BLOCK_SIZE);
    let max_abs = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if max_abs <= f32::EPSILON {
        return QuantizedBlock::empty();
    }
    let scale = max_abs / 7.0;
    let mut nibbles = [0u8; NIBBLES_PER_BLOCK];
    for i in 0..BLOCK_SIZE {
        let q = (values[i] / scale).round().clamp(-8.0, 7.0) as i8;
        let nib = (q & 0x0F) as u8;
        if i % 2 == 0 {
            nibbles[i / 2] = nib;
        } else {
            nibbles[i / 2] |= nib << 4;
        }
    }
    QuantizedBlock { scale, nibbles }
}

/// Descuantiza un bloque a sus 32 valores originales.
pub fn dequantize_block(block: &QuantizedBlock) -> [f32; BLOCK_SIZE] {
    let mut out = [0.0f32; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
        let byte = block.nibbles[i / 2];
        let nib = if i % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
        out[i] = signed_nibble(nib) as f32 * block.scale;
    }
    out
}

/// Cuantiza una rebanada (longitud múltiplo de [`BLOCK_SIZE`]).
pub fn quantize_slice(values: &[f32]) -> Vec<QuantizedBlock> {
    assert_eq!(
        values.len() % BLOCK_SIZE,
        0,
        "el tamaño debe ser múltiplo de BLOCK_SIZE"
    );
    values.chunks_exact(BLOCK_SIZE).map(quantize_block).collect()
}

/// Descuantiza una secuencia de bloques.
pub fn dequantize_slice(blocks: &[QuantizedBlock]) -> Vec<f32> {
    let mut out = Vec::with_capacity(blocks.len() * BLOCK_SIZE);
    for b in blocks {
        out.extend_from_slice(&dequantize_block(b));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_erro_acotado() {
        let values: Vec<f32> = (0..BLOCK_SIZE).map(|i| (i as f32) * 0.3 - 4.0).collect();
        let q = quantize_block(&values);
        let dq = dequantize_block(&q);
        let max_err = values
            .iter()
            .zip(dq.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // Error <= escala/2 = max_abs/14.
        assert!(max_err <= q.scale / 2.0 + 1e-6, "max_err={max_err}, scale={}", q.scale);
    }

    #[test]
    fn cero_cuantiza_a_bloque_vacio() {
        let q = quantize_block(&[0.0; BLOCK_SIZE]);
        assert_eq!(q.scale, 0.0);
        let dq = dequantize_block(&q);
        assert!(dq.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn slices_round_trip() {
        let values: Vec<f32> = (0..BLOCK_SIZE * 3).map(|i| (i as f32).sin()).collect();
        let blocks = quantize_slice(&values);
        let dq = dequantize_slice(&blocks);
        let max_err = values
            .iter()
            .zip(dq.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.5);
    }
}
