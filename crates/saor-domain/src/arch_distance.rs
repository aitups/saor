//! Distancia arquitectónica: Hamming normalizada sobre la matriz de adyacencia.
//!
//! Como el bloque original es denso (`A0 = 1` para todo par), la distancia se
//! reduce a la esparcidad del candidato:
//!
//! ```text
//! D_arch(A0, A1) = 1 - sum(A1) / (d_in * d_out) = Sparsity(A1)
//! ```
//!
//! El tensor de adyacencia se empaqueta como bits (tensor `ffn_dag_adjacency`
//! del GGUF disperso), por lo que el cálculo en host es un `popcount` sobre el
//! búfer — costo `O(n/8)`, fracciones de nanosegundo.

/// Calcula `D_arch` sobre un tensor de adyacencia empaquetado en bits (LSB-first).
///
/// * `adjacency_bits` — bytes que contienen la matriz `A1` en filas, con el bit
///   `k` de cada byte representando la conexión `(i, j)` correspondiente
///   (`bit_k = 1` → conexión activa).
/// * `total_connections` — `d_in * d_out` del bloque.
///
/// Devuelve `1 - activas/total` (es decir, la esparcidad del candidato) en `[0, 1]`.
pub fn hamming_sparsity(adjacency_bits: &[u8], total_connections: usize) -> f32 {
    debug_assert!(total_connections > 0, "total_connections debe ser > 0");
    let active: usize = adjacency_bits
        .iter()
        .map(|byte| byte.count_ones() as usize)
        .sum::<usize>()
        .min(total_connections);

    1.0 - (active as f32 / total_connections as f32)
}

/// Cuenta las conexiones activas del candidato mediante `popcount`.
pub fn active_connections(adjacency_bits: &[u8]) -> usize {
    adjacency_bits
        .iter()
        .map(|byte| byte.count_ones() as usize)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bloque_totalmente_denso_tiene_distancia_cero() {
        // 4 conexiones, todas activas (byte 0b0000_1111).
        let bits = [0b0000_1111u8];
        assert_eq!(hamming_sparsity(&bits, 4), 0.0);
    }

    #[test]
    fn bloque_vacio_tiene_distancia_uno() {
        let bits = [0u8; 2];
        assert_eq!(hamming_sparsity(&bits, 16), 1.0);
    }

    #[test]
    fn sparsity_del_40_por_ciento_supera_umbral_contrato() {
        // 16 conexiones, 6 activas (0b0101_0101 -> 4 ones + 0b0000_0011 -> 2 ones)
        let bits = [0b0101_0101u8, 0b0000_0011u8];
        let d = hamming_sparsity(&bits, 16);
        assert!((d - 0.625).abs() < 1e-6);
        assert!(d >= 0.4, "el contrato exige D_arch >= 0.4");
    }

    #[test]
    fn active_connections_cuenta_bits() {
        let bits = [0b1111_0000u8, 0b0000_0001u8];
        assert_eq!(active_connections(&bits), 5);
    }
}
