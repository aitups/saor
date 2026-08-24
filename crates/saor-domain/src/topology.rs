//! Instanciación de topologías: DAG irregular + máscara de esparsidad dinámica.
//!
//! La CPPN genera pares `(w_ij, l_ij)`; el umbral `τ ∈ (0,1)` decide qué
//! conexiones sobreviven (`A_ij = 1 si l_ij > τ`). La matriz de adyacencia se
//! empaqueta como bits para el GGUF disperso y para `D_arch` (popcount).

use crate::cppn::CppnGenome;

/// Matriz de adyacencia + pesos del DAG instanciado.
#[derive(Debug, Clone)]
pub struct Topology {
    /// Adyacencia empaquetada en bits, fila por fila (LSB-first).
    pub adjacency_bits: Vec<u8>,
    /// Número total de conexiones posibles `d_in * d_out`.
    pub total_connections: usize,
    /// Pesos `w_ij` en orden `(i, j)` — solo conexiones activas.
    pub weights: Vec<f32>,
}

impl Topology {
    /// Cuenta de conexiones activas por popcount.
    pub fn active_connections(&self) -> usize {
        self.adjacency_bits
            .iter()
            .map(|b| b.count_ones() as usize)
            .sum::<usize>()
            .min(self.total_connections)
    }

    /// Esparcidad del candidato = `D_arch(A0, A1)`.
    pub fn sparsity(&self) -> f32 {
        debug_assert!(self.total_connections > 0);
        1.0 - (self.active_connections() as f32 / self.total_connections as f32)
    }
}

/// Instancia la topología evaluando la CPPN para todo el sustrato `d_in x d_out`.
///
/// * `genome` — genoma CPPN.
/// * `d_in`, `d_out` — dimensiones del bloque (con reconciliación previa).
/// * `tau` — umbral de esparsidad dinámico (evolucionado por CMA-ES).
pub fn instantiate(genome: &CppnGenome, d_in: usize, d_out: usize, tau: f32) -> Topology {
    let total = d_in * d_out;
    let bits_len = total.div_ceil(8);
    let mut adjacency_bits = vec![0u8; bits_len];
    let mut weights = Vec::with_capacity(total);

    for i in 0..d_in {
        for j in 0..d_out {
            let v = crate::cppn::input_vector(d_in, d_out, i, j);
            let (w, l) = genome.evaluate(&v);
            let idx = i * d_out + j;
            if l > tau {
                adjacency_bits[idx / 8] |= 1 << (idx % 8);
                weights.push(w);
            }
        }
    }

    Topology {
        adjacency_bits,
        total_connections: total,
        weights,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tau_extremo_vacia_o_llena_la_topologia() {
        let genome = CppnGenome::zeros(); // l = 0.5 siempre
        // tau = 0.0 -> 0.5 > 0 -> todo activo
        let dense = instantiate(&genome, 4, 4, 0.0);
        assert_eq!(dense.active_connections(), 16);
        assert!((dense.sparsity() - 0.0).abs() < 1e-6);
        // tau = 1.0 -> 0.5 > 1 -> falso -> todo inactivo
        let empty = instantiate(&genome, 4, 4, 1.0);
        assert_eq!(empty.active_connections(), 0);
        assert!((empty.sparsity() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sparsity_por_encima_del_umbral_de_contrato() {
        let genome = CppnGenome::zeros();
        // Con l = 0.5 fijo, tau justo por encima deja todo inactivo -> dist = 1.
        let t = instantiate(&genome, 16, 16, 0.51);
        assert!(t.sparsity() >= 0.4, "D_arch debe superar el contrato de 0.4");
    }
}
