//! Reconciliación dimensional en topologías no dirigidas (sección 5 de v4).
//!
//! Conecta sub-grafos con dimensiones dispares (`d_A != d_B`) sin penalizar
//! VRAM:
//! 1. `d_A > d_B` → subsampling por índices calientes (top-`d_B` canales por
//!    varianza de activación del profesor `H0`), `O(1)` por canal.
//! 2. `d_A < d_B` → proyección lineal adaptativa `W_proj` que emula identidad.

use nalgebra::DVector;

/// Selecciona los `k` índices de mayor varianza de activación (canales calientes).
///
/// `activation_variance` — diagonal de la Fisher empírica (`Var(H0)`).
pub fn hot_indices(activation_variance: &DVector<f32>, k: usize) -> Vec<usize> {
    debug_assert!(k <= activation_variance.len());
    let mut idx: Vec<usize> = (0..activation_variance.len()).collect();
    idx.sort_by(|&a, &b| {
        activation_variance[b]
            .total_cmp(&activation_variance[a])
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx
}

/// Construye la proyección adaptativa `W_proj ∈ R^{d_A x d_B}` que emula
/// identidad: `W_proj[i, map(i)] = 1` si el canal `i` fue seleccionado.
pub fn identity_projection(d_in: usize, d_out: usize, selected: &[usize]) -> nalgebra::DMatrix<f32> {
    debug_assert!(selected.len() <= d_out.min(d_in));
    let mut proj = nalgebra::DMatrix::<f32>::zeros(d_in, d_out);
    for (col, &row) in selected.iter().enumerate() {
        proj[(row, col)] = 1.0;
    }
    proj
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hot_indices_ordena_por_varianza() {
        let var = DVector::from_vec(vec![1.0, 9.0, 4.0, 16.0]);
        let idx = hot_indices(&var, 2);
        assert_eq!(idx, vec![3, 1]); // 16 > 9 > 4 > 1
    }

    #[test]
    fn proyeccion_identidad_preserva_canales() {
        // d_in = 4, d_out = 2: los canales calientes {3, 1} se conservan.
        let p = identity_projection(4, 2, &[3, 1]);
        assert_eq!(p[(3, 0)], 1.0);
        assert_eq!(p[(1, 1)], 1.0);
        assert_eq!(p[(0, 0)], 0.0);
    }
}
