//! Fitness CKA (Centered Kernel Alignment): equivalencia funcional local.
//!
//! Compara la huella semántica del bloque candidato (`K1 = H1 H1^T`) contra la
//! del profesor (`K0 = H0 H0^T`) usando el criterio HSIC. Implementación de
//! referencia en Rust puro; los kernels de GPU viven en `saor-opencl`.

use nalgebra::DMatrix;

/// Matriz de Gram `K = H H^T` para un lote de activaciones `[B x D]`.
pub fn gram_matrix(h: &DMatrix<f32>) -> DMatrix<f32> {
    h * h.transpose()
}

/// CKA centrado entre dos matrices de Gram del mismo lote.
///
/// `0` = ninguna alineación, `1` = alineación idéntica (módulo transformación
/// lineal). El filtro determinista del Paso 5 exige `CKA >= 0.90`.
pub fn centered_cka(k0: &DMatrix<f32>, k1: &DMatrix<f32>) -> f32 {
    debug_assert_eq!(k0.nrows(), k1.nrows());
    debug_assert_eq!(k0.ncols(), k1.ncols());
    let n = k0.nrows() as f32;

    // Matriz de centrado H = I - (1/n) J
    let hc = DMatrix::<f32>::identity(k0.nrows(), k0.ncols()) - DMatrix::<f32>::repeat(
        k0.nrows(),
        k0.ncols(),
        1.0 / n,
    );
    let x = &hc * k0 * &hc;
    let y = &hc * k1 * &hc;

    // HSIC: <vec(X), vec(Y)> = sum(X ⊙ Y)
    let hsic_xy = x.component_mul(&y).sum();
    let hsic_xx = x.component_mul(&x).sum();
    let hsic_yy = y.component_mul(&y).sum();

    let denom = (hsic_xx * hsic_yy).sqrt();
    if denom <= f32::EPSILON {
        0.0
    } else {
        hsic_xy / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cka_identidad_es_uno() {
        let h = DMatrix::<f32>::from_row_slice(
            4,
            3,
            &[1.0, 0.0, 0.5, 0.0, 1.0, 0.5, 1.0, 1.0, 0.0, 0.5, 0.5, 1.0],
        );
        let k = gram_matrix(&h);
        let cka = centered_cka(&k, &k);
        assert!((cka - 1.0).abs() < 1e-4, "CKA consigo mismo debe ser ~1, obtuve {cka}");
    }

    #[test]
    fn cka_de_matrices_no_relacionadas_es_baja() {
        let k0 = DMatrix::<f32>::identity(4, 4);
        // Matriz triangular inferior casi vacía por encima de la diagonal: poca
        // alineación con la identidad.
        let k1 = DMatrix::<f32>::from_row_slice(
            4,
            4,
            &[1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        );
        let cka = centered_cka(&k0, &k1);
        assert!(cka < 0.9, "debe estar por debajo del filtro de 0.90, obtuve {cka}");
    }
}
