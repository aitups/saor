//! Red de Patrones de Composición (CPPN) — genoma indirecto desacoplado.
//!
//! Genoma de tamaño constante (~32K params) que, evaluado sobre las coordenadas
//! del sustrato 2D del bloque Transformer, genera un DAG de conexiones. Es la
//! especificación del Paso 4 de la propuesta v4.

use nalgebra::DMatrix;

/// Dimensiones del vector de entrada de la CPPN (8 dims).
pub const CPPN_INPUT_DIM: usize = 8;
/// Tamaño nominal del genoma (~32K).
pub const GENOME_SIZE: usize = 32 * 1024;

/// Tipos de activación heterogénea de los nodos internos.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Gaussiana `exp(-x²)`: esculpe parches locales de conectividad.
    Gaussian,
    /// Tangente hiperbólica: saturación suave en `[-1, 1]`.
    Tanh,
    /// Sigmoide: salida en `(0, 1)` para probabilidades de enlace.
    Sigmoid,
    /// Seno: patrones repetitivos globales.
    Sine,
}

impl Activation {
    /// Aplica la función de activación.
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            Activation::Gaussian => (-x * x).exp(),
            Activation::Tanh => x.tanh(),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Sine => x.sin(),
        }
    }
}

/// Genera el vector de entrada de 8 dims para el par `(i, j)` según v4.
///
/// Coordenadas de capa A (`x = -1`) y B (`x = +1`) en `[-1, 1]`.
pub fn input_vector(d_in: usize, d_out: usize, i: usize, j: usize) -> [f32; CPPN_INPUT_DIM] {
    let y_i = if d_in > 1 {
        -1.0 + 2.0 * i as f32 / (d_in - 1) as f32
    } else {
        0.0
    };
    let y_j = if d_out > 1 {
        -1.0 + 2.0 * j as f32 / (d_out - 1) as f32
    } else {
        0.0
    };
    let x_i = -1.0;
    let x_j = 1.0;
    let dx = x_j - x_i;
    let dy = y_j - y_i;
    [
        x_i,
        y_i,
        x_j,
        y_j,
        dx,
        dy,
        (std::f32::consts::PI * y_i).sin(),
        (std::f32::consts::PI * y_j).cos(),
    ]
}

/// Estructura del genoma CPPN (topología fija de 2 capas ocultas).
///
/// El genoma es un vector plano de pesos + un catálogo fijo de activaciones.
/// La evaluación completa de una matriz `d_in x d_out` es costosa en CPU, por
/// lo que en producción se descodifica en GPU (kernel `cppn_decode.cl`).
#[derive(Debug, Clone)]
pub struct CppnGenome {
    /// Capa 0: entrada (8) -> oculta (64).
    pub w0: DMatrix<f32>,
    /// Capa 1: oculta (64) -> oculta (64).
    pub w1: DMatrix<f32>,
    /// Capa de salida: oculta (64) -> 2 (w_ij, l_ij).
    pub w2: DMatrix<f32>,
    /// Bias de la capa oculta 0.
    pub b0: DMatrix<f32>,
    /// Bias de la capa oculta 1.
    pub b1: DMatrix<f32>,
    /// Bias de la capa de salida (2 filas: `w`, `l`).
    pub b2: DMatrix<f32>,
    /// Activación por neurona oculta (capa 0).
    pub acts0: Vec<Activation>,
    /// Activación por neurona oculta (capa 1).
    pub acts1: Vec<Activation>,
}

impl CppnGenome {
/// Dimensiones internas fijas de la CPPN.
///
/// 16+16 ocultos: equilibrio entre expresividad y rendimiento del kernel OpenCL
/// de decodificación (presión de registros). A 64+64 el evaluador forzaba
/// spilling y el decode colapsaba (~6500× más lento por dispatch en bloques
/// grandes). Para 89M–201M conexiones de los modelos objetivo es determinante.
pub const HIDDEN: usize = 16;

    /// Crea un genoma vacío (ceros) listo para inicializar.
    pub fn zeros() -> Self {
        let h = Self::HIDDEN;
        Self {
            w0: DMatrix::zeros(h, CPPN_INPUT_DIM),
            w1: DMatrix::zeros(h, h),
            w2: DMatrix::zeros(2, h),
            b0: DMatrix::zeros(h, 1),
            b1: DMatrix::zeros(h, 1),
            b2: DMatrix::zeros(2, 1),
            acts0: vec![Activation::Tanh; h],
            acts1: vec![Activation::Sine; h],
        }
    }

    /// Número total de parámetros del genoma.
    pub fn param_count(&self) -> usize {
        self.w0.len() + self.w1.len() + self.w2.len() + self.b0.len() + self.b1.len() + self.b2.len()
    }

    /// Evalúa la CPPN para un par de neuronas `(i, j)`.
    ///
    /// Devuelve `(w_ij, l_ij)` con `l_ij` ya pasado por sigmoide (`∈ (0,1)`).
    pub fn evaluate(&self, v: &[f32; CPPN_INPUT_DIM]) -> (f32, f32) {
        // Capa 0
        let mut h0 = vec![0.0f32; Self::HIDDEN];
        for o in 0..Self::HIDDEN {
            let mut acc = self.b0[(o, 0)];
            for k in 0..CPPN_INPUT_DIM {
                acc += self.w0[(o, k)] * v[k];
            }
            h0[o] = self.acts0[o].apply(acc);
        }
        // Capa 1
        let mut h1 = vec![0.0f32; Self::HIDDEN];
        for o in 0..Self::HIDDEN {
            let mut acc = self.b1[(o, 0)];
            for k in 0..Self::HIDDEN {
                acc += self.w1[(o, k)] * h0[k];
            }
            h1[o] = self.acts1[o].apply(acc);
        }
        // Salida
        let mut w = self.b2[(0, 0)];
        let mut l = self.b2[(1, 0)];
        for k in 0..Self::HIDDEN {
            w += self.w2[(0, k)] * h1[k];
            l += self.w2[(1, k)] * h1[k];
        }
        let l_sig = 1.0 / (1.0 + (-l).exp());
        (w, l_sig)
    }

    /// Aplana el genoma en el orden usado por el kernel OpenCL:
    /// `w0 | b0 | w1 | b1 | w2 | b2` (véase `kernels/cppn_decode.cl`).
    ///
    /// IMPORTANTE: `nalgebra` almacena `DMatrix` por columnas, por lo que
    /// `iter()` no sirve aquí; se itera explícitamente en orden fila-mayor para
    /// que el índice coincida con `genome[o * cols + k]` del kernel.
    pub fn flatten(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.param_count());
        for o in 0..Self::HIDDEN {
            for k in 0..CPPN_INPUT_DIM {
                out.push(self.w0[(o, k)]);
            }
        }
        for o in 0..Self::HIDDEN {
            out.push(self.b0[(o, 0)]);
        }
        for o in 0..Self::HIDDEN {
            for k in 0..Self::HIDDEN {
                out.push(self.w1[(o, k)]);
            }
        }
        for o in 0..Self::HIDDEN {
            out.push(self.b1[(o, 0)]);
        }
        for r in 0..2 {
            for k in 0..Self::HIDDEN {
                out.push(self.w2[(r, k)]);
            }
        }
        out.push(self.b2[(0, 0)]);
        out.push(self.b2[(1, 0)]);
        out
    }

    /// Reconstruye el genoma desde un aplanado producido por [`Self::flatten`].
    pub fn from_flatten(flat: &[f32]) -> Self {
        let mut g = Self::zeros();
        let mut pos = 0;
        for o in 0..Self::HIDDEN {
            for k in 0..CPPN_INPUT_DIM {
                g.w0[(o, k)] = flat[pos];
                pos += 1;
            }
        }
        for o in 0..Self::HIDDEN {
            g.b0[(o, 0)] = flat[pos];
            pos += 1;
        }
        for o in 0..Self::HIDDEN {
            for k in 0..Self::HIDDEN {
                g.w1[(o, k)] = flat[pos];
                pos += 1;
            }
        }
        for o in 0..Self::HIDDEN {
            g.b1[(o, 0)] = flat[pos];
            pos += 1;
        }
        for r in 0..2 {
            for k in 0..Self::HIDDEN {
                g.w2[(r, k)] = flat[pos];
                pos += 1;
            }
        }
        g.b2[(0, 0)] = flat[pos];
        pos += 1;
        g.b2[(1, 0)] = flat[pos];
        pos += 1;
        debug_assert_eq!(pos, flat.len());
        g
    }

    /// Genoma aleatorio determinista (semilla entera), pesos en `[-scale, scale]`.
    pub fn random_with(seed: u64, scale: f32) -> Self {
        use rand_chacha::ChaCha8Rng;
        use rand_core::{RngCore, SeedableRng};
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut fill = |m: &mut DMatrix<f32>| {
            for v in m.iter_mut() {
                *v = (rng.next_u32() as f32 / u32::MAX as f32 - 0.5) * 2.0 * scale;
            }
        };
        let mut g = Self::zeros();
        fill(&mut g.w0);
        fill(&mut g.b0);
        fill(&mut g.w1);
        fill(&mut g.b1);
        fill(&mut g.w2);
        fill(&mut g.b2);
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_vector_respeta_rangos() {
        let v = input_vector(4, 4, 0, 3);
        assert_eq!(v[0], -1.0);
        assert_eq!(v[2], 1.0);
        assert!(v[1] >= -1.0 && v[1] <= 1.0);
        assert!(v[3] >= -1.0 && v[3] <= 1.0);
        assert_eq!(v[4], 2.0); // dx = x_j - x_i
    }

    #[test]
    fn evaluacion_es_determinista_y_en_rango_l() {
        let genome = CppnGenome::zeros();
        let (_, l) = genome.evaluate(&input_vector(8, 8, 2, 5));
        // Con genoma de ceros: b2[(1,0)] = 0 -> sigmoide(0) = 0.5
        assert!((l - 0.5).abs() < 1e-6);
        assert!(l > 0.0 && l < 1.0);
    }

    #[test]
    fn tamano_genoma_es_orden_32k() {
        let genome = CppnGenome::zeros();
        assert_eq!(
            genome.param_count(),
            8 * CppnGenome::HIDDEN
                + CppnGenome::HIDDEN * CppnGenome::HIDDEN
                + CppnGenome::HIDDEN * 2
                + CppnGenome::HIDDEN
                + CppnGenome::HIDDEN
                + 2
        );
    }

    #[test]
    fn flatten_round_trip() {
        let g = CppnGenome::random_with(1234, 0.5);
        let flat = g.flatten();
        assert_eq!(flat.len(), g.param_count());
        let g2 = CppnGenome::from_flatten(&flat);
        // Comparación elemento a elemento (todas las matrices).
        for o in 0..CppnGenome::HIDDEN {
            for k in 0..CPPN_INPUT_DIM {
                assert_eq!(g.w0[(o, k)], g2.w0[(o, k)]);
            }
        }
        assert_eq!(g.b0, g2.b0);
        assert_eq!(g.w1, g2.w1);
        assert_eq!(g.b1, g2.b1);
        assert_eq!(g.w2, g2.w2);
        assert_eq!(g.b2, g2.b2);
    }
}
