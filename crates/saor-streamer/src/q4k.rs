//! Cuantización Q4_K (GGML) de los pesos activos del bloque disperso.
//!
//! Espejo del cuantizador de referencia de ggml (`quantize_row_q4_K_reference`):
//! bloques de 256 f32 → 144 bytes (d f16 + dmin f16 + 12 bytes de escalas de
//! 6 bits + 128 bytes de nibbles). El runtime `hayai` dequantiza con su
//! `dequant_q4_k`; este módulo permite exportar el GGUF D16 con los pesos en
//! Q4_K en lugar de F32 (menor tamaño de archivo y banda de memoria).

/// Número de elementos por super-bloque Q4_K.
pub const QK_K: usize = 256;
/// Bytes por super-bloque Q4_K (d 2 + dmin 2 + scales 12 + qs 128).
pub const Q4_K_BLOCK_BYTES: usize = 144;

/// Convierte `f32` a bits `f16` (IEEE 754 half).
pub fn f32_to_f16(value: f32) -> u16 {
    let f = value.to_bits();
    let sign = (f >> 16) & 0x8000;
    let mut exponent = ((f >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = f & 0x7fffff;
    if exponent <= 0 {
        if exponent < -10 {
            return sign as u16;
        }
        // subnormal f16
        let m = mantissa | 0x800000;
        let shift = (14 - exponent) as u32;
        let mut m = m >> shift;
        if mantissa & (1 << (shift.saturating_sub(1))) != 0 {
            m += 1;
        }
        return (sign as u16) | m as u16;
    }
    if exponent >= 31 {
        return (sign as u16) | 0x7c00; // inf/overflow
    }
    let mut m = mantissa >> 13;
    if mantissa & 0x1000 != 0 {
        m += 1;
    }
    if m & 0x400 == 0x400 {
        m = 0;
        exponent += 1;
    }
    if exponent >= 31 {
        return (sign as u16) | 0x7c00;
    }
    (sign as u16) | ((exponent as u16) << 10) | m as u16
}

/// Cuantiza `x` (n % 256 == 0) a Q4_K devolviendo `n/256 * 144` bytes.
pub fn quantize_q4_k(x: &[f32], n: usize) -> Result<Vec<u8>, String> {
    if n % QK_K != 0 {
        return Err(format!("q4_k n={n} no múltiplo de 256"));
    }
    let blocks = n / QK_K;
    let mut out = vec![0u8; blocks * Q4_K_BLOCK_BYTES];
    for b in 0..blocks {
        quantize_q4_k_block(&x[b * QK_K..b * QK_K + QK_K], &mut out[b * Q4_K_BLOCK_BYTES..]);
    }
    Ok(out)
}

fn quantize_q4_k_block(x: &[f32], out: &mut [u8]) {
    let mut scales = [0.0f32; 8];
    let mut mins = [0.0f32; 8];
    let mut l = [0u8; QK_K];
    for j in 0..8 {
        let sub = &x[j * 32..j * 32 + 32];
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for &v in sub {
            mn = mn.min(v);
            mx = mx.max(v);
        }
        scales[j] = (mx - mn) / 15.0;
        mins[j] = mn;
    }
    let max_scale = scales.iter().cloned().fold(0.0f32, f32::max).max(1e-30);
    let max_min = mins.iter().map(|m| m.abs()).fold(0.0f32, f32::max).max(1e-30);
    let d = max_scale / 63.0;
    let dmin = max_min / 63.0;
    let mut ls = [0u8; 8];
    let mut lm = [0u8; 8];
    for j in 0..8 {
        ls[j] = (scales[j] * 63.0 / max_scale).round().clamp(0.0, 63.0) as u8;
        // El dequant resta `dmin*m`; para `-dmin*m ≈ mins[j]` se usa el offset
        // positivo: `m = -mins[j] / dmin`.
        lm[j] = (-mins[j] * 63.0 / max_min).round().clamp(0.0, 63.0) as u8;
    }
    for j in 0..8 {
        let d1 = d * ls[j] as f32;
        if d1 == 0.0 {
            for ii in 0..32 {
                l[j * 32 + ii] = 0;
            }
            continue;
        }
        let dm = dmin * lm[j] as f32;
        for ii in 0..32 {
            let q = ((x[j * 32 + ii] + dm) / d1).round().clamp(0.0, 15.0) as u8;
            l[j * 32 + ii] = q;
        }
    }
    out[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
    out[2..4].copy_from_slice(&f32_to_f16(dmin).to_le_bytes());
    let mut scales_pack = [0u8; 12];
    for j in 0..8 {
        let sc = ls[j];
        let m = lm[j];
        if j < 4 {
            scales_pack[j] = sc;
            scales_pack[j + 4] = m;
        } else {
            scales_pack[j + 4] = (sc & 0xF) | ((m & 0xF) << 4);
            scales_pack[j - 4] |= (sc >> 4) << 6;
            scales_pack[j] |= (m >> 4) << 6;
        }
    }
    out[4..16].copy_from_slice(&scales_pack);
    for j in (0..QK_K).step_by(64) {
        let base = j / 64 * 32;
        for ii in 0..32 {
            out[16 + base + ii] = l[j + ii] | (l[j + ii + 32] << 4);
        }
    }
}
