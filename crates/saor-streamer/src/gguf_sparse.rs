//! Almacenamiento GGUF disperso (sin densificar).
//!
//! Escribe/lee el bloque candidato en formato GGUF v3 con:
//! * Metadatos `saor.*`: `d_in`, `d_out` (u64), `tau` (f32), `genome`
//!   (array f32 — el genoma CPPN de ~32K) y `sparse` (bool).
//! * Tensores: `ffn_dag_adjacency` (I8 — bit-tensor de adyacencia) y
//!   `ffn_dag_weights` (F32 — pesos activos del DAG).
//!
//! La adyacencia se conserva **tal cual** (bit-tensor `ffn_dag_adjacency`,
//! alineado con `pr_soporte_gguf_disperso_v2.md` de hayai): no se densifica.
//! El empaquetado 4-bit exacto (IQ4/Q4_K) sustituirá a los `f32` cuando la PR
//! de hayai defina el esquema definitivo.

use std::io::{Read, Write};
use std::path::Path;

/// Magic del formato GGUF ("GGUF").
pub const GGUF_MAGIC: u32 = 0x4655_4747;
/// Versión de formato GGUF usada.
pub const GGUF_VERSION: u32 = 3;

/// Nombre del tensor de adyacencia (bit-tensor).
pub const ADJACENCY_TENSOR_NAME: &str = "ffn_dag_adjacency";
/// Nombre del tensor de pesos activos.
pub const WEIGHTS_TENSOR_NAME: &str = "ffn_dag_weights";

/// Tipo GGML F32.
pub const GGML_TYPE_F32: i32 = 0;
/// Tipo GGML I8 (bytes de adyacencia).
pub const GGML_TYPE_I8: i32 = 16;

/// Tipo de valor GGUF FLOAT32.
pub const GGUF_TYPE_FLOAT32: u32 = 6;
/// Tipo de valor GGUF BOOL.
pub const GGUF_TYPE_BOOL: u32 = 7;
/// Tipo de valor GGUF STRING.
pub const GGUF_TYPE_STRING: u32 = 8;
/// Tipo de valor GGUF ARRAY.
pub const GGUF_TYPE_ARRAY: u32 = 9;
/// Tipo de valor GGUF UINT64.
pub const GGUF_TYPE_UINT64: u32 = 10;

/// Clave de metadato: dimensión de entrada.
pub const META_D_IN: &str = "saor.d_in";
/// Clave de metadato: dimensión de salida.
pub const META_D_OUT: &str = "saor.d_out";
/// Clave de metadato: umbral de esparsidad τ.
pub const META_TAU: &str = "saor.tau";
/// Clave de metadato: genoma CPPN aplanado.
pub const META_GENOME: &str = "saor.genome";
/// Clave de metadato: marca de bloque disperso.
pub const META_SPARSE: &str = "saor.sparse";

/// Bloque disperso serializable a GGUF.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseBlock {
    /// Dimensión de entrada del bloque.
    pub d_in: usize,
    /// Dimensión de salida del bloque.
    pub d_out: usize,
    /// Umbral de esparsidad dinámico.
    pub tau: f32,
    /// Genoma CPPN aplanado (referencia para regenerar/re-evolucionar).
    pub genome: Vec<f32>,
    /// Bit-tensor `ffn_dag_adjacency` (LSB-first, fila por fila).
    pub adjacency: Vec<u8>,
    /// Pesos activos del DAG (en orden de escaneo `(i, j)`).
    pub weights: Vec<f32>,
}

impl SparseBlock {
    /// Conexiones activas (popcount sobre el bit-tensor).
    pub fn active_connections(&self) -> usize {
        self.adjacency.iter().map(|b| b.count_ones() as usize).sum()
    }

    /// `D_arch(A0, A1)` = esparcidad del candidato.
    pub fn sparsity(&self) -> f32 {
        let total = (self.d_in * self.d_out).max(1);
        1.0 - self.active_connections() as f32 / total as f32
    }
}

enum MetaValue {
    U64(u64),
    F32(f32),
    Bool(bool),
    String(String),
    F32Array(Vec<f32>),
}

fn write_string(buf: &mut Vec<u8>, s: &str) {
    let b = s.as_bytes();
    buf.extend_from_slice(&(b.len() as u64).to_le_bytes());
    buf.extend_from_slice(b);
}

fn write_kv(buf: &mut Vec<u8>, key: &str, val: &MetaValue) {
    write_string(buf, key);
    match val {
        MetaValue::U64(v) => {
            buf.extend_from_slice(&GGUF_TYPE_UINT64.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        MetaValue::F32(v) => {
            buf.extend_from_slice(&GGUF_TYPE_FLOAT32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        MetaValue::Bool(v) => {
            buf.extend_from_slice(&GGUF_TYPE_BOOL.to_le_bytes());
            buf.push(*v as u8);
        }
        MetaValue::String(s) => {
            buf.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
            write_string(buf, s);
        }
        MetaValue::F32Array(v) => {
            buf.extend_from_slice(&GGUF_TYPE_ARRAY.to_le_bytes());
            buf.extend_from_slice(&GGUF_TYPE_FLOAT32.to_le_bytes());
            buf.extend_from_slice(&(v.len() as u64).to_le_bytes());
            for x in v {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
}

/// Añade un tensor info al buffer y devuelve la posición del campo offset.
fn write_tensor_info(
    buf: &mut Vec<u8>,
    name: &str,
    n_dim: u32,
    dims: &[u64],
    ggml_type: i32,
    offset: u64,
) -> usize {
    write_string(buf, name);
    buf.extend_from_slice(&n_dim.to_le_bytes());
    for d in dims {
        buf.extend_from_slice(&d.to_le_bytes());
    }
    buf.extend_from_slice(&ggml_type.to_le_bytes());
    let offset_pos = buf.len();
    buf.extend_from_slice(&offset.to_le_bytes());
    offset_pos
}

/// Escribe el bloque en un archivo GGUF v3 disperso.
pub fn write_sparse_gguf(path: &Path, block: &SparseBlock) -> Result<(), String> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    let tensor_count: u64 = 2;
    buf.extend_from_slice(&tensor_count.to_le_bytes());

    let kvs: [(&str, MetaValue); 5] = [
        (META_D_IN, MetaValue::U64(block.d_in as u64)),
        (META_D_OUT, MetaValue::U64(block.d_out as u64)),
        (META_TAU, MetaValue::F32(block.tau)),
        (META_GENOME, MetaValue::F32Array(block.genome.clone())),
        (META_SPARSE, MetaValue::Bool(true)),
    ];
    buf.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for (k, v) in &kvs {
        write_kv(&mut buf, k, v);
    }

    let adj_len = block.adjacency.len() as u64;
    let w_len = block.weights.len() as u64;
    let pos_adj = write_tensor_info(
        &mut buf,
        ADJACENCY_TENSOR_NAME,
        1,
        &[adj_len],
        GGML_TYPE_I8,
        0,
    );
    let pos_w = write_tensor_info(
        &mut buf,
        WEIGHTS_TENSOR_NAME,
        1,
        &[w_len],
        GGML_TYPE_F32,
        0,
    );

    let data_start = buf.len() as u64;
    buf[pos_adj..pos_adj + 8].copy_from_slice(&data_start.to_le_bytes());
    buf[pos_w..pos_w + 8].copy_from_slice(&(data_start + adj_len).to_le_bytes());

    buf.extend_from_slice(&block.adjacency);
    for w in &block.weights {
        buf.extend_from_slice(&w.to_le_bytes());
    }

    let mut file = std::fs::File::create(path).map_err(|e| format!("create: {e}"))?;
    file.write_all(&buf).map_err(|e| format!("write: {e}"))?;
    Ok(())
}

fn take<'a>(b: &'a [u8], pos: &mut usize, n: usize) -> Result<&'a [u8], String> {
    let end = pos.checked_add(n).ok_or("desbordamiento de posición")?;
    let s = b.get(*pos..end).ok_or("EOF inesperado")?;
    *pos = end;
    Ok(s)
}

fn read_u32(b: &[u8], pos: &mut usize) -> Result<u32, String> {
    let s = take(b, pos, 4)?;
    Ok(u32::from_le_bytes(s.try_into().map_err(|_| "u32")?))
}

fn read_i32(b: &[u8], pos: &mut usize) -> Result<i32, String> {
    let s = take(b, pos, 4)?;
    Ok(i32::from_le_bytes(s.try_into().map_err(|_| "i32")?))
}

fn read_u64(b: &[u8], pos: &mut usize) -> Result<u64, String> {
    let s = take(b, pos, 8)?;
    Ok(u64::from_le_bytes(s.try_into().map_err(|_| "u64")?))
}

fn read_f32(b: &[u8], pos: &mut usize) -> Result<f32, String> {
    let s = take(b, pos, 4)?;
    Ok(f32::from_le_bytes(s.try_into().map_err(|_| "f32")?))
}

fn read_string(b: &[u8], pos: &mut usize) -> Result<String, String> {
    let len = read_u64(b, pos)? as usize;
    let s = take(b, pos, len)?;
    String::from_utf8(s.to_vec()).map_err(|e| format!("utf8: {e}"))
}

/// Lee un bloque GGUF disperso escrito por [`write_sparse_gguf`].
pub fn read_sparse_gguf(path: &Path) -> Result<SparseBlock, String> {
    let mut file = std::fs::File::open(path).map_err(|e| format!("open: {e}"))?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).map_err(|e| format!("read: {e}"))?;
    let mut pos = 0usize;

    let magic = read_u32(&buf, &mut pos)?;
    if magic != GGUF_MAGIC {
        return Err(format!("magic inválido: {magic:#x}"));
    }
    let version = read_u32(&buf, &mut pos)?;
    if version != GGUF_VERSION {
        return Err(format!("versión no soportada: {version}"));
    }
    let tensor_count = read_u64(&buf, &mut pos)? as usize;
    let kv_count = read_u64(&buf, &mut pos)? as usize;

    let mut d_in = None;
    let mut d_out = None;
    let mut tau = None;
    let mut genome = None;
    let mut sparse = false;
    for _ in 0..kv_count {
        let key = read_string(&buf, &mut pos)?;
        let vtype = read_u32(&buf, &mut pos)?;
        match vtype {
            GGUF_TYPE_UINT64 => {
                let v = read_u64(&buf, &mut pos)?;
                match key.as_str() {
                    META_D_IN => d_in = Some(v as usize),
                    META_D_OUT => d_out = Some(v as usize),
                    _ => {}
                }
            }
            GGUF_TYPE_FLOAT32 => {
                let v = read_f32(&buf, &mut pos)?;
                if key == META_TAU {
                    tau = Some(v);
                }
            }
            GGUF_TYPE_BOOL => {
                let v = buf[pos];
                pos += 1;
                if key == META_SPARSE {
                    sparse = v != 0;
                }
            }
            GGUF_TYPE_STRING => {
                let _ = read_string(&buf, &mut pos)?;
            }
            GGUF_TYPE_ARRAY => {
                let elem_type = read_u32(&buf, &mut pos)?;
                let count = read_u64(&buf, &mut pos)? as usize;
                if elem_type == GGUF_TYPE_FLOAT32 && key == META_GENOME {
                    let mut g = Vec::with_capacity(count);
                    for _ in 0..count {
                        g.push(read_f32(&buf, &mut pos)?);
                    }
                    genome = Some(g);
                } else {
                    let elem_size = match elem_type {
                        GGUF_TYPE_FLOAT32 => 4,
                        GGUF_TYPE_UINT64 => 8,
                        other => {
                            return Err(format!(
                                "tipo de elemento de array no soportado: {other}"
                            ))
                        }
                    };
                    pos += count * elem_size;
                }
            }
            other => return Err(format!("tipo de KV no soportado: {other}")),
        }
    }

    let mut adj_off = None;
    let mut adj_len = 0usize;
    let mut w_off = None;
    let mut w_len = 0usize;
    for _ in 0..tensor_count {
        let name = read_string(&buf, &mut pos)?;
        let n_dim = read_u32(&buf, &mut pos)? as usize;
        let mut dims = Vec::with_capacity(n_dim);
        for _ in 0..n_dim {
            dims.push(read_u64(&buf, &mut pos)? as usize);
        }
        let _ggml_type = read_i32(&buf, &mut pos)?;
        let offset = read_u64(&buf, &mut pos)? as usize;
        match name.as_str() {
            ADJACENCY_TENSOR_NAME => {
                adj_off = Some(offset);
                adj_len = dims.first().copied().unwrap_or(0);
            }
            WEIGHTS_TENSOR_NAME => {
                w_off = Some(offset);
                w_len = dims.first().copied().unwrap_or(0);
            }
            _ => {}
        }
    }

    let d_in = d_in.ok_or_else(|| "falta saor.d_in".to_string())?;
    let d_out = d_out.ok_or_else(|| "falta saor.d_out".to_string())?;
    let tau = tau.ok_or_else(|| "falta saor.tau".to_string())?;
    let genome = genome.ok_or_else(|| "falta saor.genome".to_string())?;
    if !sparse {
        return Err("el bloque no está marcado como disperso".into());
    }
    let adj_off = adj_off.ok_or_else(|| "falta tensor ffn_dag_adjacency".to_string())?;
    let w_off = w_off.ok_or_else(|| "falta tensor ffn_dag_weights".to_string())?;
    let adjacency = buf
        .get(adj_off..adj_off + adj_len)
        .ok_or("EOF en adyacencia")?
        .to_vec();
    let mut weights = Vec::with_capacity(w_len);
    let mut p = w_off;
    for _ in 0..w_len {
        weights.push(read_f32(&buf, &mut p)?);
    }
    Ok(SparseBlock {
        d_in,
        d_out,
        tau,
        genome,
        adjacency,
        weights,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_block() -> SparseBlock {
        // d_in=8, d_out=4 -> 32 conexiones; 6 activas.
        SparseBlock {
            d_in: 8,
            d_out: 4,
            tau: 0.42,
            genome: (0..512).map(|i| i as f32 * 0.001).collect(),
            adjacency: vec![0b0101_0101u8, 0b0000_0011u8, 0u8, 0u8],
            weights: vec![1.0, -2.0, 3.5, 0.25, -0.5, 7.0],
        }
    }

    #[test]
    fn round_trip_gguf_sparse() {
        let dir = std::env::temp_dir();
        let path = dir.join("saor_roundtrip.gguf");
        let block = sample_block();
        write_sparse_gguf(&path, &block).expect("write");
        let read = read_sparse_gguf(&path).expect("read");
        assert_eq!(read, block);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn sparsity_derivada_del_bit_tensor() {
        let block = sample_block();
        assert_eq!(block.active_connections(), 6);
        assert!((block.sparsity() - (1.0 - 6.0 / 32.0)).abs() < 1e-6);
    }
}


