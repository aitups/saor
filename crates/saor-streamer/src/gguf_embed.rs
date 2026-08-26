//! GGUF embebido (Fase 1): reescritura de un GGUF completo sustituyendo tensores
//! densos por bloques dispersos de `saor`.
//!
//! Formato (decisión D16): para cada tensor esparcible `blk.N.<rol>.weight`
//! `[d_in, d_out]` se elimina el tensor denso y se añaden:
//!
//! * `blk.N.<rol>.ffn_dag_adjacency` — bit-tensor de adyacencia (GGML I8=24).
//! * `blk.N.<rol>.ffn_dag_weights` — pesos activos (GGML F32=0; 4-bit en una
//!   fase posterior).
//!
//! Con metadatos por bloque (KV plano prefijado):
//! `saor.<base>.d_in`, `saor.<base>.d_out`, `saor.<base>.tau`,
//! `saor.<base>.sparse`, `saor.<base>.genome` y el global `saor.sparse_count`.
//!
//! El rewriter es *streaming*: nunca carga la sección de datos completa en
//! memoria (crítico para GGUFs de 15–27 GB), copiando cada tensor conservado
//! desde su offset de origen y re-calculando los offsets relativos (align 32).

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::gguf_sparse::{SparseBlock, GGML_TYPE_F32, GGML_TYPE_I8};

/// Magic del formato GGUF ("GGUF").
pub const GGUF_MAGIC: u32 = 0x4655_4747;
/// Versión de formato GGUF usada.
pub const GGUF_VERSION: u32 = 3;
/// Alineación de la sección de datos GGUF (hayai usa 32 por defecto).
pub const GGUF_ALIGN: usize = 32;
/// Sufijo del tensor de adyacencia embebido.
pub const ADJACENCY_SUFFIX: &str = ".ffn_dag_adjacency";
/// Sufijo del tensor de pesos activos embebido.
pub const WEIGHTS_SUFFIX: &str = ".ffn_dag_weights";
/// Clave global: número de bloques dispersos embebidos.
pub const META_SPARSE_COUNT: &str = "saor.sparse_count";

/// Tipos GGML: (id, (numel por bloque, bytes por bloque)). Densos: (1, elem).
const GGML_BLOCK: &[(i32, (u64, u64))] = &[
    (0, (1, 4)),   // F32
    (1, (1, 2)),   // F16
    (2, (32, 18)), // Q4_0
    (3, (32, 20)), // Q4_1
    (6, (32, 22)), // Q5_0
    (7, (32, 24)), // Q5_1
    (8, (32, 34)), // Q8_0
    (9, (32, 40)), // Q8_1
    (10, (256, 84)),  // Q2_K
    (11, (256, 110)), // Q3_K
    (12, (256, 144)), // Q4_K
    (13, (256, 176)), // Q5_K
    (14, (256, 210)), // Q6_K
    (15, (256, 292)), // Q8_K
    (16, (256, 36)),  // IQ2_XXS
    (17, (256, 40)),  // IQ2_XS
    (18, (256, 72)),  // IQ3_XXS
    (19, (256, 36)),  // IQ1_S
    (20, (32, 18)),   // IQ4_NL
    (21, (256, 56)),  // IQ3_S
    (22, (256, 46)),  // IQ2_S
    (23, (256, 64)),  // IQ4_XS
    (24, (1, 1)),     // I8
    (25, (1, 2)),     // I16
    (26, (1, 4)),     // I32
    (27, (1, 8)),     // I64
    (28, (1, 8)),     // F64
    (29, (256, 56)),  // IQ1_M
    (30, (1, 2)),     // BF16
    (34, (256, 74)),  // TQ1_0
    (35, (256, 120)), // TQ2_0
];

/// Bytes de un tensor GGML dado su tipo y `numel`.
pub fn ggml_nbytes(ggml_type: i32, numel: u64) -> Result<u64, String> {
    let &(_, (numel_per_block, bytes_per_block)) = GGML_BLOCK
        .iter()
        .find(|(id, _)| *id == ggml_type)
        .ok_or_else(|| format!("tipo GGML no soportado: {ggml_type}"))?;
    let blocks = numel.div_ceil(numel_per_block);
    Ok(blocks * bytes_per_block)
}

/// Información de un tensor del GGUF de origen (sin datos).
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Nombre del tensor.
    pub name: String,
    /// Dimensiones en orden GGUF.
    pub dims: Vec<u64>,
    /// Tipo GGML.
    pub ggml_type: i32,
    /// Offset relativo a la sección de datos alineada.
    pub offset: u64,
    /// Bytes reales en el archivo de origen (derivados del delta de offsets).
    pub nbytes: u64,
}

/// Cabecera GGUF parseada genéricamente (KV + tensor infos).
pub struct SrcHeader {
    /// Número de pares KV originales.
    pub kv_count: u64,
    /// Bytes crudos de la sección KV (para re-serialización idéntica).
    pub kv_bytes: Vec<u8>,
    /// Infos de los tensores de origen, en orden.
    pub tensors: Vec<TensorInfo>,
    /// Posición absoluta donde empieza la sección de datos alineada.
    pub data_start_abs: u64,
}

fn align_up(x: u64, align: u64) -> u64 {
    x.div_ceil(align) * align
}

fn read_exact_at(f: &mut File, buf: &mut [u8], pos: u64) -> Result<(), String> {
    f.seek(SeekFrom::Start(pos)).map_err(|e| format!("seek: {e}"))?;
    f.read_exact(buf).map_err(|e| format!("read: {e}"))
}

/// Lector secuencial sobre un `File` abierto (avanza por `pos` absoluto).
struct SeqReader<'a> {
    f: &'a mut File,
    pos: u64,
}

impl SeqReader<'_> {
    fn take(&mut self, n: usize) -> Result<Vec<u8>, String> {
        let mut b = vec![0u8; n];
        read_exact_at(self.f, &mut b, self.pos)?;
        self.pos += n as u64;
        Ok(b)
    }

    fn u32(&mut self) -> Result<u32, String> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes(b.try_into().unwrap()))
    }

    fn i32(&mut self) -> Result<i32, String> {
        Ok(self.u32()? as i32)
    }

    fn u64(&mut self) -> Result<u64, String> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String, String> {
        let n = self.u64()? as usize;
        let b = self.take(n)?;
        std::str::from_utf8(&b)
            .map(|s| s.to_string())
            .map_err(|e| format!("string no utf8: {e}"))
    }

    /// Salta el valor de un KV de tipo `vtype`.
    fn skip_kv_value(&mut self, vtype: u32) -> Result<(), String> {
        let scalar_size = |v: u32| -> Option<usize> {
            Some(match v {
                0 | 1 => 1,      // uint8, int8
                2 | 3 => 2,      // uint16, int16
                4 | 5 | 6 => 4,  // uint32, int32, float32
                7 => 1,          // bool
                10 | 11 | 12 => 8, // uint64, int64, float64
                _ => return None,
            })
        };
        match vtype {
            8 => {
                let n = self.u64()? as usize;
                self.take(n)?;
            }
            9 => {
                let elem_type = self.u32()?;
                let count = self.u64()?;
                if elem_type == crate::gguf_sparse::GGUF_TYPE_STRING {
                    // ARRAY de STRING (p. ej. tokenizer.ggml.tokens): cada
                    // elemento es una string (longitud + bytes).
                    for _ in 0..count {
                        let n = self.u64()? as usize;
                        self.take(n)?;
                    }
                } else {
                    let elem_size = scalar_size(elem_type).ok_or_else(|| {
                        format!("elemento de array GGUF no soportado: {elem_type}")
                    })?;
                    self.take((count as usize).saturating_mul(elem_size))?;
                }
            }
            other => {
                let size = scalar_size(other)
                    .ok_or_else(|| format!("tipo de KV GGUF no soportado: {other}"))?;
                self.take(size)?;
            }
        }
        Ok(())
    }
}

/// Parsea la cabecera GGUF v3 de un archivo abierto (KV + tensor infos).
pub fn parse_header(src: &mut File) -> Result<SrcHeader, String> {
    let mut r = SeqReader { f: src, pos: 0 };
    let magic = r.u32()?;
    if magic != GGUF_MAGIC {
        return Err(format!("magic inválido: {magic:#x}"));
    }
    let version = r.u32()?;
    if version != GGUF_VERSION {
        return Err(format!("versión no soportada: {version}"));
    }
    let tensor_count = r.u64()?;
    let kv_count = r.u64()?;
    let kv_start = r.pos;

    // Recorrer la sección KV para localizar su final (los valores se saltan).
    for _ in 0..kv_count {
        r.string()?; // key
        let vtype = r.u32()?;
        r.skip_kv_value(vtype)?;
    }
    let kv_bytes = {
        let n = (r.pos - kv_start) as usize;
        let mut b = vec![0u8; n];
        read_exact_at(&mut *r.f, &mut b, kv_start)?;
        b
    };

    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = r.string()?;
        let n_dim = r.u32()? as usize;
        let mut dims = Vec::with_capacity(n_dim);
        for _ in 0..n_dim {
            dims.push(r.u64()?);
        }
        let ggml_type = r.i32()?;
        let offset = r.u64()?;
        tensors.push(TensorInfo {
            name,
            dims,
            ggml_type,
            offset,
            nbytes: 0,
        });
    }
    // Bytes reales por tensor: delta de offsets (con padding) para los no
    // últimos; el último por la tabla GGML.
    for i in 0..tensors.len() {
        let nbytes = if i + 1 < tensors.len() {
            tensors[i + 1].offset.saturating_sub(tensors[i].offset)
        } else {
            let numel: u64 = tensors[i].dims.iter().product();
            ggml_nbytes(tensors[i].ggml_type, numel)?
        };
        tensors[i].nbytes = nbytes;
    }
    let data_start_abs = align_up(r.pos, GGUF_ALIGN as u64);
    Ok(SrcHeader {
        kv_count,
        kv_bytes,
        tensors,
        data_start_abs,
    })
}


/// Sustitución de un tensor denso por un bloque disperso embebido.
#[derive(Debug, Clone)]
pub struct BlockReplacement {
    /// Nombre del tensor denso original (ej. `blk.0.ffn_gate.weight`).
    pub tensor: String,
    /// Bloque disperso (bit-tensor + pesos activos + tau + genoma).
    pub block: SparseBlock,
    /// Ruta a un fichero con los pesos F32 del bloque (streaming): para bloques
    /// grandes evita retener los pesos en RAM (ALIA 40B: ~35 GB en RAM).
    pub weights_file: Option<std::path::PathBuf>,
}

/// Reporte de la reescritura.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EmbedReport {
    /// Tensores totales del GGUF resultante.
    pub tensor_count: usize,
    /// Bloques sustituidos.
    pub replaced: usize,
    /// Bytes copiados de tensores conservados.
    pub kept_bytes: u64,
    /// Bytes de los bloques dispersos embebidos.
    pub sparse_bytes: u64,
    /// Tamaño total del archivo resultante.
    pub total_bytes: u64,
}

/// Nombre base de un tensor (`blk.0.ffn_gate.weight` -> `blk.0.ffn_gate`).
pub fn base_name(name: &str) -> &str {
    name.strip_suffix(".weight").unwrap_or(name)
}

fn write_kv(buf: &mut Vec<u8>, key: &str, vtype: u32, value: &[u8]) {
    let kb = key.as_bytes();
    buf.extend_from_slice(&(kb.len() as u64).to_le_bytes());
    buf.extend_from_slice(kb);
    buf.extend_from_slice(&vtype.to_le_bytes());
    buf.extend_from_slice(value);
}

fn kv_u64(key: &str, v: u64, out: &mut Vec<u8>) {
    write_kv(out, key, crate::gguf_sparse::GGUF_TYPE_UINT64, &v.to_le_bytes());
}

fn kv_f32(key: &str, v: f32, out: &mut Vec<u8>) {
    write_kv(out, key, crate::gguf_sparse::GGUF_TYPE_FLOAT32, &v.to_le_bytes());
}

fn kv_bool_true(key: &str, out: &mut Vec<u8>) {
    write_kv(out, key, crate::gguf_sparse::GGUF_TYPE_BOOL, &[1u8]);
}

fn kv_f32_array(key: &str, v: &[f32], out: &mut Vec<u8>) {
    let mut value = Vec::with_capacity(4 + 8 + v.len() * 4);
    value.extend_from_slice(&crate::gguf_sparse::GGUF_TYPE_FLOAT32.to_le_bytes());
    value.extend_from_slice(&(v.len() as u64).to_le_bytes());
    for x in v {
        value.extend_from_slice(&x.to_le_bytes());
    }
    write_kv(out, key, crate::gguf_sparse::GGUF_TYPE_ARRAY, &value);
}

/// Tensor del archivo de salida.
struct OutTensor {
    name: String,
    dims: Vec<u64>,
    ggml_type: i32,
    nbytes: u64,
    offset: u64,
    src_abs: Option<u64>,
    data: Option<Vec<u8>>,
    /// Pesos F32 en un fichero externo (streaming): evita retenerlos en RAM.
    data_file: Option<std::path::PathBuf>,
}

/// Reescribe `src` en `dst` sustituyendo los tensores indicados por bloques
/// dispersos embebidos. **Streaming:** copia la sección de datos en chunks,
/// sin cargar nunca el archivo completo (crítico para GGUFs de 15–27 GB).
pub fn rewrite_embedded(
    src: &Path,
    dst: &Path,
    replacements: &[BlockReplacement],
) -> Result<EmbedReport, String> {
    let mut srcf = File::open(src).map_err(|e| format!("abrir {src:?}: {e}"))?;
    let h = parse_header(&mut srcf)?;

    let mut repl: HashMap<&str, &SparseBlock> = HashMap::new();
    for r in replacements {
        if repl.insert(r.tensor.as_str(), &r.block).is_some() {
            return Err(format!("tensor duplicado en reemplazos: {}", r.tensor));
        }
    }

    // 1) Tensores de salida: conservados (orden de origen) + dispersos.
    let mut out: Vec<OutTensor> = Vec::with_capacity(h.tensors.len() + replacements.len() * 2);
    for t in &h.tensors {
        if repl.contains_key(t.name.as_str()) {
            continue;
        }
        out.push(OutTensor {
            name: t.name.clone(),
            dims: t.dims.clone(),
            ggml_type: t.ggml_type,
            nbytes: t.nbytes,
            offset: 0,
            src_abs: Some(h.data_start_abs + t.offset),
            data: None,
            data_file: None,
        });
    }
    let mut base_names: Vec<&str> = repl.keys().copied().collect();
    base_names.sort();
    for r in replacements {
        let block = &r.block;
        let bn = base_name(&r.tensor);
        // Pesos F32: en memoria (normal) o en fichero externo (streaming, ALIA).
        let (wdata, wfile, wnbytes) = match &r.weights_file {
            Some(path) => {
                let len = std::fs::metadata(path)
                    .map_err(|e| format!("pesos {path:?}: {e}"))?
                    .len();
                (None, Some(path.clone()), len)
            }
            None => {
                let mut wdata = Vec::with_capacity(block.weights.len() * 4);
                for w in &block.weights {
                    wdata.extend_from_slice(&w.to_le_bytes());
                }
                let len = wdata.len() as u64;
                (Some(wdata), None, len)
            }
        };
        out.push(OutTensor {
            name: format!("{bn}{ADJACENCY_SUFFIX}"),
            dims: vec![block.adjacency.len() as u64],
            ggml_type: GGML_TYPE_I8,
            nbytes: block.adjacency.len() as u64,
            offset: 0,
            src_abs: None,
            data: Some(block.adjacency.clone()),
            data_file: None,
        });
        out.push(OutTensor {
            name: format!("{bn}{WEIGHTS_SUFFIX}"),
            dims: vec![if r.weights_file.is_some() {
                wnbytes / 4
            } else {
                block.weights.len() as u64
            }],
            ggml_type: GGML_TYPE_F32,
            nbytes: wnbytes,
            offset: 0,
            src_abs: None,
            data: wdata,
            data_file: wfile,
        });
    }

    // 2) Offsets relativos a la sección de datos alineada (align 32).
    let mut off = 0u64;
    for t in &mut out {
        off = align_up(off, GGUF_ALIGN as u64);
        t.offset = off;
        off += t.nbytes;
    }
    build_and_write(&mut srcf, &h, &out, &repl, &base_names, replacements.len(), dst)
}


/// Construye la cabecera nueva y escribe cabecera + sección de datos en `dst`.
#[allow(clippy::too_many_arguments)]
fn build_and_write(
    srcf: &mut File,
    h: &SrcHeader,
    out: &[OutTensor],
    repl: &HashMap<&str, &SparseBlock>,
    base_names: &[&str],
    replaced: usize,
    dst: &Path,
) -> Result<EmbedReport, String> {
    let new_kv_count = 1 + replaced * 5;
    let mut hdr = Vec::new();
    hdr.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    hdr.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    hdr.extend_from_slice(&(out.len() as u64).to_le_bytes());
    hdr.extend_from_slice(&(h.kv_count + new_kv_count as u64).to_le_bytes());
    hdr.extend_from_slice(&h.kv_bytes);
    kv_u64(META_SPARSE_COUNT, replaced as u64, &mut hdr);
    for base in base_names {
        let block = repl[*base];
        let bn = base_name(base);
        kv_u64(&format!("saor.{bn}.d_in"), block.d_in as u64, &mut hdr);
        kv_u64(&format!("saor.{bn}.d_out"), block.d_out as u64, &mut hdr);
        kv_f32(&format!("saor.{bn}.tau"), block.tau, &mut hdr);
        kv_bool_true(&format!("saor.{bn}.sparse"), &mut hdr);
        kv_f32_array(&format!("saor.{bn}.genome"), &block.genome, &mut hdr);
    }
    for t in out {
        let nb = t.name.as_bytes();
        hdr.extend_from_slice(&(nb.len() as u64).to_le_bytes());
        hdr.extend_from_slice(nb);
        hdr.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            hdr.extend_from_slice(&d.to_le_bytes());
        }
        hdr.extend_from_slice(&t.ggml_type.to_le_bytes());
        hdr.extend_from_slice(&t.offset.to_le_bytes());
    }
    while (hdr.len() as u64) % GGUF_ALIGN as u64 != 0 {
        hdr.push(0);
    }

    let mut dstf = File::create(dst).map_err(|e| format!("crear {dst:?}: {e}"))?;
    dstf
        .write_all(&hdr)
        .map_err(|e| format!("escribir cabecera: {e}"))?;
    let mut buf = vec![0u8; 1 << 20];
    let mut kept_bytes = 0u64;
    let mut sparse_bytes = 0u64;
    let mut written = 0u64;
    for t in out {
        // Rellenar hasta el offset alineado declarado en la cabecera.
        while written < t.offset {
            dstf
                .write_all(&[0u8])
                .map_err(|e| format!("rellenar: {e}"))?;
            written += 1;
        }
        match (t.src_abs, &t.data) {
            (Some(src_abs), None) => {
                let mut remaining = t.nbytes;
                let mut pos = src_abs;
                while remaining > 0 {
                    let n = (remaining as usize).min(buf.len());
                    read_exact_at(&mut *srcf, &mut buf[..n], pos)?;
                    dstf
                        .write_all(&buf[..n])
                        .map_err(|e| format!("escribir datos: {e}"))?;
                    pos += n as u64;
                    remaining -= n as u64;
                }
                written += t.nbytes;
                kept_bytes += t.nbytes;
            }
            (None, Some(d)) => {
                dstf
                    .write_all(d)
                    .map_err(|e| format!("escribir disperso: {e}"))?;
                written += t.nbytes;
                sparse_bytes += t.nbytes;
            }
            (None, None) if t.data_file.is_some() => {
                // Pesos F32 en fichero externo (streaming): no caben en RAM.
                let path = t.data_file.as_ref().unwrap().clone();
                let mut srcw =
                    File::open(&path).map_err(|e| format!("abrir pesos: {e}"))?;
                let mut remaining = t.nbytes;
                while remaining > 0 {
                    let n = (remaining as usize).min(buf.len());
                    srcw
                        .read_exact(&mut buf[..n])
                        .map_err(|e| format!("leer pesos: {e}"))?;
                    dstf
                        .write_all(&buf[..n])
                        .map_err(|e| format!("escribir pesos: {e}"))?;
                    remaining -= n as u64;
                }
                // Liberar el fichero temporal (el pico wdata+output no cabe en
                // discos medianos para ALIA-40b).
                let _ = std::fs::remove_file(&path);
                written += t.nbytes;
                sparse_bytes += t.nbytes;
            }
            _ => return Err("tensor sin origen ni datos en la salida".into()),
        }
    }
    dstf.flush().map_err(|e| format!("flush: {e}"))?;
    let total_bytes = dstf.metadata().map(|m| m.len()).unwrap_or(0);
    Ok(EmbedReport {
        tensor_count: out.len(),
        replaced,
        kept_bytes,
        sparse_bytes,
        total_bytes,
    })
}


/// Valor de un KV GGUF (solo los tipos que interesan al bloque embebido).
enum KvValue {
    U64(u64),
    F32(f32),
    Bool(bool),
    F32Array(Vec<f32>),
}

/// Parsea los KV de la sección cruda y devuelve los prefijados con `saor.`.
fn parse_saor_kv(kv_bytes: &[u8]) -> Result<HashMap<String, KvValue>, String> {
    let mut pos = 0usize;
    let mut out = HashMap::new();
    let read_string = |pos: &mut usize| -> Result<String, String> {
        if *pos + 8 > kv_bytes.len() {
            return Err("EOF en kv string".into());
        }
        let n = u64::from_le_bytes(kv_bytes[*pos..*pos + 8].try_into().unwrap()) as usize;
        *pos += 8;
        if *pos + n > kv_bytes.len() {
            return Err("EOF en kv string data".into());
        }
        let s = std::str::from_utf8(&kv_bytes[*pos..*pos + n])
            .map_err(|e| format!("kv no utf8: {e}"))?
            .to_string();
        *pos += n;
        Ok(s)
    };
    while pos < kv_bytes.len() {
        let key = read_string(&mut pos)?;
        if pos + 4 > kv_bytes.len() {
            return Err("EOF en kv vtype".into());
        }
        let vtype = u32::from_le_bytes(kv_bytes[pos..pos + 4].try_into().unwrap());
        pos += 4;
        match vtype {
            10 => {
                let v = u64::from_le_bytes(kv_bytes[pos..pos + 8].try_into().unwrap());
                pos += 8;
                out.insert(key, KvValue::U64(v));
            }
            6 => {
                let v = f32::from_le_bytes(kv_bytes[pos..pos + 4].try_into().unwrap());
                pos += 4;
                out.insert(key, KvValue::F32(v));
            }
            7 => {
                let v = kv_bytes[pos] != 0;
                pos += 1;
                out.insert(key, KvValue::Bool(v));
            }
            9 => {
                let elem = u32::from_le_bytes(kv_bytes[pos..pos + 4].try_into().unwrap());
                pos += 4;
                let count = u64::from_le_bytes(kv_bytes[pos..pos + 8].try_into().unwrap());
                pos += 8;
                if elem == crate::gguf_sparse::GGUF_TYPE_FLOAT32 {
                    let mut v = Vec::with_capacity(count as usize);
                    for _ in 0..count {
                        v.push(f32::from_le_bytes(
                            kv_bytes[pos..pos + 4].try_into().unwrap(),
                        ));
                        pos += 4;
                    }
                    out.insert(key, KvValue::F32Array(v));
                } else if elem == crate::gguf_sparse::GGUF_TYPE_STRING {
                    // ARRAY de STRING: saltar cada string.
                    for _ in 0..count {
                        let n = u64::from_le_bytes(kv_bytes[pos..pos + 8].try_into().unwrap())
                            as usize;
                        pos += 8 + n;
                    }
                } else {
                    // Elementos de otros tipos: avanzar con el tamaño correcto.
                    let elem_size = match elem {
                        0 | 1 => 1,
                        2 | 3 => 2,
                        4 | 5 => 4,
                        7 => 1,
                        10 | 11 | 12 => 8,
                        other => return Err(format!("elemento de array GGUF no soportado: {other}")),
                    };
                    pos += (count as usize).saturating_mul(elem_size);
                }
            }
            8 => {
                let n = u64::from_le_bytes(kv_bytes[pos..pos + 8].try_into().unwrap()) as usize;
                pos += 8 + n;
            }
            0 | 1 => pos += 1,
            2 | 3 => pos += 2,
            4 | 5 => pos += 4,
            11 => pos += 8,
            12 => pos += 8,
            _ => return Err(format!("KV de tipo {vtype} no soportado en reader")),
        }
    }
    Ok(out)
}

/// Lee el bloque disperso embebido de un tensor base (`blk.0.ffn_gate.weight`).
/// Devuelve `Ok(None)` si el tensor no está marcado como disperso.
pub fn read_embedded_block(src: &Path, tensor_name: &str) -> Result<Option<SparseBlock>, String> {
    let mut f = File::open(src).map_err(|e| format!("abrir {src:?}: {e}"))?;
    let h = parse_header(&mut f)?;
    let bn = base_name(tensor_name);
    let adj_name = format!("{bn}{ADJACENCY_SUFFIX}");
    let w_name = format!("{bn}{WEIGHTS_SUFFIX}");

    let kv = parse_saor_kv(&h.kv_bytes)?;
    let m = |k: &str| kv.get(&format!("saor.{bn}.{k}"));
    let sparse = matches!(m("sparse"), Some(KvValue::Bool(true)));
    if !sparse {
        return Ok(None);
    }
    let get = |n: &str| -> Result<&TensorInfo, String> {
        h.tensors
            .iter()
            .find(|t| t.name == n)
            .ok_or_else(|| format!("tensor embebido '{n}' no encontrado"))
    };
    let adj_t = get(&adj_name)?;
    let w_t = get(&w_name)?;

    let adj_len = adj_t.dims.first().copied().unwrap_or(0) as usize;
    let mut adjacency = vec![0u8; adj_len];
    read_exact_at(&mut f, &mut adjacency, h.data_start_abs + adj_t.offset)?;

    // Lectura en bloque de los pesos activos (evita un seek por float).
    let w_count = w_t.dims.first().copied().unwrap_or(0) as usize;
    let mut raw = vec![0u8; w_count.saturating_mul(4)];
    read_exact_at(&mut f, &mut raw, h.data_start_abs + w_t.offset)?;
    let weights: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let genome = match m("genome") {
        Some(KvValue::F32Array(g)) => g.clone(),
        _ => Vec::new(),
    };
    let d_in = match m("d_in") {
        Some(KvValue::U64(v)) => *v as usize,
        _ => return Err("falta saor.<base>.d_in".into()),
    };
    let d_out = match m("d_out") {
        Some(KvValue::U64(v)) => *v as usize,
        _ => return Err("falta saor.<base>.d_out".into()),
    };
    let tau = match m("tau") {
        Some(KvValue::F32(v)) => *v,
        _ => return Err("falta saor.<base>.tau".into()),
    };
    Ok(Some(SparseBlock {
        d_in,
        d_out,
        tau,
        genome,
        adjacency,
        weights,
    }))
}


#[cfg(test)]
mod tests {
    use super::*;

    fn w64(buf: &mut Vec<u8>, v: u64) {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    fn w32(buf: &mut Vec<u8>, v: u32) {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    fn wi32(buf: &mut Vec<u8>, v: i32) {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    fn wstr(buf: &mut Vec<u8>, s: &str) {
        let b = s.as_bytes();
        w64(buf, b.len() as u64);
        buf.extend_from_slice(b);
    }

    fn f32b(v: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(v.len() * 4);
        for x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }

    /// GGUF v3 sintético con 4 tensores (F32 + Q4_0) y una KV de arquitectura.
    fn write_mini_model(path: &Path) {
        let ffn: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let norm: Vec<f32> = (0..4).map(|i| i as f32 * 0.5).collect();
        let mut q4 = vec![0u8; 18]; // Q4_0: 1 bloque de 32 (18 bytes)
        q4[0] = 40;
        q4[1] = 0x00;
        let out: Vec<f32> = (100..132).map(|i| i as f32).collect();

        let tensors: Vec<(String, Vec<u64>, i32, Vec<u8>)> = vec![
            ("blk.0.ffn_gate.weight".into(), vec![6, 4], 0, f32b(&ffn)),
            ("blk.0.attn_norm.weight".into(), vec![4], 0, f32b(&norm)),
            ("token_embd.weight".into(), vec![8, 4], 2, q4),
            ("output.weight".into(), vec![8, 4], 0, f32b(&out)),
        ];

        let mut hdr = Vec::new();
        hdr.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        hdr.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        w64(&mut hdr, tensors.len() as u64);
        w64(&mut hdr, 1); // kv_count
        wstr(&mut hdr, "general.architecture");
        w32(&mut hdr, 8); // GGUF_TYPE_STRING
        wstr(&mut hdr, "llm");

        let mut off = 0u64;
        for (name, dims, gtype, _data) in tensors.iter() {
            off = align_up(off, GGUF_ALIGN as u64);
            wstr(&mut hdr, name);
            w32(&mut hdr, dims.len() as u32);
            for d in dims {
                w64(&mut hdr, *d);
            }
            wi32(&mut hdr, *gtype);
            w64(&mut hdr, off);
            let numel: u64 = dims.iter().product();
            off += ggml_nbytes(*gtype, numel).unwrap();
        }
        while (hdr.len() as u64) % GGUF_ALIGN as u64 != 0 {
            hdr.push(0);
        }
        let mut f = File::create(path).unwrap();
        f.write_all(&hdr).unwrap();
        let mut written = 0u64;
        for (_name, dims, gtype, data) in tensors.iter() {
            // Rellenar hasta el offset alineado declarado en la cabecera.
            let expected = align_up(written, GGUF_ALIGN as u64);
            while written < expected {
                f.write_all(&[0u8]).unwrap();
                written += 1;
            }
            f.write_all(data).unwrap();
            written += data.len() as u64;
            let _ = dims;
            let _ = gtype;
        }
    }

    fn sample_block() -> SparseBlock {
        SparseBlock {
            d_in: 4,
            d_out: 6,
            tau: 0.42,
            genome: vec![0.1, -0.2, 0.3],
            adjacency: vec![0b0011_0000],
            weights: vec![1.5, -2.25],
        }
    }

    #[test]
    fn rewrite_embedded_roundtrip() {
        let dir = std::env::temp_dir();
        let src = dir.join("saor_embed_src.gguf");
        let dst = dir.join("saor_embed_dst.gguf");
        write_mini_model(&src);

        let block = sample_block();
        let report = rewrite_embedded(
            &src,
            &dst,
            &[BlockReplacement {
                tensor: "blk.0.ffn_gate.weight".into(),
                block: block.clone(),
                weights_file: None,
            }],
        )
        .expect("rewrite");
        assert_eq!(report.replaced, 1);
        assert_eq!(report.tensor_count, 5); // 4 - 1 + 2

        // 1) El bloque disperso se lee idéntico.
        let got = read_embedded_block(&dst, "blk.0.ffn_gate.weight")
            .expect("read")
            .expect("sparse");
        assert_eq!(got, block);

        // 2) El tensor denso original desapareció y aparecen los dispersos.
        let mut f = File::open(&dst).unwrap();
        let h = parse_header(&mut f).unwrap();
        assert!(!h.tensors.iter().any(|t| t.name == "blk.0.ffn_gate.weight"));
        assert!(
            h.tensors
                .iter()
                .any(|t| t.name == "blk.0.ffn_gate.ffn_dag_adjacency")
        );
        assert_eq!(h.kv_count, 1 + 1 + 5); // original + sparse_count + 5 por bloque

        // 3) Los tensores conservados conservan sus datos byte a byte.
        let mut srcf = File::open(&src).unwrap();
        let hs = parse_header(&mut srcf).unwrap();
        for name in ["blk.0.attn_norm.weight", "token_embd.weight", "output.weight"] {
            let ts = hs.tensors.iter().find(|t| t.name == name).unwrap();
            let td = h.tensors.iter().find(|t| t.name == name).unwrap();
            assert_eq!(ts.nbytes, td.nbytes, "bytes de {name}");
            let mut a = vec![0u8; ts.nbytes as usize];
            let mut b = vec![0u8; td.nbytes as usize];
            read_exact_at(&mut srcf, &mut a, hs.data_start_abs + ts.offset).unwrap();
            read_exact_at(&mut f, &mut b, h.data_start_abs + td.offset).unwrap();
            assert_eq!(a, b, "datos de {name} deben copiarse intactos");
        }

        // 4) El GGUF resultante sigue siendo parseable dentro del rango del archivo.
        assert!(
            h.data_start_abs + h.tensors.last().unwrap().offset <= dst.metadata().unwrap().len()
        );

        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dst);
    }

    #[test]
    fn read_embedded_block_no_sparse_devuelve_none() {
        let dir = std::env::temp_dir();
        let src = dir.join("saor_embed_plain.gguf");
        write_mini_model(&src);
        let r = read_embedded_block(&src, "blk.0.ffn_gate.weight").expect("read");
        assert!(r.is_none());
        let _ = std::fs::remove_file(&src);
    }
}

