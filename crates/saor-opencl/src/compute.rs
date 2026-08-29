//! Motor de cómputo OpenCL (Fase 3).
//!
//! Gestiona el contexto en la RTX 4050, compila los kernels de la crate
//! (`kernels/*.cl`, OpenCL C 1.2 — D9) y ejecuta la decodificación CPPN, el
//! SpMM del DAG y la matriz de Gram para CKA.
//!
//! **Mitigación WDDM/TDR:** todos los envíos 1D se fragmentan en trozos de
//! [`WDDM_CHUNK`] work-items encolados de forma encadenada por eventos, de modo
//! que ningún dispatch individual exceda ~2 s (límite del mecanismo TDR de
//! Windows).

use std::ptr;

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::Device;
use opencl3::kernel::Kernel;
use opencl3::memory::Buffer;
use opencl3::program::Program;
use opencl3::types::{cl_event, cl_mem_flags};

/// CL_MEM_READ_WRITE (1 << 0) del estándar OpenCL (no re-exportado por cl3).
const CL_MEM_READ_WRITE: cl_mem_flags = 1;

const CPPN_DECODE_SRC: &str = include_str!("../kernels/cppn_decode.cl");
const SPMM_SRC: &str = include_str!("../kernels/spmm.cl");
const GRAM_SRC: &str = include_str!("../kernels/gram.cl");

/// Máximo de work-items por dispatch (mitigación WDDM/TDR).
pub const WDDM_CHUNK: usize = 8192;

/// Chunk para la decodificación CPPN (1 work-item por conexión): 8K conexiones
/// por dispatch evita el TDR de Windows (cada dispatch muy por debajo de ~2 s
/// incluso con presión de registros del evaluador CPPN). Para bloques reales de
/// 89M–201M conexiones se emiten ~11K–25K dispatches fragmentados.
pub const CPPN_DECODE_CHUNK: usize = 8192;

/// Motor OpenCL sobre la primera GPU del sistema (RTX 4050).
pub struct ClEngine {
    context: Context,
    device: Device,
    queue: CommandQueue,
    programs: Vec<Program>,
}

impl ClEngine {
    /// Inicializa el motor en la primera GPU (NVIDIA por preferencia).
    pub fn init() -> Result<Self, String> {
        let (_platform, device) = crate::context::first_gpu_device()?;
        let context =
            Context::from_device(&device).map_err(|e| format!("context: {e}"))?;
        let queue = unsafe { CommandQueue::create(&context, device.id(), 0) }
            .map_err(|e| format!("command_queue: {e}"))?;
        Ok(Self {
            context,
            device,
            queue,
            programs: Vec::new(),
        })
    }

    /// Nombre del dispositivo activo.
    pub fn device_name(&self) -> String {
        self.device.name().unwrap_or_default()
    }

    fn build_program(&mut self, name: &str, source: &str) -> Result<Program, String> {
        let mut program = Program::create_from_source(&self.context, source)
            .map_err(|e| format!("{name}: create_from_source: {e}"))?;
        program.build(&[self.device.id()], "").map_err(|e| {
            let log = program
                .get_build_log(self.device.id())
                .map(|l| l.to_string())
                .unwrap_or_else(|_| "sin log".into());
            format!("{name}: build falló: {e}\n{log}")
        })?;
        Ok(program)
    }

    /// Compila los tres programas de la fase.
    pub fn prepare(&mut self) -> Result<(), String> {
        self.programs = vec![
            self.build_program("cppn_decode", CPPN_DECODE_SRC)?,
            self.build_program("spmm", SPMM_SRC)?,
            self.build_program("gram", GRAM_SRC)?,
        ];
        Ok(())
    }

    fn kernel(&self, program_idx: usize, name: &str) -> Result<Kernel, String> {
        Kernel::create(&self.programs[program_idx], name)
            .map_err(|e| format!("kernel {name}: {e}"))
    }

    /// Establece un argumento de kernel (envoltura segura de `Kernel::set_arg`,
    /// que es `unsafe` en opencl3 0.12).
    fn set_arg<T>(&self, kernel: &Kernel, idx: usize, val: &T) -> Result<(), String> {
        unsafe { kernel.set_arg(idx as u32, val) }.map_err(|e| format!("arg {idx}: {e}"))
    }

    fn buffer<T>(&self, count: usize) -> Result<Buffer<T>, String> {
        unsafe { Buffer::<T>::create(&self.context, CL_MEM_READ_WRITE, count, ptr::null_mut()) }
            .map_err(|e| format!("buffer: {e}"))
    }

    fn write<T>(&self, buffer: &mut Buffer<T>, data: &[T]) -> Result<(), String> {
        unsafe {
            self.queue
                .enqueue_write_buffer(buffer, 1u32, 0, data, &[])
        }
        .map_err(|e| format!("write_buffer: {e}"))?;
        Ok(())
    }

    fn read<T: Copy + Default>(&self, buffer: &Buffer<T>, count: usize) -> Result<Vec<T>, String> {
        let mut data = vec![T::default(); count];
        unsafe {
            self.queue
                .enqueue_read_buffer(buffer, 1u32, 0, &mut data, &[])
        }
        .map_err(|e| format!("read_buffer: {e}"))?;
        Ok(data)
    }

    /// Envía un kernel 1D fragmentado en trozos encadenados por eventos
    /// (mitigación WDDM/TDR: ningún dispatch supera ~2 s).
    ///
    /// Los `Event`s se mantienen vivos en `events` hasta el final: `opencl3`
    /// llama a `clReleaseEvent` en el `Drop`, y esperar sobre un evento liberado
    /// cuelga el dispatch siguiente.
    fn enqueue_chunked(&self, kernel: &Kernel, total: usize, chunk: usize) -> Result<(), String> {
        let mut events: Vec<opencl3::event::Event> = Vec::new();
        let mut wait: Vec<cl_event> = Vec::new();
        let mut offset = 0usize;
        while offset < total {
            let size = chunk.min(total - offset);
            let off_arr = [offset];
            let size_arr = [size];
            let event = unsafe {
                self.queue
                    .enqueue_nd_range_kernel(
                        kernel.get(),
                        1,
                        off_arr.as_ptr(),
                        size_arr.as_ptr(),
                        ptr::null(),
                        &wait,
                    )
            }
            .map_err(|e| format!("enqueue_nd_range_kernel: {e}"))?;
            wait = vec![event.get()];
            events.push(event);
            offset += size;
        }
        drop(wait);
        self.queue.finish().map_err(|e| format!("finish: {e}"))?;
        Ok(())
    }

    /// Envía un kernel 2D (usado por `gram`).
    fn enqueue_2d(&self, kernel: &Kernel, rows: usize, cols: usize) -> Result<(), String> {
        let global = [rows, cols];
        unsafe {
            self.queue
                .enqueue_nd_range_kernel(kernel.get(), 2, ptr::null(), global.as_ptr(), ptr::null(), &[])
        }
        .map_err(|e| format!("enqueue_nd_range_kernel 2d: {e}"))?;
        self.queue.finish().map_err(|e| format!("finish: {e}"))?;
        Ok(())
    }

    /// Envía un kernel 2D fragmentado por FILAS (mitigación WDDM/TDR): la columna
    /// `cols` (= nº de candidatos de la población) permanece completa en cada
    /// dispatch; las filas se trocean encadenadas por eventos.
    fn enqueue_chunked_2d(
        &self,
        kernel: &Kernel,
        rows: usize,
        cols: usize,
        chunk: usize,
    ) -> Result<(), String> {
        let mut events: Vec<opencl3::event::Event> = Vec::new();
        let mut wait: Vec<cl_event> = Vec::new();
        let mut offset = 0usize;
        while offset < rows {
            let size = chunk.min(rows - offset);
            let off_arr = [offset, 0usize];
            let size_arr = [size, cols];
            let event = unsafe {
                self.queue.enqueue_nd_range_kernel(
                    kernel.get(),
                    2,
                    off_arr.as_ptr(),
                    size_arr.as_ptr(),
                    ptr::null(),
                    &wait,
                )
            }
            .map_err(|e| format!("enqueue_nd_range_kernel 2d: {e}"))?;
            wait = vec![event.get()];
            events.push(event);
            offset += size;
        }
        drop(wait);
        self.queue.finish().map_err(|e| format!("finish: {e}"))?;
        Ok(())
    }

    /// Ejecuta el decodificador CPPN para `d_in x d_out`.
    ///
    /// Devuelve `(w_dense, adjacency, active)` donde `w_dense` es la matriz
    /// `[d_out x d_in]` enmascarada (ceros donde `l_ij <= tau`), `adjacency` es
    /// el bit-tensor `ffn_dag_adjacency` y `active` el nº de conexiones.
    pub fn cppn_decode(
        &self,
        genome: &[f32],
        d_in: usize,
        d_out: usize,
        tau: f32,
        layer: usize,
        n_layers: usize,
    ) -> Result<(Vec<f32>, Vec<u8>, u32), String> {
        let total = d_in * d_out;
        let n_words = total.div_ceil(32);
        let n_bytes = total.div_ceil(8);
        let mut g = self.buffer::<f32>(genome.len())?;
        let mut w = self.buffer::<f32>(total)?;
        let mut a_words = self.buffer::<u32>(n_words)?;
        let mut act = self.buffer::<u32>(1)?;
        self.write(&mut g, genome)?;
        self.write(&mut act, &[0u32])?;
        self.write(&mut a_words, &vec![0u32; n_words])?;

        // Kernel 1: decode por conexión (máximo paralelismo), adyacencia en u32.
        let kernel = self.kernel(0, "cppn_decode")?;
        self.set_arg(&kernel, 0, &g)?;
        self.set_arg(&kernel, 1, &(d_in as i32))?;
        self.set_arg(&kernel, 2, &(d_out as i32))?;
        self.set_arg(&kernel, 3, &tau)?;
        self.set_arg(&kernel, 4, &(layer as i32))?;
        self.set_arg(&kernel, 5, &(n_layers as i32))?;
        self.set_arg(&kernel, 6, &w)?;
        self.set_arg(&kernel, 7, &a_words)?;
        self.set_arg(&kernel, 8, &act)?;
        self.enqueue_chunked(&kernel, total, CPPN_DECODE_CHUNK)?;

        // Kernel 2: empaquetado u32 -> bit-tensor u8 (LSB-first).
        let mut a = self.buffer::<u8>(n_bytes)?;
        let pack = self.kernel(0, "pack_adjacency")?;
        self.set_arg(&pack, 0, &a_words)?;
        self.set_arg(&pack, 1, &a)?;
        self.enqueue_chunked(&pack, n_words, CPPN_DECODE_CHUNK)?;

        let w_out = self.read(&w, total)?;
        let a_out = self.read(&a, n_bytes)?;
        let act_out = self.read(&act, 1)?;
        Ok((w_out, a_out, act_out[0]))
    }

    /// Decodifica **solo la adyacencia** (warm-start teacher-copy): no
    /// materializa la matriz densa `w` (evita ~805 MB de escritura GPU +
    /// lectura host en bloques ALIA/Qwen). Devuelve `(adjacency, active)`.
    pub fn cppn_decode_adjacency(
        &self,
        genome: &[f32],
        d_in: usize,
        d_out: usize,
        tau: f32,
        layer: usize,
        n_layers: usize,
    ) -> Result<(Vec<u8>, u32), String> {
        let total = d_in * d_out;
        let n_words = total.div_ceil(32);
        let n_bytes = total.div_ceil(8);
        let mut g = self.buffer::<f32>(genome.len())?;
        let mut a_words = self.buffer::<u32>(n_words)?;
        let mut act = self.buffer::<u32>(1)?;
        self.write(&mut g, genome)?;
        self.write(&mut act, &[0u32])?;
        self.write(&mut a_words, &vec![0u32; n_words])?;

        let kernel = self.kernel(0, "cppn_decode_adj")?;
        self.set_arg(&kernel, 0, &g)?;
        self.set_arg(&kernel, 1, &(d_in as i32))?;
        self.set_arg(&kernel, 2, &(d_out as i32))?;
        self.set_arg(&kernel, 3, &tau)?;
        self.set_arg(&kernel, 4, &(layer as i32))?;
        self.set_arg(&kernel, 5, &(n_layers as i32))?;
        self.set_arg(&kernel, 6, &a_words)?;
        self.set_arg(&kernel, 7, &act)?;
        self.enqueue_chunked(&kernel, total, CPPN_DECODE_CHUNK)?;

        let mut a = self.buffer::<u8>(n_bytes)?;
        let pack = self.kernel(0, "pack_adjacency")?;
        self.set_arg(&pack, 0, &a_words)?;
        self.set_arg(&pack, 1, &a)?;
        self.enqueue_chunked(&pack, n_words, CPPN_DECODE_CHUNK)?;

        let a_out = self.read(&a, n_bytes)?;
        let act_out = self.read(&act, 1)?;
        Ok((a_out, act_out[0]))
    }

    /// Decodifica **solo la adyacencia** de **N candidatos** en un único dispatch
    /// (Vía B — tensorización de la población). `genomes` debe tener `N * 466` f32
    /// (el genoma CPPN real; `CppnGenome::param_count()`). Rejilla 2D
    /// `[conexiones, N]`; el layout de salida es `[N][n_bytes]` (adyacencia por
    /// candidato) + `active[cand]`. Bit-exacto frente a N llamadas a
    /// [`Self::cppn_decode_adjacency`] (puerta de aceptación Fase 1).
    pub fn cppn_decode_adjacency_batched(
        &self,
        genomes: &[f32],          // N * 466
        n_candidates: usize,
        d_in: usize,
        d_out: usize,
        tau: f32,
        layer: usize,
        n_layers: usize,
    ) -> Result<(Vec<u8>, Vec<u32>), String> {
        let genome_len = 466usize;
        if genomes.len() != n_candidates * genome_len {
            return Err(format!(
                "cppn_decode_adjacency_batched: esperaba {n_candidates}×466 f32, hay {}",
                genomes.len()
            ));
        }
        let total = d_in * d_out;
        let n_words = total.div_ceil(32);
        let n_bytes = total.div_ceil(8);
        let mut g = self.buffer::<f32>(genomes.len())?;
        let mut a_words = self.buffer::<u32>(n_words * n_candidates)?;
        let mut act = self.buffer::<u32>(n_candidates)?;
        self.write(&mut g, genomes)?;
        self.write(&mut act, &vec![0u32; n_candidates])?;
        self.write(&mut a_words, &vec![0u32; n_words * n_candidates])?;

        let kernel = self.kernel(0, "cppn_decode_adj_batched")?;
        self.set_arg(&kernel, 0, &g)?;
        self.set_arg(&kernel, 1, &(d_in as i32))?;
        self.set_arg(&kernel, 2, &(d_out as i32))?;
        self.set_arg(&kernel, 3, &tau)?;
        self.set_arg(&kernel, 4, &(layer as i32))?;
        self.set_arg(&kernel, 5, &(n_layers as i32))?;
        self.set_arg(&kernel, 6, &(n_candidates as i32))?;
        self.set_arg(&kernel, 7, &a_words)?;
        self.set_arg(&kernel, 8, &act)?;
        self.enqueue_chunked_2d(&kernel, total, n_candidates, CPPN_DECODE_CHUNK)?;

        let mut a = self.buffer::<u8>(n_bytes * n_candidates)?;
        let pack = self.kernel(0, "pack_adjacency_batched")?;
        self.set_arg(&pack, 0, &a_words)?;
        self.set_arg(&pack, 1, &(n_words as i32))?;
        self.set_arg(&pack, 2, &(n_candidates as i32))?;
        self.set_arg(&pack, 3, &a)?;
        self.enqueue_chunked_2d(&pack, n_words, n_candidates, CPPN_DECODE_CHUNK)?;

        let a_out = self.read(&a, n_bytes * n_candidates)?;
        let act_out = self.read(&act, n_candidates)?;
        Ok((a_out, act_out))
    }

    /// SpMM del DAG en modo teacher-copy: construye el CSR **en GPU** desde la
    /// adyacencia + el tensor del profesor (D17), evitando el loop CPU O(N) por
    /// candidato y la transferencia host del CSR (ci+vals ≈ 1 GB en ALIA).
    pub fn spmm_csr_teacher(
        &self,
        x: &[f32],
        w0: &[f32],
        adj: &[u8],
        d_in: usize,
        d_out: usize,
    ) -> Result<Vec<f32>, String> {
        let batch = x.len() / d_in;
        let mut xb = self.buffer::<f32>(x.len())?;
        let mut wb = self.buffer::<f32>(w0.len())?;
        let mut adjb = self.buffer::<u8>(adj.len())?;
        let mut counts = self.buffer::<i32>(d_out)?;
        self.write(&mut xb, x)?;
        self.write(&mut wb, w0)?;
        self.write(&mut adjb, adj)?;

        // 1) Contar activos por fila en GPU.
        let count_k = self.kernel(1, "count_rows")?;
        self.set_arg(&count_k, 0, &adjb)?;
        self.set_arg(&count_k, 1, &(d_in as i32))?;
        self.set_arg(&count_k, 2, &(d_out as i32))?;
        self.set_arg(&count_k, 3, &counts)?;
        self.enqueue_chunked(&count_k, d_out, WDDM_CHUNK)?;
        let counts_h = self.read(&counts, d_out)?;

        // 2) Suma de prefijo en host (O(d_out), 96 KB en ALIA).
        let mut row_ptr = vec![0i32; d_out + 1];
        let mut acc = 0i32;
        for j in 0..d_out {
            row_ptr[j] = acc;
            acc += counts_h[j];
        }
        row_ptr[d_out] = acc;
        let nnz = acc as usize;
        if nnz == 0 {
            return Ok(vec![0.0f32; batch * d_out]);
        }

        let mut rpb = self.buffer::<i32>(row_ptr.len())?;
        let mut ci = self.buffer::<i32>(nnz)?;
        let mut vb = self.buffer::<f32>(nnz)?;
        let mut yb = self.buffer::<f32>(batch * d_out)?;
        self.write(&mut rpb, &row_ptr)?;

        // 3) Gather de los pesos del profesor en las posiciones activas (GPU).
        let gather_k = self.kernel(1, "gather_csr_teacher")?;
        self.set_arg(&gather_k, 0, &adjb)?;
        self.set_arg(&gather_k, 1, &wb)?;
        self.set_arg(&gather_k, 2, &rpb)?;
        self.set_arg(&gather_k, 3, &(d_in as i32))?;
        self.set_arg(&gather_k, 4, &(d_out as i32))?;
        self.set_arg(&gather_k, 5, &ci)?;
        self.set_arg(&gather_k, 6, &vb)?;
        self.enqueue_chunked(&gather_k, d_out, WDDM_CHUNK)?;

        // 4) SpMM CSR sobre buffers GPU (sin roundtrip host del CSR).
        let spmm_k = self.kernel(1, "spmm_csr")?;
        self.set_arg(&spmm_k, 0, &xb)?;
        self.set_arg(&spmm_k, 1, &rpb)?;
        self.set_arg(&spmm_k, 2, &ci)?;
        self.set_arg(&spmm_k, 3, &vb)?;
        self.set_arg(&spmm_k, 4, &(d_in as i32))?;
        self.set_arg(&spmm_k, 5, &(d_out as i32))?;
        self.set_arg(&spmm_k, 6, &yb)?;
        self.enqueue_chunked(&spmm_k, batch * d_out, WDDM_CHUNK)?;

        self.read(&yb, batch * d_out)
    }

    /// SpMM denso-enmascarado: `Y[b][j] = sum_i X[b][i] W[j][i]`.
    pub fn spmm_dense(
        &self,
        x: &[f32],
        w: &[f32],
        d_in: usize,
        d_out: usize,
    ) -> Result<Vec<f32>, String> {
        let batch = x.len() / d_in;
        let mut xb = self.buffer::<f32>(x.len())?;
        let mut wb = self.buffer::<f32>(w.len())?;
        let mut yb = self.buffer::<f32>(batch * d_out)?;
        self.write(&mut xb, x)?;
        self.write(&mut wb, w)?;

        let kernel = self.kernel(1, "spmm_dense")?;
        self.set_arg(&kernel, 0, &xb)?;
        self.set_arg(&kernel, 1, &wb)?;
        self.set_arg(&kernel, 2, &(d_in as i32))?;
        self.set_arg(&kernel, 3, &(d_out as i32))?;
        self.set_arg(&kernel, 4, &yb)?;

        self.enqueue_chunked(&kernel, batch * d_out, WDDM_CHUNK)?;
        self.read(&yb, batch * d_out)
    }

    /// SpMM CSR del DAG: `Y[b][j] = sum_k X[b][col_idx[k]] vals[k]`.
    #[allow(clippy::too_many_arguments)]
    pub fn spmm_csr(
        &self,
        x: &[f32],
        row_ptr: &[i32],
        col_idx: &[i32],
        vals: &[f32],
        d_in: usize,
        d_out: usize,
    ) -> Result<Vec<f32>, String> {
        let batch = x.len() / d_in;
        // Topología vacía (τ alto): OpenCL no admite buffers de tamaño 0.
        if col_idx.is_empty() || vals.is_empty() {
            return Ok(vec![0.0f32; batch * d_out]);
        }
        let mut xb = self.buffer::<f32>(x.len())?;
        let mut rp = self.buffer::<i32>(row_ptr.len())?;
        let mut ci = self.buffer::<i32>(col_idx.len())?;
        let mut vb = self.buffer::<f32>(vals.len())?;
        let mut yb = self.buffer::<f32>(batch * d_out)?;
        self.write(&mut xb, x)?;
        self.write(&mut rp, row_ptr)?;
        self.write(&mut ci, col_idx)?;
        self.write(&mut vb, vals)?;

        let kernel = self.kernel(1, "spmm_csr")?;
        self.set_arg(&kernel, 0, &xb)?;
        self.set_arg(&kernel, 1, &rp)?;
        self.set_arg(&kernel, 2, &ci)?;
        self.set_arg(&kernel, 3, &vb)?;
        self.set_arg(&kernel, 4, &(d_in as i32))?;
        self.set_arg(&kernel, 5, &(d_out as i32))?;
        self.set_arg(&kernel, 6, &yb)?;

        self.enqueue_chunked(&kernel, batch * d_out, WDDM_CHUNK)?;
        self.read(&yb, batch * d_out)
    }

    /// Matriz de Gram `K = H H^T` (B x B) para CKA.
    pub fn gram(&self, h: &[f32], d: usize, batch: usize) -> Result<Vec<f32>, String> {
        let mut hb = self.buffer::<f32>(h.len())?;
        let mut kb = self.buffer::<f32>(batch * batch)?;
        self.write(&mut hb, h)?;

        let kernel = self.kernel(2, "gram")?;
        self.set_arg(&kernel, 0, &hb)?;
        self.set_arg(&kernel, 1, &(d as i32))?;
        self.set_arg(&kernel, 2, &(batch as i32))?;
        self.set_arg(&kernel, 3, &kb)?;

        self.enqueue_2d(&kernel, batch, batch)?;
        self.read(&kb, batch * batch)
    }
}

/// Máxima diferencia absoluta entre dos vectores f32 (para validación).
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

