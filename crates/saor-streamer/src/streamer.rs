//! `RustStreamer` — motor de streaming capa a capa con doble buffer.
//!
//! Implementa el Paso 3 de la propuesta: un hilo secundario copia la capa
//! `l+1` (prefetch) mientras la capa `l` se procesa (cómputo en la GPU a
//! partir de la Fase 3), manteniendo un pico de huella = 2 buffers y
//! respetando el presupuesto de VRAM (~2 GB).
//!
//! El closure de cómputo recibe además un contador atómico con los bytes
//! actualmente en prefetch (útil para backpressure y monitoreo WDDM).
//! En esta fase el "cómputo" es un closure que simula el trabajo del kernel;
//! la Fase 3 sustituye el closure por la ejecución de kernels OpenCL.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;

/// Estadísticas de una corrida de streaming.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StreamStats {
    /// Capas procesadas.
    pub layers_streamed: usize,
    /// Pico de bytes en vuelo (buffer actual + buffer precargado).
    pub peak_bytes: u64,
    /// Bytes totales transferidos.
    pub total_bytes: u64,
}

/// Motor de streaming con doble buffer.
pub struct RustStreamer {
    budget_bytes: u64,
}

impl RustStreamer {
    /// Crea el motor con un presupuesto de pico de VRAM en bytes.
    pub fn new(budget_bytes: u64) -> Self {
        Self { budget_bytes }
    }

    /// Streams de los `chunks` (pesos de capas en RAM) ejecutando `compute`
    /// sobre cada capa en orden, mientras el hilo de prefetch copia la
    /// siguiente al buffer en anillo.
    ///
    /// El closure `compute(layer_index, bytes, prefetch_in_flight)` recibe el
    /// índice de capa, el búfer actual y los bytes en prefetch; debe devolver
    /// `Err` si el cómputo falla.
    pub fn stream<C>(&self, chunks: &[Arc<[u8]>], compute: C) -> Result<StreamStats, String>
    where
        C: Fn(usize, &[u8], &AtomicU64) -> Result<(), String>,
    {
        let n = chunks.len();
        if n == 0 {
            return Ok(StreamStats::default());
        }
        let max_chunk = chunks.iter().map(|c| c.len()).max().unwrap_or(0);
        let footprint = 2 * max_chunk;
        if footprint > self.budget_bytes as usize {
            return Err(format!(
                "el doble buffer necesita {footprint} B, presupuesto {} B",
                self.budget_bytes
            ));
        }

        let prefetch_bytes = Arc::new(AtomicU64::new(0));
        // Carga inicial de la capa 0.
        let mut current: Vec<u8> = chunks[0].as_ref().to_vec();
        let mut next_idx = 1usize;
        let mut peak = current.len() as u64;
        let mut stats = StreamStats::default();

        while next_idx < n {
            // Prefetch asíncrono: la copia pesada ocurre DENTRO del hilo.
            let src = chunks[next_idx].clone();
            let len = src.len() as u64;
            let pb = prefetch_bytes.clone();
            let (tx, rx) = mpsc::channel::<Vec<u8>>();
            let handle = thread::spawn(move || {
                pb.store(len, Ordering::SeqCst);
                let copied: Vec<u8> = src.as_ref().to_vec();
                let _ = tx.send(copied);
                pb.store(0, Ordering::SeqCst);
            });

            // Cómputo de la capa l, solapado con el prefetch de l+1.
            let cur = &current;
            compute(next_idx - 1, cur, &prefetch_bytes)?;
            stats.layers_streamed += 1;
            stats.total_bytes += cur.len() as u64;

            let prefetched = rx.recv().map_err(|_| "hilo de prefetch falló".to_string())?;
            handle.join().map_err(|_| "hilo de prefetch falló".to_string())?;
            peak = peak.max(cur.len() as u64 + prefetched.len() as u64);
            current = prefetched;
            next_idx += 1;
        }

        // Última capa (sin prefetch en vuelo).
        compute(next_idx - 1, &current, &prefetch_bytes)?;
        stats.layers_streamed += 1;
        stats.total_bytes += current.len() as u64;
        stats.peak_bytes = peak;
        Ok(stats)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn arc_chunks(sizes: &[usize]) -> Vec<Arc<[u8]>> {
        sizes
            .iter()
            .enumerate()
            .map(|(i, &s)| Arc::from(vec![i as u8; s]))
            .collect()
    }

    #[test]
    fn stream_procesa_todas_las_capas_en_orden() {
        let streamer = RustStreamer::new(1024 * 1024);
        let chunks = arc_chunks(&[128; 5]);
        let order = std::sync::Mutex::new(Vec::new());
        let stats = streamer
            .stream(&chunks, |idx, bytes, _| {
                assert_eq!(bytes[0], idx as u8, "capa fuera de orden");
                order.lock().unwrap().push(idx);
                Ok(())
            })
            .expect("stream");
        assert_eq!(stats.layers_streamed, 5);
        assert_eq!(stats.total_bytes, 5 * 128);
        assert_eq!(stats.peak_bytes, 256); // 2 buffers de 128
        assert_eq!(*order.lock().unwrap(), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn stream_respeta_presupuesto() {
        let streamer = RustStreamer::new(256); // 2 buffers de 128 = 256 OK
        let chunks = arc_chunks(&[128; 3]);
        let stats = streamer
            .stream(&chunks, |_, _, _| Ok(()))
            .expect("stream");
        assert!(stats.peak_bytes <= 256);

        let estricto = RustStreamer::new(200);
        let err = estricto
            .stream(&chunks, |_, _, _| Ok(()))
            .expect_err("debe rechazar el presupuesto");
        assert!(err.contains("doble buffer"));
    }

    #[test]
    fn stream_prefetch_ocurre_durante_el_compute() {
        let streamer = RustStreamer::new(64 * 1024 * 1024);
        // 4 capas de 1 MiB; las capas 0..3 (que tienen siguiente) deben ver el
        // prefetch concurrente vía el contador atómico de bytes en vuelo.
        let chunks = arc_chunks(&[1 << 20; 4]);
        let stats = streamer
            .stream(&chunks, |idx, _bytes, in_flight| {
                if idx < 3 {
                    let deadline = Instant::now() + Duration::from_secs(5);
                    while Instant::now() < deadline {
                        if in_flight.load(Ordering::SeqCst) > 0 {
                            return Ok(());
                        }
                        thread::yield_now();
                    }
                    panic!("el prefetch de la capa siguiente no se solapó con el cómputo");
                }
                Ok(())
            })
            .expect("stream");
        assert_eq!(stats.layers_streamed, 4);
        assert!(stats.peak_bytes <= 2 * (1 << 20));
    }

    #[test]
    fn stream_vacio_devuelve_estadisticas_vacias() {
        let streamer = RustStreamer::new(1024);
        let chunks: Vec<Arc<[u8]>> = Vec::new();
        let stats = streamer.stream(&chunks, |_, _, _| Ok(())).expect("ok");
        assert_eq!(stats.layers_streamed, 0);
    }
}
