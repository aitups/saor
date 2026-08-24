//! `PinnedMemoryAllocator` — RAM page-locked para mapeo PCIe directo.
//!
//! Reserva contigua con `VirtualAlloc` y la fija en memoria con `VirtualLock`
//! (best-effort: puede requerir privilegios). Evita fallos de página durante el
//! streaming de pesos cuantizados por PCIe hacia la VRAM (Paso 3 de la
//! propuesta).

use std::ptr;

use windows_sys::Win32::System::Memory::{
    VirtualAlloc, VirtualFree, VirtualLock, VirtualUnlock, MEM_COMMIT, MEM_RELEASE,
    MEM_RESERVE, PAGE_READWRITE,
};

/// Tamaño de página de Windows (x64).
pub const PAGE_SIZE: usize = 4096;

fn align_up(n: usize, align: usize) -> usize {
    (n + align - 1) / align * align
}

/// Búfer de memoria page-locked.
///
/// Al hacer `Drop` libera las páginas y las desbloquea.
pub struct PinnedBuffer {
    ptr: *mut u8,
    len_bytes: usize,
    alloc_bytes: usize,
}

// El puntero es exclusivo del búfer; el tipo es seguro si se accede vía las
// vistas con lifetime que proveen `as_slice`/`as_mut_slice`.
unsafe impl Send for PinnedBuffer {}

impl PinnedBuffer {
    /// Reserva `len_bytes` de RAM page-locked (redondeado a páginas de 4 KiB).
    pub fn allocate(len_bytes: usize) -> Result<Self, String> {
        let alloc_bytes = align_up(len_bytes.max(1), PAGE_SIZE);
        let ptr = unsafe {
            VirtualAlloc(
                ptr::null_mut(),
                alloc_bytes,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            )
        };
        if ptr.is_null() {
            return Err(format!("VirtualAlloc falló para {alloc_bytes} bytes"));
        }
        // VirtualLock es best-effort (puede fallar sin privilegios de se-lock).
        let _ = unsafe { VirtualLock(ptr, alloc_bytes) };
        Ok(Self {
            ptr: ptr as *mut u8,
            len_bytes,
            alloc_bytes,
        })
    }

    /// Bytes realmente solicitados (no redondeados).
    pub fn len(&self) -> usize {
        self.len_bytes
    }

    /// Vista inmutable de la memoria.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len_bytes) }
    }

    /// Vista mutable de la memoria.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len_bytes) }
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        unsafe {
            let _ = VirtualUnlock(self.ptr as *const _, self.alloc_bytes);
            VirtualFree(self.ptr as *mut _, 0, MEM_RELEASE);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_escribe_y_lee() {
        let mut buf = PinnedBuffer::allocate(8192).expect("VirtualAlloc debe funcionar");
        assert_eq!(buf.len(), 8192);
        buf.as_mut_slice()[0] = 42;
        buf.as_mut_slice()[8191] = 7;
        assert_eq!(buf.as_slice()[0], 42);
        assert_eq!(buf.as_slice()[8191], 7);
    }

    #[test]
    fn allocate_redondea_a_paginas() {
        let buf = PinnedBuffer::allocate(1).expect("alloc");
        assert_eq!(buf.alloc_bytes, PAGE_SIZE);
    }
}
