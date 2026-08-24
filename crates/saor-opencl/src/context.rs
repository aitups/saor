//! Descubrimiento de plataformas y dispositivos OpenCL.
//!
//! Valida que la RTX 4050 sea visible vía OpenCL 3.0 y reporta su VRAM y la
//! versión del compilador OpenCL C, datos necesarios para el presupuesto de
//! memoria del motor de streaming (~2 GB) y la mitigación WDDM/TDR.

use opencl3::device::{Device, CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU};
use opencl3::platform::{get_platforms, Platform};

/// Información resumida de un dispositivo OpenCL.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Nombre de la plataforma OpenCL (p. ej. NVIDIA CUDA).
    pub platform_name: String,
    /// Proveedor de la plataforma.
    pub platform_vendor: String,
    /// Versión de la plataforma OpenCL.
    pub platform_version: String,
    /// Nombre del dispositivo (p. ej. "NVIDIA GeForce RTX 4050 Laptop GPU").
    pub device_name: String,
    /// Proveedor del dispositivo.
    pub device_vendor: String,
    /// Clasificación del dispositivo: "GPU", "CPU" u "OTHER".
    pub device_type: String,
    /// Memoria global total (VRAM) en bytes.
    pub global_mem_bytes: u64,
    /// Tamaño máximo de un solo buffer de memoria en bytes.
    pub max_mem_alloc_bytes: u64,
    /// Versión de OpenCL C soportada (p. ej. "OpenCL C 3.0").
    pub opencl_c_version: String,
    /// Tamaño máximo de un work-group (base para la mitigación WDDM/TDR).
    pub max_work_group_size: usize,
}

impl DeviceInfo {
    /// VRAM total en MiB.
    pub fn global_mem_mib(&self) -> f64 {
        self.global_mem_bytes as f64 / (1024.0 * 1024.0)
    }
}

/// Devuelve un vector con todos los dispositivos OpenCL del sistema.
pub fn discover_devices() -> Result<Vec<DeviceInfo>, String> {
    let platforms = get_platforms().map_err(|e| format!("get_platforms: {e}"))?;
    if platforms.is_empty() {
        return Err("no se encontraron plataformas OpenCL".into());
    }

    let mut out = Vec::new();
    for platform in &platforms {
        let device_ids = platform
            .get_devices(CL_DEVICE_TYPE_ALL)
            .map_err(|e| format!("get_devices: {e}"))?;
        for id in device_ids {
            let device = Device::new(id);
            let dev_type = device.dev_type().unwrap_or(0);
            let device_type = if dev_type & CL_DEVICE_TYPE_GPU != 0 {
                "GPU"
            } else if dev_type & CL_DEVICE_TYPE_CPU != 0 {
                "CPU"
            } else {
                "OTHER"
            };
            out.push(DeviceInfo {
                platform_name: platform.name().unwrap_or_default(),
                platform_vendor: platform.vendor().unwrap_or_default(),
                platform_version: platform.version().unwrap_or_default(),
                device_name: device.name().unwrap_or_default(),
                device_vendor: device.vendor().unwrap_or_default(),
                device_type: device_type.into(),
                global_mem_bytes: device.global_mem_size().unwrap_or(0),
                max_mem_alloc_bytes: device.max_mem_alloc_size().unwrap_or(0),
                opencl_c_version: device.opencl_c_version().unwrap_or_default(),
                max_work_group_size: device.max_work_group_size().unwrap_or(0),
            });
        }
    }
    Ok(out)
}

/// Primera GPU del sistema (la RTX 4050).
pub fn first_gpu() -> Result<DeviceInfo, String> {
    discover_devices()?
        .into_iter()
        .find(|d| d.device_type == "GPU")
        .ok_or_else(|| "no se encontró ninguna GPU OpenCL".to_string())
}

/// Devuelve la plataforma y el `Device` de la primera GPU, prefiriendo NVIDIA
/// cuando hay varias (la máquina también expone la iGPU Intel).
pub fn first_gpu_device() -> Result<(Platform, Device), String> {
    let platforms = get_platforms().map_err(|e| format!("get_platforms: {e}"))?;
    for platform in &platforms {
        let device_ids = platform
            .get_devices(CL_DEVICE_TYPE_GPU)
            .map_err(|e| format!("get_devices: {e}"))?;
        let mut nvidia: Option<Device> = None;
        let mut first: Option<Device> = None;
        for id in device_ids {
            let device = Device::new(id);
            let vendor = device.vendor().unwrap_or_default();
            if vendor.to_ascii_lowercase().contains("nvidia") {
                nvidia = Some(device);
                break;
            }
            if first.is_none() {
                first = Some(device);
            }
        }
        if let Some(d) = nvidia {
            return Ok((platform.clone(), d));
        }
        if let Some(d) = first {
            return Ok((platform.clone(), d));
        }
    }
    Err("no se encontró ninguna GPU OpenCL".to_string())
}

/// Imprime un resumen legible de los dispositivos.
pub fn print_devices(infos: &[DeviceInfo]) {
    for d in infos {
        println!("=== {} ===", d.device_name);
        println!(
            "  platform : {} ({}) — {}",
            d.platform_name, d.platform_vendor, d.platform_version
        );
        println!("  vendor   : {}", d.device_vendor);
        println!("  type     : {}", d.device_type);
        println!(
            "  VRAM     : {:.2} MiB (alloc máx. {:.2} MiB)",
            d.global_mem_mib(),
            d.max_mem_alloc_bytes as f64 / (1024.0 * 1024.0)
        );
        println!("  OpenCL C : {}", d.opencl_c_version);
        println!("  max WG   : {}", d.max_work_group_size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hay_al_menos_un_dispositivo() {
        // El entorno tiene OpenCL.dll + RTX 4050; la prueba es informativa y
        // tolerante a entornos sin OpenCL (CI).
        if let Ok(infos) = discover_devices() {
            assert!(!infos.is_empty());
        }
    }
}

