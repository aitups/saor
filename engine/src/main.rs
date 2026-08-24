//! # saor-engine
//!
//! Binario CLI/IPC del experimento. En Fase 0 expone `device-info` (descubrimiento
//! OpenCL). En fases posteriores expondrá los subcomandos del pipeline evolutivo
//! y un protocolo JSON-lines sobre stdio para el orquestador Python.

use std::process::ExitCode;

use serde::Serialize;

mod kernels_run;

/// Resultado de `device-info` en formato JSON (consumible por Python).
#[derive(Serialize)]
struct DeviceInfoReport {
    ok: bool,
    devices: Vec<DeviceJson>,
}

#[derive(Serialize)]
struct DeviceJson {
    name: String,
    vendor: String,
    device_type: String,
    global_mem_mib: f64,
    opencl_c_version: String,
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("device-info") => cmd_device_info(),
        Some("kernels-run") => kernels_run::cmd(&args[2..]),
        Some("version") => {
            println!(
                "saor-engine {} (domain {}, streamer {}, opencl)",
                env!("CARGO_PKG_VERSION"),
                saor_domain::VERSION,
                saor_streamer::VERSION
            );
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("saor-engine: comando desconocido '{other}'");
            eprintln!("uso: saor-engine <device-info|kernels-run|version>");
            ExitCode::from(2)
        }
        None => {
            eprintln!("uso: saor-engine <device-info|kernels-run|version>");
            ExitCode::from(2)
        }
    }
}

fn cmd_device_info() -> ExitCode {
    match saor_opencl::context::discover_devices() {
        Ok(infos) => {
            saor_opencl::context::print_devices(&infos);
            // Reporte JSON a stdout para el orquestador Python.
            let report = DeviceInfoReport {
                ok: true,
                devices: infos
                    .iter()
                    .map(|d| DeviceJson {
                        name: d.device_name.clone(),
                        vendor: d.device_vendor.clone(),
                        device_type: d.device_type.clone(),
                        global_mem_mib: d.global_mem_mib(),
                        opencl_c_version: d.opencl_c_version.clone(),
                    })
                    .collect(),
            };
            let json = serde_json::to_string(&report).expect("serializable");
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("saor-engine: error descubriendo dispositivos OpenCL: {e}");
            println!(
                "{}",
                serde_json::json!({ "ok": false, "error": e }).to_string()
            );
            ExitCode::FAILURE
        }
    }
}
