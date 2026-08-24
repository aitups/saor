@echo off
rem Entorno de desarrollo MSVC para Windows: configura vcvars64, añade el
rem import lib local de OpenCL al LIB y ejecuta cargo.
rem Uso: build_dev.bat <args de cargo>  (ej: build_dev.bat build --workspace)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
  echo [build_dev] ERROR: no se pudo configurar vcvars64
  exit /b 1
)
rem Import lib local generado por scripts\generate_opencl_lib.bat
set "VENDOR_DIR=%~dp0..\.local\windows-lib"
if exist "%VENDOR_DIR%\OpenCL.lib" set "LIB=%VENDOR_DIR%;%LIB%"
cargo %*
