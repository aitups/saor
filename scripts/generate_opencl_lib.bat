@echo off
rem Genera OpenCL.lib (import lib x64) desde el OpenCL.dll del sistema, en
rem .local\windows-lib. Requiere MSVC Build Tools (vcvars64) y dumpbin/lib.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
  echo [opencl-lib] ERROR: no se pudo configurar vcvars64
  exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0generate_opencl_lib.ps1"
exit /b %ERRORLEVEL%
