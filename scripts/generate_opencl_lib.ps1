# Genera OpenCL.lib (import lib) a partir del OpenCL.dll del sistema.
# El crate `cl-sys` (dependencia de opencl3) enlaza `OpenCL.lib`, que no se
# distribuye con el driver NVIDIA; esta librería de importación se obtiene
# volcando los exports de C:\Windows\System32\OpenCL.dll (ICD loader Khronos).

$ErrorActionPreference = "Stop"
$vendor = Join-Path $PSScriptRoot "..\.local\windows-lib"
New-Item -ItemType Directory -Path $vendor -Force | Out-Null

$dll = "C:\Windows\System32\OpenCL.dll"
if (-not (Test-Path $dll)) {
    throw "no se encontró $dll"
}

$exportsFile = Join-Path $vendor "exports.txt"
& dumpbin /exports $dll | Out-File -Encoding ascii $exportsFile
if ($LASTEXITCODE -ne 0) { throw "dumpbin falló" }

# Parseo de líneas "ordinal hint RVA name" de dumpbin.
$names = @()
foreach ($line in Get-Content $exportsFile) {
    if ($line -match '^\s*[0-9A-F]+\s+[0-9A-F]+\s+[0-9A-F]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*$') {
        $names += $Matches[1]
    }
}
if ($names.Count -eq 0) {
    throw "no se parsearon exports de OpenCL.dll (revisa $exportsFile)"
}

$def = @("LIBRARY OpenCL", "EXPORTS") + $names
$defFile = Join-Path $vendor "opencl.def"
$def | Set-Content -Encoding ascii $defFile

$libFile = Join-Path $vendor "OpenCL.lib"
& lib /machine:x64 "/def:$defFile" "/out:$libFile" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "lib.exe falló al crear OpenCL.lib" }

Write-Host "OpenCL.lib generado en $libFile ($($names.Count) exports)"
