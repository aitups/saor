"""Cliente IPC (JSON-lines por stdio) hacia el binario `saor-engine`.

Ruta principal: PyO3 (cuando el build de la extensión esté disponible). Este
cliente es el *fallback* documentado y además sirve para inspección manual
durante el desarrollo.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class SaorEngineClient:
    """Ejecuta subcomandos de `saor-engine` y parsea su salida JSON."""

    def __init__(self, engine_bin: str | Path | None = None) -> None:
        self.engine_bin = str(engine_bin) if engine_bin is not None else self._locate()

    @staticmethod
    def _locate() -> str:
        """Busca `saor-engine(.exe)` en el PATH o en `target/release|debug`."""
        found = shutil.which("saor-engine")
        if found:
            return found
        repo = Path(__file__).resolve().parents[2]
        for sub in ("release", "debug"):
            cand = repo / "target" / sub / "saor-engine.exe"
            if cand.exists():
                return str(cand)
        raise FileNotFoundError(
            "no se encontró saor-engine; compílalo con `cargo build` primero"
        )

    def run(self, *args: str) -> dict:
        proc = subprocess.run(
            [self.engine_bin, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"saor-engine {args!r} falló: {proc.stderr}")
        # La última línea JSON es el reporte estructurado.
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        raise RuntimeError(f"saor-engine no emitió JSON: {proc.stdout!r}")

    def device_info(self) -> dict:
        return self.run("device-info")

    def version(self) -> str:
        proc = subprocess.run(
            [self.engine_bin, "version"], capture_output=True, text=True, timeout=30
        )
        return proc.stdout.strip()
