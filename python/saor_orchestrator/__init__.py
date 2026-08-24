"""Orquestador Python del experimento de optimización no dirigida.

En Fase 0 provee el esqueleto del paquete y el cliente IPC hacia el binario
`saor-engine`. En Fase 1 contiene la referencia NumPy (ground-truth) de toda la
matemática del dominio, que valida los kernels OpenCL de Fase 3.
"""

from ._version import __version__
from .ipc import SaorEngineClient

__all__ = ["__version__", "SaorEngineClient"]
