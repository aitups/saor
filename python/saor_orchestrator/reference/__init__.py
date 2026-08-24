"""Referencia NumPy del dominio (ground-truth para validar kernels OpenCL).

Contenido planificado (Fase 1):
- cppn.py          — genoma indirecto CPPN (8 dims, activaciones heterogéneas).
- topology.py      — instanciación DAG + máscara de esparsidad τ.
- cka.py           — matrices de Gram y CKA centrado (HSIC).
- cmaes.py         — CMA-ES en subespacio activo + seed replay.
- reconciler.py    — subsampling por índices calientes + proyección identidad.
- arch_distance.py — Hamming normalizada / sparsity del candidato.
"""
