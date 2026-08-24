"""Hooks sobre el modelo real (Pasos 1–2 del diseño del experimento).

Contenido planificado (Fase 5):
- gguf_audit.py   — auditoría de cabeceras GGUF del modelo base sin cargarlo.
- role_catalog.py — catálogo de roles para inicialización estricta por regex.
- calibration.py  — lote B=128 (código + matemáticas + prosa) y captura X/H0.
"""
