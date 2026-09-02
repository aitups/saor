"""Sube los 3 modelos SAOR a Hugging Face como una colección.

Crea los repos <ns>/<modelo>-saor, sube el GGUF + README (model card) + perfil de
esparsidad, y crea/actualiza la colección `saor-gguf`.

Uso: python upload_hf_saor.py  (lee HF_TOKEN del .env)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub import Collection

STAGING = Path(r"d:\Documents\pySrc\.scratch\hf_saor")
ENV = Path(r"d:\Documents\pySrc\saor\.env")
NAMESPACE = "aitups"

MODELS = [
    "Qwen3.8-27B-saor",
    "Qwen3.5-4B-saor",
    "ALIA-40b-saor",
]


def load_token() -> str:
    for line in ENV.read_text(encoding="utf-8").splitlines():
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("HF_TOKEN no encontrado en .env")


def main() -> int:
    token = load_token()
    api = HfApi(token=token)
    print(f"[hf] usuario: {api.whoami()['name']}", flush=True)

    repo_ids = []
    for model in MODELS:
        src = STAGING / model
        if not (src / "README.md").exists():
            print(f"[hf] SKIP {model}: sin README.md", flush=True)
            continue
        repo_id = f"{NAMESPACE}/{model}"
        try:
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            print(f"[hf] repo {repo_id} ok", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[hf] create_repo {repo_id} fallo: {exc}", flush=True)
            return 1

        for f in src.iterdir():
            if f.is_file():
                try:
                    api.upload_file(
                        path_or_fileobj=str(f),
                        path_in_repo=f.name,
                        repo_id=repo_id,
                        repo_type="model",
                    )
                    print(f"[hf] subido {model}/{f.name}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[hf] upload {model}/{f.name} fallo: {exc}", flush=True)
                    return 1
        repo_ids.append(repo_id)

    # Colección.
    title = "SAOR sparse GGUF models (magnitude-pruned)"
    description = (
        "Modelos GGUF con FFN podado por magnitud + perfil de densidad por capa "
        "optimizado por CMA-ES (SAOR). Formato D16 (ffn_dag_adjacency + "
        "ffn_dag_weights, pesos activos Q4_K). Ejecución con Hayai "
        "(hayai generate --model <file>)."
    )
    col = api.create_collection(
        title=title,
        namespace=NAMESPACE,
        description=description,
        exists_ok=True,
    )
    col_id = col.slug if hasattr(col, "slug") else col.name
    for repo_id in repo_ids:
        api.add_collection_item(
            collection_slug=col_id,
            item_id=repo_id,
            item_type="model",
            exists_ok=True,
        )
        print(f"[hf] coleccion {col_id}: añadido {repo_id}", flush=True)
    print(f"[hf] colección: https://huggingface.co/collections/{NAMESPACE}/{col_id}", flush=True)
    print("[hf] LISTO", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
