from pathlib import Path
from typing import Optional, List
import yaml

from . import REGISTRY
from .base import BaseModel


def load_models_from_yaml(cfg_path: Path, select_names: Optional[List[str]] = None) -> List[BaseModel]:
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    items = cfg.get("models", [])
    models: List[BaseModel] = []

    for m in items:
        name = m["name"]
        if select_names and name not in select_names:
            continue

        if name not in REGISTRY:
            raise ValueError(f"Unknown model: {name}. Register it in REGISTRY first.")

        cls = REGISTRY[name]
        kwargs = {}
        for key in ("model_id", "endpoint", "tgt_lang", "host"):
            if key in m:
                kwargs[key] = m[key]

        model = cls(**kwargs)
        models.append(model)

    if not models:
        raise RuntimeError("No models loaded from config.")
    return models