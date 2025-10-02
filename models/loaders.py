from pathlib import Path
from typing import Optional, List
import yaml

from . import REGISTRY
from .basemodel import BaseModel, Decoding, OpenAIModel, OllamaModel, HFChatModel

def load_models_from_yaml(cfg_path: Path, select_names: Optional[List[str]] = None) -> List[BaseModel]:
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    items = cfg.get("models", [])
    models: List[BaseModel] = []

    for m in items:
        name = m["name"]
        if select_names and name not in select_names:
            continue

        decoding_cfg = m.get("decoding") or {}
        decoding = decoding_cfg if isinstance(decoding_cfg, Decoding) else Decoding(**decoding_cfg)

        # 1) Apply registered models:
        if name in REGISTRY:
            cls = REGISTRY[name]

            kwargs = {}
            for k in ("model_id", "endpoint", "tgt_lang", "host", "stop"):
                if k in m:
                    kwargs[k] = m[k]
            try:
                model = cls(decoding=decoding, **kwargs)
            except TypeError:
                model = cls(decoding=decoding)
                for k, v in kwargs.items():
                    setattr(model, k, v)

            if not hasattr(model, "name"):
                setattr(model, "name", name)
            if "model_id" in m:
                setattr(model, "model_id", m["model_id"])

            models.append(model)
            continue

        # 2) Does not exist in REGISTRY
        backend = m.get("backend")
        model_id = m.get("model_id")

        if backend == "openai":
            model = OpenAIModel(name=name, model_id=model_id, decoding=decoding)

        elif backend == "ollama":
            host = m.get("host")
            model = OllamaModel(name=name, model_id=model_id, decoding=decoding, host=host)

        elif backend == "hf":
            endpoint = m.get("endpoint")
            if not endpoint:
                raise ValueError(f"[{name}] backend=hf requires 'endpoint'")
            model = HFChatModel(
                name=name,
                model_id=model_id,
                decoding=decoding,
                endpoint=endpoint,
            )

        else:
            raise ValueError(f"Unknown backend: {backend} (model: {name})")

        models.append(model)

    if not models:
        raise RuntimeError("No models loaded from config (check names/backend).")
    return models