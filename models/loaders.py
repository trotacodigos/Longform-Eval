from pathlib import Path
from typing import Optional, List
import yaml

from . import REGISTRY
from .base import BaseModel
from .base_openai import OpenAIModel
from .base_ollama import OllamaModel
from .base_hf import HFChatModel
from .sampling import SamplingParams, OpenAIParams, OllamaParams, HuggingFaceParams, ClaudeParams


BACKEND_PARAMS = {
    "openai": OpenAIParams,
    "ollama": OllamaParams,
    "hf": HuggingFaceParams,
    "claude": ClaudeParams,

}


def load_models_from_yaml(cfg_path: Path, select_names: Optional[List[str]] = None) -> List[BaseModel]:
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    items = cfg.get("models", [])
    models: List[BaseModel] = []

    for m in items:
        name = m["name"]
        if select_names and name not in select_names:
            continue

        backend = m.get("backend")
        sampling_cfg = m.get("sampling_params") or {}
        params_cls = BACKEND_PARAMS.get(backend, SamplingParams)
        sampling_params = sampling_cfg if isinstance(sampling_cfg, SamplingParams) else params_cls(**sampling_cfg)

        # 1) Apply registered models
        if name in REGISTRY:
            cls = REGISTRY[name]
            kwargs = {}
            for k in ("model_id", "endpoint", "tgt_lang", "host", "stop"):
                if k in m:
                    kwargs[k] = m[k]
            try:
                model = cls(sampling_params=sampling_params, **kwargs)
            except TypeError:
                model = cls(sampling_params=sampling_params)
                for k, v in kwargs.items():
                    setattr(model, k, v)

            if not hasattr(model, "name"):
                setattr(model, "name", name)
            if "model_id" in m:
                setattr(model, "model_id", m["model_id"])

            models.append(model)
            continue

        # 2) Does not exist in REGISTRY
        model_id = m.get("model_id")

        if backend == "openai":
            model = OpenAIModel(name=name, model_id=model_id, sampling_params=sampling_params)

        elif backend == "ollama":
            model = OllamaModel(name=name, model_id=model_id, sampling_params=sampling_params, host=m.get("host"))

        elif backend == "hf":
            endpoint = m.get("endpoint")
            if not endpoint:
                raise ValueError(f"[{name}] backend=hf requires 'endpoint'")
            model = HFChatModel(name=name, model_id=model_id, sampling_params=sampling_params, endpoint=endpoint)

        else:
            raise ValueError(f"Unknown backend: {backend} (model: {name})")

        models.append(model)

    if not models:
        raise RuntimeError("No models loaded from config (check names/backend).")
    return models