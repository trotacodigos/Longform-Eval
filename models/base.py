from .tools import rough_token_count, _drop_none

import os, time
import copy
import csv
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Decoding:
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 1024
    stop: Optional[List[str]] = None
    num_ctx: Optional[int] = None               # Ollama context size
    frequency_penalty: Optional[float] = None   # OpenAI
    presence_penalty: Optional[float] = None    # OpenAI
    repetition_penalty: Optional[float] = None  # HF generate 계열
    min_p: Optional[float] = None


class BaseModel:
    def __init__(self, name: str, model_id: str, decoding: dict | Decoding | None):
        self.name = name
        self.model_id = model_id
        self.decoding = decoding if isinstance(decoding, Decoding) else Decoding(**(decoding or {}))

    def _call(self, system: str, user: str) -> tuple:
        raise NotImplementedError

    def generate(self, system: str, user: str, log_path: str = "inference_result.csv"):
        t0 = time.perf_counter()
        text, in_token, out_token = self._call(system, user)
        latency = time.perf_counter() - t0

        if in_token is None:
            in_token = rough_token_count(user)
        if out_token is None:
            out_token = rough_token_count(text)

        # tokens per second
        tps = out_token / latency if latency > 0 else 0.0

        metrics = {
            "model_name": self.name,
            "input_token": int(in_token),
            "output_token": int(out_token),
            "latency_sec": round(float(latency), 4), # end-to-end latency
            "tps": round(float(tps), 2)
        }    

        self._log_to_csv(metrics, log_path)

        return text, metrics
    
    def _log_to_csv(self, metrics: dict, log_path: str):
        file_exists = os.path.isfile(log_path)

        with open(log_path, mode="a", newline="", encoding="utf-8") as f:
            fieldnames = ["model_name", "input_token", "output_token", "latency_sec", "tps"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(metrics)

    
    def with_decoding(self, **overrides):
        new_model = copy.copy(self)
        new_model.decoding = copy.deepcopy(self.decoding)

        for k, v in overrides.items():
            if hasattr(new_model.decoding, k):
                setattr(new_model.decoding, k, v)
        return new_model
