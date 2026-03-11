from tools import rough_token_count
from sampling import SamplingParams

import os, time
import copy
import csv
import dataclasses
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract base class for all model implementations.
    Subclasses must implement _call.
    """
    def __init__(self, name: str, model_id: str, sampling_params: dict | SamplingParams | None):
        self.name = name
        self.model_id = model_id
        self.sampling_params = (
            sampling_params
            if isinstance(sampling_params, SamplingParams)
            else SamplingParams.from_dict(sampling_params or {})
        )

    @abstractmethod
    def _call(self, system: str, user: str) -> tuple:
        ...

    def generate(self, system: str, user: str, log_path: str = "inference_result.csv"):
        t0 = time.perf_counter()
        text, in_token, out_token = self._call(system, user)
        latency = time.perf_counter() - t0

        if in_token is None:
            in_token = rough_token_count(user)
        if out_token is None:
            out_token = rough_token_count(text)

        tps = out_token / latency if latency > 0 else 0.0

        metrics = {
            "model_name": self.name,
            "input_token": int(in_token),
            "output_token": int(out_token),
            "latency_sec": round(float(latency), 4),
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

    def with_sampling(self, **overrides):
        """
        Return a copy of the model with updated sampling parameters.
        The original model is not modified.

        Example:
            greedy = model.with_sampling(temperature=0.0)
            creative = model.with_sampling(temperature=1.0, top_p=0.95)
        """
        new_model = copy.copy(self)
        new_model.sampling_params = dataclasses.replace(self.sampling_params, **overrides)
        return new_model