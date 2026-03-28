from .tools import rough_token_count
from .sampling import SamplingParams

import os, time
import copy
import csv
import dataclasses
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed


class BaseModel(ABC):
    """
    Abstract base class for all model implementations.
    Subclasses must implement _complete.
    """
    BATCH_WORKERS: int = 32

    def __init__(self, name: str, model_id: str, sampling_params: dict | SamplingParams | None, log_path: str = "inference_result.csv"):
        self.name = name
        self.model_id = model_id
        self.sampling_params = (
            sampling_params
            if isinstance(sampling_params, SamplingParams)
            else SamplingParams.from_dict(sampling_params or {})
        )
        self.log_path = log_path
        self._csv_lock = threading.Lock()
        
    @abstractmethod
    def _complete(self, system: str, user: str) -> tuple:
        ...

    def generate(self, system: str, user: str):
        t0 = time.perf_counter()
        text, in_token, out_token = self._complete(system, user)
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
            "tps": round(float(tps), 2),
        }

        self._log_to_csv(metrics)
        return text, metrics

    def generate_batch(
        self,
        prompts: list[tuple[str, str]],
    ) -> list[tuple[str, dict] | None]:
        """
        Default implementation: parallel _complete via ThreadPoolExecutor.
        Claude subclass overrides this with the Anthropic Batch API.

        Returns a list aligned with `prompts`.
        Failed items are returned as None.
        """
        results = [None] * len(prompts)

        with ThreadPoolExecutor(max_workers=self.BATCH_WORKERS) as executor:
            futures = {
                executor.submit(self.generate, sys, usr): i
                for i, (sys, usr) in enumerate(prompts)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"[generate_batch] index {idx} failed: {e}")

        return results

    def _log_to_csv(self, metrics: dict):
        with self._csv_lock:
            file_exists = os.path.isfile(self.log_path)
            with open(self.log_path, mode="a", newline="", encoding="utf-8") as f:
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
