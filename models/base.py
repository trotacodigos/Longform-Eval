from .tools import rough_token_count, _drop_none

from dataclasses import dataclass
from typing import Optional, List
import copy

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

    def _call(self, system: str, user: str):
        raise NotImplementedError

    def generate(self, system: str, user: str):
        (text, in_token, out_token), latency = self._call(system, user)
        if in_token is None:
            in_token = rough_token_count(user)
        if out_token is None:
            out_token = rough_token_count(text)
        return text, {
            "input_token": int(in_token),
            "output_token": int(out_token),
            "latency": float(latency),
        }

    def with_decoding(self, **overrides):
        new_model = copy.copy(self)
        new_model.decoding = copy.deepcopy(self.decoding)

        for k, v in overrides.items():
            if hasattr(new_model.decoding, k):
                setattr(new_model.decoding, k, v)
        return new_model
