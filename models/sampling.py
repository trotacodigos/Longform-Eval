from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any


@dataclass
class SamplingParams:
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 8_192
    stop: Optional[List[str]] = None

    def to_kwargs(self, keys: List[str]) -> Dict[str, Any]:
        d = asdict(self)
        return {k: d[k] for k in keys if d.get(k) is not None}
    
    @classmethod
    def from_dict(cls, dic: dict):
        return cls(**dic)
    
@dataclass
class ThinkingParams:
    thinking: bool = True
    max_tokens: int = 32_768

    def to_kwargs(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    
# child classes
@dataclass
class OpenAIParams(SamplingParams):
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    thinking: bool = False
    max_tokens: int = 16_384

    def to_kwargs(self):
        kwargs = super().to_kwargs()
        if self.thinking:
            kwargs.update(ThinkingParams().to_kwargs())
            kwargs.pop("temperature", None) # fixed temperature to 1.0
        return kwargs

@dataclass
class ClaudeParams(SamplingParams):

    def to_kwargs(self) -> Dict[str, Any]:
        kwargs = super().to_kwargs()
        if "stop" in kwargs:
            kwargs["stop_sequences"] = kwargs.pop("stop")
        return kwargs

@dataclass
class HuggingFaceParams(SamplingParams):
    repetition_penalty: Optional[float] = None
    
    def to_kwargs(self):
        kwargs = super().to_kwargs()
        kwargs["do_sample"] = (self.temperature or 0.0) > 0.0
        return kwargs
    
@dataclass
class OllamaParams(SamplingParams):
    num_ctx: Optional[int] = None
    repetition_penalty: Optional[float] = None
    min_p: Optional[float] = None

    def to_kwargs(self):
        kwargs = super().to_kwargs()
        kwargs["num_predict"] = kwargs.pop("max_tokens")
        if "repetition_penalty" in kwargs:
            kwargs["repeat_penalty"] = kwargs.pop("repetition_penalty")
        return kwargs