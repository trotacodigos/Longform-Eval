from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any


@dataclass
class SamplingParams:
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 8_192
    stop: Optional[List[str]] = None

    def to_kwargs(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    @classmethod
    def from_dict(cls, dic: dict):
        return cls(**dic)
    
@dataclass
class ThinkingMixin:
    thinking: bool = False

    def to_kwargs(self) -> Dict[str, Any]:
        kwargs = super().to_kwargs()
        if self.thinking:
            kwargs["thinking"] = True
            kwargs["max_tokens"] = 32_768
            kwargs.pop("temperature", None)
        return kwargs
    
    
# child classes
@dataclass
class OpenAIParams(ThinkingMixin, SamplingParams):
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_tokens: int = 16_384


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
    
@dataclass
class HyperClovaXParams(ThinkingMixin, HuggingFaceParams):
    thinking: bool = False
    thinking_token_budget: int = 4000 # up to 8000

    def to_kwargs(self) -> Dict[str, Any]:
        kwargs = super().to_kwargs()
        if self.thinking:
            kwargs.pop("temperature", None)
            kwargs["extra_body"] = {
                "thinking_token_budget": self.thinking_token_budget,
                "chat_template_kwargs": {"thinking": True}
            }
        else:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"thinking": False}
            }
        return kwargs