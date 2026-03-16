from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: float = 0.7
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

@dataclass
class ClaudeParams(SamplingParams):
    def to_kwargs(self) -> Dict[str, Any]:
        kwargs = super().to_kwargs()
        if "stop" in kwargs:
            kwargs["stop_sequences"] = kwargs.pop("stop")
        return kwargs

@dataclass
class OpenAIChatParams(SamplingParams):
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
class HyperClovaXParams(ThinkingMixin, OpenAIChatParams):
    thinking_token_budget: int = 4000

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
    
@dataclass
class QwenParams(ThinkingMixin, OpenAIChatParams):
    min_p: float = 0.0
    presence_penalty: float = 1.5

    def to_kwargs(self):
        kwargs = super().to_kwargs()
        if self.thinking:
            kwargs["temperature"] = 1.0
            kwargs["top_p"] = 0.95
        return kwargs

@dataclass
class ExaoneParams(ThinkingMixin, OpenAIChatParams):
    pass

@dataclass
class TowerPlusParams(OpenAIChatParams):
    best_of: int = 1

@dataclass
class DeepseekParams(ThinkingMixin, OpenAIChatParams):
    def to_kwargs(self):
        kwargs = super().to_kwargs()
        if self.thinking:
            kwargs["temperature"] = 1.0
        return kwargs
    
@dataclass
class GeminiParams(ThinkingMixin, SamplingParams):
    
    def to_kwargs(self):
        kwargs = super().to_kwargs()
        kwargs.pop("thinking", None)
        if self.thinking:
            kwargs["temperature"] = 1.0
            kwargs["thinking_budget"] = -1
        else:
            kwargs["thinking_budget"] = 0

@dataclass
class GrokParams(ThinkingMixin, OpenAIChatParams):
    """
    thinking=True  → Add reasoning_effort to the payload
    thinking=False → Remove reasoning_effort (non-reasoning mode)
    """
    reasoning_effort: str = "high"  # "low" | "medium" | "high"

    def to_kwargs(self) -> Dict[str, Any]:
        kwargs = OpenAIChatParams.to_kwargs(self)
        kwargs.pop("thinking", None)
        kwargs.pop("reasoning_effort", None)
        if self.thinking:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return kwargs