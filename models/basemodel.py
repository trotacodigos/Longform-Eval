import os
import requests
import copy

from .tools import timed, rough_token_count, get_keys

from openai import OpenAI
from anthropic import Anthropic

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any


@dataclass
class Decoding:
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 1024
    stop: Optional[List[str]] = None
    num_ctx: Optional[int] = None               # Ollama 컨텍스트
    frequency_penalty: Optional[float] = None   # OpenAI
    presence_penalty: Optional[float] = None    # OpenAI
    repetition_penalty: Optional[float] = None  # HF generate 계열
    min_p: Optional[float] = None               # 일부 샘플링 구현

    

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


# OpenAI
class OpenAIModel(BaseModel):
    def __init__(self, name: str, model_id: str, decoding: Decoding | dict | None = None):
        super().__init__(name, model_id, decoding)
        keys = get_keys("OPENAI_API_KEYS")  # ['sk-...','sk-...']
        self.client = OpenAI(api_key=keys[0])

    @timed
    def _call(self, system: str, user: str):
        kwargs = to_openai_kwargs(self.decoding)
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        content = resp.choices[0].message.content
        text = (content or "").strip()

        usage = getattr(resp, "usage", None)
        in_token = getattr(usage, "prompt_tokens", None) if usage else None
        out_token = getattr(usage, "completion_tokens", None) if usage else None
        return text, in_token, out_token
    

class OpenAIThinking(OpenAIModel):
    def __init__(self, name, model_id, decoding, thinking: bool = True):
        super().__init__(name, model_id, decoding)
        self.thinking = thinking

    @timed
    def _call(self, system: str, user: str):
        kwargs = to_openai_kwargs(self.decoding)

        kwargs.pop("top_p", None)
        kwargs.pop("frequency_penalty", None)
        kwargs.pop("presence_penalty", None)

        if self.thinking:
            kwargs.pop("temperature", None)
        
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            thinking=self.thinking,
            **kwargs,
        )
        content = resp.choices[0].message.content
        text = (content or "").strip()

        usage = getattr(resp, "usage", None)
        in_token = getattr(usage, "prompt_tokens", None) if usage else None
        out_token = getattr(usage, "completion_tokens", None) if usage else None
        return text, in_token, out_token


# Ollama
class OllamaModel(BaseModel):
    def __init__(self, name, model_id, decoding=None, host=None):
        super().__init__(name, model_id, decoding or {})
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.chat_url = f"{self.host}/api/chat"
        self.gen_url  = f"{self.host}/api/generate"
        self.session = requests.Session()
        self.session.headers.update({"Connection": "keep-alive"})

    @timed
    def _call(self, system: str, user: str):
        options = to_ollama_options(self.decoding)

        # 1) most recent Ollama: /api/chat
        chat_payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": options,
            "stream": False,
        }
        r = self.session.post(self.chat_url, json=chat_payload, timeout=600)

        # 2) old version: /api/generate pullback
        if r.status_code == 404:
            prompt = f"System: {system}\n\nUser: {user}\n\nAssistant:"
            gen_payload = {
                "model": self.model_id,
                "prompt": prompt,
                "options": options,
                "stream": False,
            }
            r = self.session.post(self.gen_url, json=gen_payload, timeout=600)

        r.raise_for_status()
        data = r.json()

        if "message" in data: 
            text = (data["message"]["content"] or "").strip()
        else:                 
            text = (data.get("response") or "").strip()

        in_token  = data.get("prompt_eval_count")
        out_token = data.get("eval_count")
        return text, in_token, out_token


# HF Chat (vLLM/TGI /v1/chat/completions)
class HFChatModel(BaseModel):
    def __init__(
        self,
        name: str,
        model_id: str,
        endpoint: str,
        decoding: Decoding | dict | None = None,
        prompt_adapter=None,
        tgt_lang: str | None = None,
    ):
        super().__init__(name, model_id, decoding or Decoding())
        if not endpoint:
            raise ValueError("HFChatModel requires `endpoint`")
        self.endpoint = endpoint
        self.prompt_adapter = prompt_adapter
        self.tgt_lang = tgt_lang

    @timed
    def _call(self, system: str, user: str):
        kwargs = to_openai_kwargs(self.decoding)

        if self.prompt_adapter:
            user = self.prompt_adapter(user, self.tgt_lang) if self.tgt_lang else self.prompt_adapter(user)

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        }
        response = requests.post(self.endpoint, json=payload, timeout=600)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        text = (content or "").strip()

        usage = data.get("usage", {}) or {}
        in_token = usage.get("prompt_tokens")
        out_token = usage.get("completion_tokens")
        return text, in_token, out_token
    

class AnthropicModel(BaseModel):
    def __init__(self, name: str, model_id: str, decoding: Decoding | dict | None = None):
        super().__init__(name, model_id, decoding)

        keys = get_keys("ANTHROPIC_API_KEYS")
        api_key = keys[0] if keys else os.environ.get("ANTHROPIC_API_KEY")
        
        self.client = Anthropic(api_key=api_key)

    @timed
    def _call(self, system: str, user: str):
        kwargs = to_anthropic_kwargs(self.decoding)
        
        resp = self.client.messages.create(
            model=self.model_id,
            system=system,
            messages=[
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        
        text = (resp.content[0].text or "").strip() if resp.content else ""
        
        usage = getattr(resp, "usage", None)
        in_token = getattr(usage, "input_tokens", None) if usage else None
        out_token = getattr(usage, "output_tokens", None) if usage else None
        
        return text, in_token, out_token
    

def _drop_none(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def to_openai_kwargs(dec: Decoding) -> Dict[str, Any]:
    d = asdict(dec)
    out = {
        "temperature": d["temperature"],
        "top_p": d["top_p"],
        "max_tokens": d["max_tokens"],
    }
    if d.get("frequency_penalty") is not None:
        out["frequency_penalty"] = d["frequency_penalty"]
    if d.get("presence_penalty") is not None:
        out["presence_penalty"] = d["presence_penalty"]
    if d.get("stop"):
        out["stop"] = d["stop"]
    return out

def to_ollama_options(dec: Decoding) -> Dict[str, Any]:
    d = asdict(dec)
    out = {
        "temperature": d["temperature"],
        "top_p": d["top_p"],
        "num_predict": d["max_tokens"],
    }
    if d.get("num_ctx") is not None: out["num_ctx"] = d["num_ctx"]
    if d.get("repetition_penalty") is not None: out["repeat_penalty"] = d["repetition_penalty"]
    if d.get("min_p") is not None: out["min_p"] = d["min_p"]
    if d.get("stop"): out["stop"] = d["stop"]
    return _drop_none(out)

def to_hf_generate_kwargs(dec: Decoding) -> Dict[str, Any]:
    d = asdict(dec)
    out = {
        "max_new_tokens": d["max_tokens"],
        "temperature": d["temperature"],
        "top_p": d["top_p"],
        "do_sample": (d["temperature"] or 0.0) > 0.0,
    }
    if d.get("repetition_penalty") is not None:
        out["repetition_penalty"] = d["repetition_penalty"]
    return _drop_none(out)

def to_anthropic_kwargs(dec: Decoding) -> Dict[str, Any]:
    d = asdict(dec)
    out = {
        "temperature": d["temperature"],
        "top_p": d["top_p"],
        "max_tokens": d["max_tokens"] if d["max_tokens"] is not None else 1024,
    }
    if d.get("stop"):
        out["stop_sequences"] = d["stop"]
    
    return _drop_none(out)