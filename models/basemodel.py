import os
import time
import math
import requests

from .tools import timed, rough_token_count, get_keys

from openai import OpenAI
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any


@dataclass
class Decoding:
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 1024
    # 선택 필드(있는 경우만 각 백엔드로 pass-through)
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
        # dict, None 모두 허용. 최종적으로 Decoding 인스턴스로 보관
        self.decoding = decoding if isinstance(decoding, Decoding) else Decoding(**(decoding or {}))

    def _call(self, system: str, user: str):
        """서브클래스에서 구현. (text, in_tok, out_tok) 반환. @timed는 서브클래스에 붙일 것."""
        raise NotImplementedError

    def generate(self, system: str, user: str):
        # _call은 @timed로 감싸져 (result, latency)를 반환해야 함
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
        # 공통 디코딩 덮어쓰기
        for k, v in overrides.items():
            if hasattr(self.decoding, k):
                setattr(self.decoding, k, v)
        return self


# -----------------------------
# OpenAI
# -----------------------------
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
        text = resp.choices[0].message.content.strip()
        usage = getattr(resp, "usage", None)
        in_token = getattr(usage, "prompt_tokens", None) if usage else None
        out_token = getattr(usage, "completion_tokens", None) if usage else None
        return text, in_token, out_token


# -----------------------------
# Ollama
# -----------------------------
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


# -----------------------------
# HF Chat (vLLM/TGI /v1/chat/completions)
# -----------------------------
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
        text = data["choices"][0]["message"]["content"].strip()

        usage = data.get("usage", {}) or {}
        in_token = usage.get("prompt_tokens")
        out_token = usage.get("completion_tokens")
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