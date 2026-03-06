from .tools import get_keys, extract_token_usage
from .base import BaseModel, Decoding

from openai import OpenAI
from typing import Dict, Any
from dataclasses import asdict


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


class OpenAIModel(BaseModel):
    def __init__(self, name: str, model_id: str, decoding: Decoding | dict | None = None):
        super().__init__(name, model_id, decoding)
        keys = get_keys("OPENAI_API_KEYS")
        self.client = OpenAI(api_key=keys[0])

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
        in_token, out_token = extract_token_usage(usage)

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
        in_token, out_token = extract_token_usage(usage)
        
        return text, in_token, out_token