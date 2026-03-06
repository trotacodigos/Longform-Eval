import os
from dataclasses import asdict
from typing import Dict, Any
from anthropic import Anthropic

from .base import Decoding, BaseModel
from .tools import get_keys, _drop_none, extract_token_usage


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


class AnthropicModel(BaseModel):
    def __init__(self, name: str, model_id: str, decoding: Decoding | dict | None = None):
        super().__init__(name, model_id, decoding)

        keys = get_keys("ANTHROPIC_API_KEYS")
        api_key = keys[0] if keys else os.environ.get("ANTHROPIC_API_KEY")
        
        self.client = Anthropic(api_key=api_key)

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
        in_token, out_token = extract_token_usage(usage)
        
        return text, in_token, out_token