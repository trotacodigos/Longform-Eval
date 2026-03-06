from .base import BaseModel, Decoding
from .tools import _drop_none, extract_token_usage
from .base_openai import to_openai_kwargs

from dataclasses import asdict
from typing import Dict, Any
import requests



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

        usage = data.get("usage"}
        in_token, out_token = extract_token_usage(usage)

        return text, in_token, out_token
    