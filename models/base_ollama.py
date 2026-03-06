from .base import BaseModel, Decoding
from .tools import _drop_none, extract_token_usage

import os
import requests
from dataclasses import asdict
from typing import Dict, Any


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


class OllamaModel(BaseModel):
    def __init__(self, name, model_id, decoding=None, host=None):
        super().__init__(name, model_id, decoding or {})
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.chat_url = f"{self.host}/api/chat"
        self.gen_url  = f"{self.host}/api/generate"
        self.session = requests.Session()
        self.session.headers.update({"Connection": "keep-alive"})

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

        in_token, out_token = extract_token_usage(data)
        return text, in_token, out_token