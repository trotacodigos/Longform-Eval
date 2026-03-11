from base import BaseModel
from sampling import OllamaParams
from tools import extract_token_usage

import os
import requests

    
class OllamaModel(BaseModel):
    def __init__(self, name: str, model_id: str, sampling_params: OllamaParams | dict | None = None, host: str | None = None):
        super().__init__(name, model_id, sampling_params)
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.chat_url = f"{self.host}/api/chat"
        self.gen_url  = f"{self.host}/api/generate"
        self.session = requests.Session()
        self.session.headers.update({"Connection": "keep-alive"})

    def _call(self, system: str, user: str):
        options = self.sampling_params.to_kwargs()

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

        text = (data["message"]["content"] or "").strip() if "message" in data else (data.get("response") or "").strip()
        in_token, out_token = extract_token_usage(data)
        return text, in_token, out_token