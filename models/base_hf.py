from .base import BaseModel
from .sampling import HuggingFaceParams
from .tools import extract_token_usage

import requests

    
class HFChatModel(BaseModel):
    def __init__(
        self,
        name: str,
        model_id: str,
        endpoint: str,
        sampling_params: HuggingFaceParams | dict | None = None,
        prompt_adapter=None,
        tgt_lang: str | None = None,
    ):
        super().__init__(name, model_id, sampling_params or HuggingFaceParams())
        if not endpoint:
            raise ValueError("HFChatModel requires `endpoint`")
        self.endpoint = endpoint
        self.prompt_adapter = prompt_adapter
        self.tgt_lang = tgt_lang

    def _call(self, system: str, user: str):
        if self.prompt_adapter:
            user = self.prompt_adapter(user, self.tgt_lang) if self.tgt_lang else self.prompt_adapter(user)

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **self.sampling_params.to_kwargs(),
        }
        response = requests.post(self.endpoint, json=payload, timeout=600)
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        text = (content or "").strip()

        usage = data.get("usage", "")
        in_token, out_token = extract_token_usage(usage)

        return text, in_token, out_token