from .base_hf import HFChatModel
from .base import BaseModel
from .sampling import QwenParams
from .tools import extract_token_usage, get_keys

import requests


class Qwen3_5Model(HFChatModel):
    # https://huggingface.co/Qwen/Qwen3.5-27B
    def __init__(self, 
                 name="qwen3.5-27b", 
                 model_id="Qwen/Qwen3.5-27B", 
                 endpoint="http://localhost:8000/v1/chat/completions", 
                 sampling_params: QwenParams | dict | None = None,):
        super().__init__(name, model_id, endpoint, sampling_params)

    def _call(self, system: str, user: str):
        sampling_kwargs = self.sampling_params.to_kwargs()
        thinking = sampling_kwargs.pop("thinking", False)

        payload = {
        "model": self.model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **sampling_kwargs,
        **(
            {}
            if thinking
            else {"chat_template_kwargs": {"enable_thinking": False}}
        ),
    }
        response = requests.post(self.endpoint, json=payload, timeout=600)
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"] or ""

        if "</think>" in content:
            content = content.split("</think>", 1)[-1]

        usage = data.get("usage", {})
        in_token, out_token = extract_token_usage(usage)

        return content.strip(), in_token, out_token


class Qwen3Thinking(Qwen3_5Model):
    """Qwen3-235B-A22B-Thinking-2507 — thinking only"""

    def __init__(
        self,
        name="qwen3-235b-thinking-2507",
        model_id="Qwen/Qwen3-235B-A22B-Thinking-2507",
        endpoint="http://localhost:8000/v1/chat/completions",
        sampling_params: QwenParams | dict | None = None,
    ):
        super().__init__(
            name, model_id, endpoint,
            sampling_params or QwenParams(thinking=True),
        )

    def _call(self, system: str, user: str):
        sampling_kwargs = self.sampling_params.to_kwargs()
        sampling_kwargs.pop("thinking", None) # Remove unnecessary keys

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **sampling_kwargs,
        }

        response = requests.post(self.endpoint, json=payload, timeout=600)
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"] or ""
        if "</think>" in content:
            content = content.split("</think>", 1)[-1]

        usage = data.get("usage", {})
        in_token, out_token = extract_token_usage(usage)

        return content.strip(), in_token, out_token
    

class Qwen3MTModel(BaseModel):
    # via DashScopeAPI
    ENDPOINT = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"

    def __init__(
        self,
        name: str = "qwen-mt-plus",
        model_id: str = "qwen-mt-plus",
        src_lang: str = "auto",
        tgt_lang: str = "Korean",
        # Optional params to enhance translation quality
        terms: list[dict] | None = None,       # [{"source": "...", "target": "..."}]
        tm_list: list[dict] | None = None,     # [{"source": "...", "target": "..."}]
        domains: str | None = None,            # domain prompt (english-only)
    ):
        # Do not require sampling_params
        super().__init__(name, model_id, sampling_params=None)
        self.api_keys = get_keys("ALIBABA_API_KEYS") # Alibaba Cloud DashScope API key
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.terms = terms
        self.tm_list = tm_list
        self.domains = domains

    def _call(self, system: str, user: str):
        # Do not support system message
        translation_options: dict = {
            "source_lang": self.src_lang,
            "target_lang": self.tgt_lang,
        }
        if self.terms:
            translation_options["terms"] = self.terms
        if self.tm_list:
            translation_options["tm_list"] = self.tm_list
        if self.domains:
            translation_options["domains"] = self.domains

        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": user}],
            "translation_options": translation_options,
        }

        response = requests.post(
            self.ENDPOINT,
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_keys[0]}",
                "Content-Type": "application/json",
            },
            timeout=600,
        )
        response.raise_for_status()
        data = response.json()

        text = (data["choices"][0]["message"]["content"] or "").strip()

        usage = data.get("usage", {})
        in_token, out_token = extract_token_usage(usage)

        return text, in_token, out_token