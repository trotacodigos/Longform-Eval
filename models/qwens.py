from .base_hf import HFChatModel
from .base import BaseModel
from .sampling import QwenParams
from .tools import extract_token_usage

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

        payload = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "user", "content": system},
                {"role": "user", "content": user},
            ],
            **sampling_kwargs,
            **(
                {} if thinking
                else {"chat_template_kwargs": {"enable_thinking": False}}
            )
        )
        response = requests.post(self.endpoint, json=payload, timeout=600)
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"] or ""

        # <think>...</think> 블록 제거 (thinking=True일 때 포함됨)
        if "</think>" in content:
            content = content.split("</think>", 1)[-1]

        text = content.strip()

        usage = data.get("usage", "")
        in_token, out_token = extract_token_usage(usage)

        return text, in_token, out_token


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
        sampling_kwargs.pop("thinking", None)  # 불필요한 키 제거

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

        if "<think>" in content:
            content = content.split("<think>", 1)[-1]
        if "</think>" in content:
            content = content.split("</think>", 1)[-1]

        text = content.strip()

        usage = data.get("usage", "")
        in_token, out_token = extract_token_usage(usage)

        return text, in_token, out_token
    

class Qwen3MTModel(BaseModel):
    # DashScopeAPI 전용
    ENDPOINT = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"

    def __init__(
        self,
        name: str = "qwen-mt-plus",
        model_id: str = "qwen-mt-plus",
        api_key: str | None = None, # Alibaba Cloud DashScope API key
        src_lang: str = "auto",
        tgt_lang: str = "Korean",
        # 번역 품질 향상 옵션 (선택)
        terms: list[dict] | None = None,       # [{"source": "...", "target": "..."}]
        tm_list: list[dict] | None = None,     # [{"source": "...", "target": "..."}]
        domains: str | None = None,            # 도메인 프롬프트 (영어만 지원)
    ):
        # MT 모델은 sampling_params 불필요
        super().__init__(name, model_id, sampling_params=None)
        if not api_key:
            raise ValueError("QwenMTModel requires a DashScope `api_key`")
        self.api_key = api_key
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.terms = terms
        self.tm_list = tm_list
        self.domains = domains

    def _call(self, system: str, user: str):
        # system 메시지 미지원 — user에 번역 대상 텍스트만 전달
        # (system 인자는 인터페이스 호환을 위해 받되 무시)

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
            "extra_body": {"translation_options": translation_options},
        }

        response = requests.post(
            self.ENDPOINT,
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=600,
        )
        response.raise_for_status()
        data = response.json()

        text = (data["choices"][0]["message"]["content"] or "").strip()

        usage = data.get("usage", "")
        in_token, out_token = extract_token_usage(usage)

        return text, in_token, out_token