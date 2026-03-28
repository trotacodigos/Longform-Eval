from .base import BaseModel
from .sampling import GrokParams
from .tools import get_keys, extract_token_usage

from openai import OpenAI


class GrokModel(BaseModel):
    """OpenAI SDK-based, xAI endpoint"""
    def __init__(
        self,
        name: str,
        model_id: str,
        sampling_params: GrokParams | dict | None = None,
        strip_thinking: bool = True,
        prompt_adapter=None,
        tgt_lang: str | None = None,
    ):
        super().__init__(name, model_id, sampling_params or GrokParams())
        self.client = OpenAI(
            api_key=get_keys("XAI_API_KEYS")[0],
            base_url="https://api.x.ai/v1",
        )
        self.strip_thinking = strip_thinking
        self.prompt_adapter = prompt_adapter
        self.tgt_lang = tgt_lang

    def _complete(self, system: str | None, user: str):
        if self.prompt_adapter:
            user = self.prompt_adapter(user, self.tgt_lang) if self.tgt_lang else self.prompt_adapter(user)

        messages = [{"role": "user", "content": user}] if system is None else [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            **self.sampling_params.to_kwargs(),
        )

        content = (resp.choices[0].message.content or "")
        if self.strip_thinking and "</think>" in content:
            content = content.split("</think>", 1)[-1]
        text = content.strip()

        in_token, out_token = extract_token_usage(getattr(resp, "usage", {}))
        return text, in_token, out_token


class GrokFast41Model(GrokModel):
    def __init__(self, name="grok-4.1-fast", model_id="grok-4-1-fast",
                 sampling_params: GrokParams | dict | None = None,
                 strip_thinking: bool = True, prompt_adapter=None, tgt_lang=None):
        super().__init__(name, model_id, sampling_params or GrokParams(thinking=True),
                         strip_thinking, prompt_adapter, tgt_lang)


class Grok420BetaReasoningModel(GrokModel):
    def __init__(self, name="grok-4.20-beta-reasoning", model_id="grok-4.20-beta-0309-reasoning",
                 sampling_params: GrokParams | dict | None = None,
                 strip_thinking: bool = True, prompt_adapter=None, tgt_lang=None):
        super().__init__(name, model_id, sampling_params or GrokParams(thinking=False),
                         strip_thinking, prompt_adapter, tgt_lang)