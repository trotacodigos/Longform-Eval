# https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-32B
from openai import OpenAI

from .base_openai import OpenAIModel
from .sampling import HyperClovaXParams
from .tools import extract_token_usage


class HyperClovaXModel(OpenAIModel):
    """VLM"""
    def __init__(self,
                 name="hcx-seed-thinking-32b",
                 model_id="track_a_model",
                 endpoint="http://localhost:8000/a/v1",
                 sampling_params: HyperClovaXParams | dict | None = None,
                 strip_thinking: bool = True):
        super().__init__(name, model_id, sampling_params or HyperClovaXParams())
        self.client = OpenAI(base_url=endpoint, api_key="not-needed")
        self.strip_thinking = strip_thinking

    def _complete(self, system: str, user: str):
        # Do not require system prompt
        merged = f"{system}\n{user}" if system else user

        extra_body = {
            "chat_template_kwargs": {"thinking": self.sampling_params.thinking},
        }
        if self.sampling_params.thinking:
            extra_body["thinking_token_budget"] = self.sampling_params.thinking_token_budget

        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": merged}],
            extra_body=extra_body,
            **self.sampling_params.to_kwargs(),
        )

        content = resp.choices[0].message.content or ""
        if self.strip_thinking and "</think>" in content:
            content = content.split("</think>", 1)[-1]

        usage = getattr(resp, "usage", {})
        in_token, out_token = extract_token_usage(usage)
        return content.strip(), in_token, out_token