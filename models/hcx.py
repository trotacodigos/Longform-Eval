from openai import OpenAI

from .base_hf import HFChatModel
from .sampling import HyperClovaXParams
from .tools import extract_token_usage


class HyperClovaXModel(HFChatModel):
    def __init__(self, name="hcx-seed-thinking-32b", 
                 model_id="naver/HyperCLOVAX-SEED-Think-32B", 
                 endpoint="http://localhost:8000/v1", 
                 sampling_params: HyperClovaXParams | dict | None = None):
        super().__init__(name, model_id, endpoint, sampling_params or HyperClovaXParams())
        self.client = OpenAI(base_url=self.endpoint, api_key="not-needed")

    def _call(self, system: str, user: str):
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **self.sampling_params.to_kwargs(),
        )
        content = resp.choices[0].message.content
        if "</think>" in content:
            content = content.split("</think>")[-1]
        text = (content or "").strip()

        usage = getattr(resp, "usage", None)
        in_token, out_token = extract_token_usage(usage)
        return text, in_token, out_token