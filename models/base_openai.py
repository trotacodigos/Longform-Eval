from tools import get_keys, extract_token_usage
from base import BaseModel
from sampling import OpenAIParams

from openai import OpenAI


class OpenAIModel(BaseModel):
    def __init__(self, name: str, 
                 model_id: str, 
                 sampling_params: OpenAIParams | dict | None = None
                 ):
        super().__init__(name, model_id, sampling_params)
        keys = get_keys("OPENAI_API_KEYS")
        self.client = OpenAI(api_key=keys[0])

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
        text = (content or "").strip()

        usage = getattr(resp, "usage", None)
        in_token, out_token = extract_token_usage(usage)

        return text, in_token, out_token