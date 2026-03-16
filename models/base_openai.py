from .tools import get_keys, extract_token_usage
from .base import BaseModel
from .sampling import OpenAIParams, OpenAIChatParams

from openai import OpenAI
import requests

class OpenAIModel(BaseModel):
    def __init__(self, name: str, 
                 model_id: str, 
                 sampling_params: OpenAIParams | dict | None = None
                 ):
        super().__init__(name, model_id, sampling_params or OpenAIParams())
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

        usage = getattr(resp, "usage", {})
        in_token, out_token = extract_token_usage(usage)

        return text, in_token, out_token
    

class OpenAIChatModel(BaseModel):
    def __init__(
        self,
        name: str,
        model_id: str,
        endpoint: str,
        sampling_params: OpenAIChatParams | dict | None = None,
        prompt_adapter=None,
        tgt_lang: str | None = None,
        strip_thinking: bool = False,
    ):
        super().__init__(name, model_id, sampling_params or OpenAIChatParams())
        if not endpoint:
            raise ValueError("OpenAIChatModel requires `endpoint`")
        self.endpoint = endpoint
        self.prompt_adapter = prompt_adapter
        self.tgt_lang = tgt_lang
        self.strip_thinking = strip_thinking

    def _call(self, system: str | None, user: str):
        """MAIN"""
        if self.prompt_adapter:
            user = self.prompt_adapter(user, self.tgt_lang) if self.tgt_lang else self.prompt_adapter(user)

        if system is None:
            message = [{"role": "user", "content": user}]
        else:
            message = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                ]
        payload = {
            "model": self.model_id,
            "messages": message,
            **self.sampling_params.to_kwargs(),
            **self._extra_payload(),            
        }
        response = requests.post(self.endpoint, json=payload, headers=self._headers(), timeout=600)
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"] or ""

        if self.strip_thinking and "</think>" in content:
            content = content.split("</think>", 1)[-1]

        text = self._post_process(content).strip()  

        usage = data.get("usage", {})
        in_token, out_token = extract_token_usage(usage)
        return text, in_token, out_token

    def _extra_payload(self) -> dict:
        """To add customized keys to the payload"""
        return {}
    
    def _headers(self) -> dict:
        """Override to add custom request headers (e.g. auth)"""
        return {}

    def _post_process(self, content: str) -> str:
        """To post-process responses"""
        return content

    