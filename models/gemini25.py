from .base import BaseModel
from .sampling import GeminiParams
from .tools import get_keys

from google import genai
from google.genai import types


class GeminiModel(BaseModel):
    def __init__(
        self,
        name: str = "gemini-2.5-pro",
        model_id: str = "gemini-2.5-pro-preview-05-06",
        sampling_params: GeminiParams | dict | None = None,
        prompt_adapter=None,
        tgt_lang: str | None = None,
    ):
        super().__init__(name, model_id, sampling_params or GeminiParams())
        api_key = get_keys("GEMINI_API_KEYS")[0]
        self.client = genai.Client(api_key=api_key)
        self.prompt_adapter = prompt_adapter
        self.tgt_lang = tgt_lang

    def _complete(self, system: str | None, user: str):
        if self.prompt_adapter:
            user = self.prompt_adapter(user, self.tgt_lang) if self.tgt_lang else self.prompt_adapter(user)

        config = types.GenerateContentConfig(
            system_instruction=system,
            **self.sampling_params.to_kwargs(),
        )

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=user,
            config=config,
        )

        text = response.text.strip()
        usage = response.usage_metadata
        in_token = getattr(usage, "prompt_token_count", None)
        out_token = getattr(usage, "candidates_token_count", None)

        return text, in_token, out_token