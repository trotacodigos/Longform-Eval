import os
from anthropic import Anthropic

from .base import BaseModel
from .sampling import ClaudeParams
from .tools import get_keys, extract_token_usage

    
class AnthropicModel(BaseModel):
    def __init__(self, name: str, model_id: str, sampling_params: ClaudeParams | dict | None = None):
        super().__init__(name, model_id, sampling_params or ClaudeParams)
        keys = get_keys("ANTHROPIC_API_KEYS")
        api_key = keys[0] if keys else os.environ.get("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=api_key)

    def _call(self, system: str, user: str):
        resp = self.client.messages.create(
            model=self.model_id,
            system=system,
            messages=[{"role": "user", "content": user}],
            **self.sampling_params.to_kwargs(),
        )
        text = (resp.content[0].text or "").strip() if resp.content else ""
        usage = getattr(resp, "usage", {})
        in_token, out_token = extract_token_usage(usage)
        return text, in_token, out_token
    
    
class ClaudeSonnet4_5(AnthropicModel):
    def __init__(self):
        super().__init__(name="claude-sonnet-4.5", model_id="claude-sonnet-4-5")