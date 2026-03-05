from .base_anthropic import AnthropicModel
from .base import Decoding


class ClaudeSonnet4_5(AnthropicModel):
    def __init__(self, decoding=None):
        super().__init__(
            name="claude-sonnet-4.5", 
            model_id="claude-sonnet-4-5",
            decoding=decoding or Decoding()
        )