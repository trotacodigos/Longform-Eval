from base_anthropic import AnthropicModel

class ClaudeSonnet4_5(AnthropicModel):
    def __init__(self):
        super().__init__(name="claude-sonnet-4.5", model_id="claude-sonnet-4-5")