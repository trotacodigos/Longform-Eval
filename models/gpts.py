from .base_openai import OpenAIModel
from .sampling import OpenAIParams


class GPT4o(OpenAIModel):
    def __init__(self, sampling_params=None):
        super().__init__("gpt-4o", "gpt-4o", sampling_params or OpenAIParams())


class GPT4oMini(OpenAIModel):
    def __init__(self, sampling_params=None):
        super().__init__("gpt-4o-mini", "gpt-4o-mini", sampling_params or OpenAIParams())


class GPT52Thinking(OpenAIModel):
    def __init__(self, sampling_params=None):
        super().__init__("gpt-5.2-thinking", "gpt-5.2-thinking", sampling_params or OpenAIParams(thinking=True))