from .base_openai import OpenAIChatModel
from .sampling import OpenAIChatParams

class Gemma3Model(OpenAIChatModel):
    # https://huggingface.co/google/gemma-3-27b-it
    def __init__(self,
                 name: str = "gemma-3-27b-it",
                 model_id: str = "google/gemma-3-27b-it",
                 endpoint: str = "http://localhost:8000/v1/chat/completions",
                 sampling_params: OpenAIChatParams | dict | None = None,
                 strip_thinking: bool = False):
        super().__init__(name, model_id, endpoint, 
                         sampling_params or OpenAIChatParams(), strip_thinking=strip_thinking)