from .base_openai import OpenAIChatModel
from .sampling import OpenAIChatParams

class Gemma3Model(OpenAIChatModel):
    # https://huggingface.co/google/gemma-3-27b-it
    def __init__(self,
                 name: str = "gemma-3-27b-it",
                 model_id: str = "gemma-3-27b-it", #"google/gemma-3-27b-it",
                 sampling_params: OpenAIChatParams | dict | None = None,
                 strip_thinking: bool = False,
                 merge_system_prompt: bool = False):
        super().__init__(name, model_id, sampling_params or OpenAIChatParams(), 
                         strip_thinking=strip_thinking, merge_system_prompt=merge_system_prompt)