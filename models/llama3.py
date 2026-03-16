from .sampling import OpenAIChatParams
from .base_openai import OpenAIChatModel


class Llama3Model(OpenAIChatModel):
    # https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
    def __init__(self,
                 name: str = "llama-3.3-70b-instruct",
                 model_id: str = "meta-llama/Llama-3.3-70B-Instruct",
                 endpoint: str = "http://localhost:8000/v1/chat/completions",
                 sampling_params: OpenAIChatParams | dict | None = None,
                 strip_thinking: bool = False,
                 merge_system_prompt: bool = False,):
        super().__init__(name, model_id, endpoint, sampling_params or OpenAIChatParams(), 
                         strip_thinking=strip_thinking, merge_system_prompt=merge_system_prompt)