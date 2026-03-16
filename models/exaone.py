# Exaone4.0-32B, K-Exaone-236B-A23B
from .base_openai import OpenAIChatModel
from .sampling import ExaoneParams

class LGExaoneModel(OpenAIChatModel):
    # https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B
    def __init__(self, 
                 name: str = "exaone-4.0-32b", 
                 model_id: str = "LGAI-EXAONE/EXAONE-4.0-32B",
                 endpoint: str = "http://localhost:8000/v1/chat/completions",
                 sampling_params: ExaoneParams | dict | None = None,
                 strip_thinking: bool = False):
            super().__init__(name, model_id, endpoint, sampling_params or ExaoneParams(), strip_thinking)

    def _extra_payload(self) -> dict:
        return {} if self.sampling_params.thinking \
            else {"chat_template_kwargs": {"enable_thinking": False}}
    
    def _call(self, system: str, user: str):
        merged = f"{system}\n{user}" if system else user
        return super()._call(None, merged)