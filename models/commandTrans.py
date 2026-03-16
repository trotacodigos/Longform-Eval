from .sampling import OpenAIChatParams
from .base_openai import OpenAIChatModel

class CommandATranslateModel(OpenAIChatModel):
    # https://huggingface.co/CohereLabs/command-a-translate-08-2025
    def __init__(self,
                 name: str = "command-a-translate-08-2025",
                 model_id: str = "CohereLabs/command-a-translate-08-2025",
                 endpoint: str = "http://localhost:8000/v1/chat/completions",
                 sampling_params: OpenAIChatParams | dict | None = None):
        super().__init__(name, model_id, endpoint, 
                         sampling_params or OpenAIChatParams(), merge_system_prompt=True)