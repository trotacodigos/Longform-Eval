from .sampling import OpenAIChatParams
from .base_openai import OpenAIChatModel

class CommandATranslateModel(OpenAIChatModel):
    # https://huggingface.co/CohereLabs/command-a-translate-08-2025
    def __init__(self,
                 name: str = "command-a-translate-08-2025",
                 model_id: str = "CohereLabs/command-a-translate-08-2025",
                 sampling_params: OpenAIChatParams | dict | None = None,
                 strip_thinking: bool = False,
                 ):
        super().__init__(name, model_id, sampling_params or OpenAIChatParams(), 
                         strip_thinking=strip_thinking, merge_system_prompt=True)