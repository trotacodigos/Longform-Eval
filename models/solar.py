from .sampling import OpenAIChatParams
from .base_openai import OpenAIChatModel

class SolarOpenModel(OpenAIChatModel):
    def __init__(self, 
                 name="solar-open-100b", 
                 model_id="upstage/Solar-Open-100B", 
                 sampling_params: OpenAIChatParams | dict | None = None,
                 strip_thinking: bool = False,
                 merge_system_prompt: bool = False):
        super().__init__(name, model_id, sampling_params=sampling_params or OpenAIChatParams(),
                         strip_thinking=strip_thinking, merge_system_prompt=merge_system_prompt)
