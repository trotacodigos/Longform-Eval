# Tencent
from .base_openai import OpenAIChatModel
from .sampling import OpenAIChatParams


class HunyuanMTModel(OpenAIChatModel):
    def __init__(self,
                 name="hunyuan-mt-7b",
                 model_id="hunyuan-mt-7b",
                 sampling_params: OpenAIChatParams | dict | None = None,
                 strip_thinking = False,
                 ):
        super().__init__(name, model_id,
                         sampling_params=sampling_params or OpenAIChatParams(),
                         strip_thinking=strip_thinking, merge_system_prompt=True)