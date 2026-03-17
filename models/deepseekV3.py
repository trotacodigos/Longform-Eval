from .sampling import DeepseekParams
from .base_openai import OpenAIChatModel

class DeepSeekV3Model(OpenAIChatModel):
    # https://huggingface.co/deepseek-ai/DeepSeek-V3.2
    def __init__(self,
                 name: str = "deepseek-v3.2",
                 model_id: str = "deepseek-ai/DeepSeek-V3.2",
                 sampling_params: DeepseekParams | dict | None = None,
                 strip_thinking: bool = True,
                 merge_system_prompt: bool = False):
        super().__init__(name, model_id, sampling_params or DeepseekParams(thinking=True),
                         strip_thinking=strip_thinking, merge_system_prompt=merge_system_prompt)

    def _extra_payload(self) -> dict:
        return {} if self.sampling_params.thinking \
            else {"chat_template_kwargs": {"enable_thinking": False}}
    