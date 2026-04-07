from .base_openai import OpenAIChatModel
from .sampling import ExaoneParams

class LGExaoneModel(OpenAIChatModel):
    # https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B
    def __init__(self, 
                 name: str = "exaone-4.0-32b", 
                 model_id: str = "LGAI-EXAONE--EXAONE-4.0-32B",
                 sampling_params: ExaoneParams | dict | None = None,
                 strip_thinking: bool = True,
                ):
           super().__init__(name, model_id, sampling_params=sampling_params or ExaoneParams(),
                             strip_thinking=strip_thinking, merge_system_prompt=True)

    def _extra_payload(self) -> dict:
        return {} if self.sampling_params.thinking \
            else {"chat_template_kwargs": {"enable_thinking": False}}


class K_ExaoneModel(LGExaoneModel):
     # https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B
    def __init__(self,
                 name: str = "k-exaone-236b-a23b",
                 model_id: str = "LGAI-EXAONE/K-EXAONE-236B-A23B",
                 sampling_params: ExaoneParams | dict | None = None,
                 strip_thinking: bool = True,
                 merge_system_prompt: bool = False):
        super().__init__(name=name, model_id=model_id, sampling_params=sampling_params or ExaoneParams(thinking=False),
                         strip_thinking=strip_thinking)