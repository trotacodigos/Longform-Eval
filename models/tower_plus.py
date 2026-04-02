from .sampling import TowerPlusParams
from .base_openai import OpenAIChatModel


class TowerPlusModel(OpenAIChatModel):
    # https://huggingface.co/Unbabel/Tower-Plus-72B

    def __init__(self,
                 name: str = "tower-plus-72b",
                 model_id: str = "tower-plus-72b",
                 sampling_params: TowerPlusParams | dict | None = None,
                 strip_thinking: bool = False,
                 src_lang: str = "en",
                 tgt_lang: str = "ko"):
        super().__init__(name, model_id, sampling_params=sampling_params or TowerPlusParams(),
                         strip_thinking=strip_thinking, merge_system_prompt=True)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang