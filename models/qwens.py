from .base_openai import OpenAIChatModel
from .sampling import QwenParams, OpenAIChatParams
from .tools import get_keys


class Qwen3_5Model(OpenAIChatModel):
    # https://huggingface.co/Qwen/Qwen3.5-27B
    def __init__(self, 
                 name="qwen3.5-27b", 
                 model_id="Qwen/Qwen3.5-27B", 
                 sampling_params: QwenParams | dict | None = None,
                 strip_thinking = False,
                 merge_system_prompt: bool = False):
        super().__init__(name, model_id, sampling_params or QwenParams(), 
                         strip_thinking=strip_thinking, merge_system_prompt=merge_system_prompt)

    def _extra_payload(self):
        return {} if self.sampling_params.thinking \
            else {"chat_template_kwargs": {"enable_thinking": False}}


class Qwen3Thinking(Qwen3_5Model):
    """Qwen3-235B-A22B-Thinking-2507 — thinking only"""

    def __init__(
        self,
        name="qwen3-235b-thinking-2507",
        model_id="Qwen/Qwen3-235B-A22B-Thinking-2507",
        sampling_params: QwenParams | dict | None = None,
        strip_thinking = True,
        merge_system_prompt: bool = False
    ):
        super().__init__(name, model_id, sampling_params or QwenParams(thinking=True), 
                         strip_thinking=strip_thinking, merge_system_prompt=merge_system_prompt)

    def _extra_payload(self):
        return {} # To remove `thinking`
    

class Qwen3MTModel(OpenAIChatModel):
    def __init__(
        self,
        name: str = "qwen-mt-plus",
        model_id: str = "qwen-mt-plus",
        endpoint: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        src_lang: str = "auto",
        tgt_lang: str = "Korean",
        sampling_params: OpenAIChatParams | dict | None = None,
        strip_thinking = False,
        # Optional params to enhance translation quality
        terms: list[dict] | None = None,       # [{"source": "...", "target": "..."}]
        tm_list: list[dict] | None = None,     # [{"source": "...", "target": "..."}]
        domains: str | None = None,            # domain prompt (english-only)
    ):
        # Do not require sampling_params
        super().__init__(name, model_id, endpoint, sampling_params or OpenAIChatParams(), 
                         strip_thinking=strip_thinking, merge_system_prompt=True)
        self.api_keys = get_keys("ALIBABA_API_KEYS") # Alibaba Cloud DashScope API key
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.terms = terms
        self.tm_list = tm_list
        self.domains = domains
    
    def _extra_payload(self):
        translation_options: dict = {
            "source_lang": self.src_lang,
            "target_lang": self.tgt_lang,
        }
        if self.terms:
            translation_options["terms"] = self.terms
        if self.tm_list:
            translation_options["tm_list"] = self.tm_list
        if self.domains:
            translation_options["domains"] = self.domains

        return {"translation_options": translation_options}
    
    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_keys[0]}",
            "Content-Type": "application/json",
            }