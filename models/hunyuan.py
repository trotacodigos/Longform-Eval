# Tencent
from .base_hf import HFChatModel
from .sampling import HuggingFaceParams
from .tools import extract_token_usage

import requests


class HunyuanMTModel(HFChatModel):
    """
    Hunyuan-MT-7B wrapper.
    - No system prompt (model card: "does not have the default system_prompt")
    - add_generation_prompt=False (required per model card)
    - English prompt template only
    """
    LANG_DISPLAY: dict[str, str] = {
        "zh": "中文", "zh-Hant": "繁体中文", "yue": "粤语",
        "en": "English", "fr": "French", "pt": "Portuguese",
        "es": "Spanish", "ja": "日本語", "tr": "Turkish",
        "ru": "Russian", "ar": "Arabic", "ko": "Korean",
        "th": "Thai", "it": "Italian", "de": "German",
        "vi": "Vietnamese", "ms": "Malay", "id": "Indonesian",
        "tl": "Filipino", "hi": "Hindi", "pl": "Polish",
        "cs": "Czech", "nl": "Dutch", "km": "Khmer",
        "my": "Burmese", "fa": "Persian", "gu": "Gujarati",
        "ur": "Urdu", "te": "Telugu", "mr": "Marathi",
        "he": "Hebrew", "bn": "Bengali", "ta": "Tamil",
        "uk": "Ukrainian", "bo": "Tibetan", "kk": "Kazakh",
        "mn": "Mongolian", "ug": "Uyghur",
    }
    _ZH_FAMILY = {"zh", "zh-Hant", "yue"}

    def __init__(self,
                 name="hunyuan-mt-7b",
                 model_id="tencent/Hunyuan-MT-7B",
                 endpoint="http://localhost:8000/v1/chat/completions",
                 sampling_params: HuggingFaceParams | dict | None = None,
                 src_lang: str = "en",
                 tgt_lang: str = "ko"):
        super().__init__(name, model_id, endpoint, sampling_params or HuggingFaceParams())
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
    
    def _call(self, system: str, user: str):
        payload = self.client.chat.completions.create(
            model=self.model_id,
            # The model doesn't receive system prompt
            messages=[
                {"role": "user", "content": system + "\n" + user},
            ],
            add_generation_prompt=False,
            **self.sampling_params.to_kwargs(),
        )
        
        response = requests.post(self.endpoint, json=payload, timeout=600)
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        text = (content or "").strip()

        usage = data.get("usage", "")
        in_token, out_token = extract_token_usage(usage)
        return text, in_token, out_token