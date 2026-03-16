# https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-32B
from openai import OpenAI

from .base_hf import HFChatModel
from .sampling import HyperClovaXParams
from .tools import extract_token_usage


class HyperClovaXModel(HFChatModel):
    """VLM"""
    def __init__(self, name="hcx-seed-thinking-32b", 
                 model_id="track_a_model", 
                 endpoint="http://localhost:8000/a/v1", 
                 sampling_params: HyperClovaXParams | dict | None = None,
                 strip_thinking = True):
        super().__init__(name, model_id, endpoint, sampling_params or HyperClovaXParams(), strip_thinking)
        self.client = OpenAI(base_url=self.endpoint, api_key="not-needed")