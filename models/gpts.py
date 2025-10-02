from .basemodel import OpenAIModel, Decoding
from .tools import timed

from openai import OpenAI


class GPT4o(OpenAIModel):
    def __init__(self, decoding=None):
        super().__init__("gpt-4o", "gpt-4o", decoding or Decoding())
        
class GPT4oMini(OpenAIModel):
    def __init__(self, decoding=None):
        super().__init__("gpt-4o-mini", "gpt-4o-mini", decoding or Decoding())