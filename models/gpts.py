from .basemodel import OpenAIModel, OpenAIThinking, Decoding
from .tools import timed

from openai import OpenAI


class GPT4o(OpenAIModel):
    def __init__(self, decoding=None):
        super().__init__("gpt-4o", "gpt-4o", decoding or Decoding())
        

class GPT4oMini(OpenAIModel):
    def __init__(self, decoding=None):
        super().__init__("gpt-4o-mini", "gpt-4o-mini", decoding or Decoding())


class GPT52Thinking(OpenAIThinking):
    def __init__(self, decoding=None, thinking=True):
        super().__init__("gpt-5.2-thinking", "gpt-5.2-thinking", decoding or Decoding(), thinking)

    @classmethod
    def with_thinking(cls, decoding=None):
        return cls(decoding=decoding, thinking=True)
    
    @classmethod
    def without_thinking(cls, decoding=None):
        return cls(decoding=decoding, thinking=False)
    