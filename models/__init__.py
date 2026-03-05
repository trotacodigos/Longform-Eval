from .base import Decoding, BaseModel
from .base_openai import OpenAIModel, OpenAIThinking
from .base_anthropic import AnthropicModel
from .base_ollama import OllamaModel
from .base_hf import HFChatModel
from .claude import ClaudeSonnet4_5
from .gpts import GPT4o, GPT52Thinking


REGISTRY = {
    "gpt-4o": GPT4o,
    #"gpt-4o-mini": GPT4oMini,
    #"llama3-8b-instruct": LLaMa3_8BInstruct,
    #"qwen2.5-32b-instruct": Qwen25_32BInstruct,
    "gpt-5.2-thinking": GPT52Thinking,
    "claude-sonnet-4.5": ClaudeSonnet4_5,
}