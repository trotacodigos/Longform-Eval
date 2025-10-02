from .gpts import GPT4o, GPT4oMini
from .llama3 import LLaMa3_8BInstruct
from .qwen25 import Qwen25_32BInstruct

REGISTRY = {
    "gpt-4o": GPT4o,
    "gpt-4o-mini": GPT4oMini,
    "llama3-8b-instruct": LLaMa3_8BInstruct,
    "qwen2.5-32b-instruct": Qwen25_32BInstruct,
}