from .claude import ClaudeSonnet4_5
from .commandTrans import CommandATranslateModel
from .deepseekV3 import DeepSeekV3Model
from .exaone import LGExaoneModel, K_ExaoneModel
from .gemini25 import GeminiModel
from .gemma import Gemma3Model
from .gpts import GPT4o, GPT52Thinking
from .grok4 import GrokFast41Model, Grok420BetaReasoningModel
from .hcx import HyperClovaXModel
from .hunyuan import HunyuanMTModel
from .llama3 import Llama3Model
from .qwens import Qwen3_5Model, Qwen3Thinking, Qwen3MTModel
from .solar import SolarOpenModel
from .tower_plus import TowerPlusModel


REGISTRY = {
    ### old models
    #"gpt-4o-mini": GPT4oMini,
    #"llama3-8b-instruct": LLaMa3_8BInstruct,
    #"qwen2.5-32b-instruct": Qwen25_32BInstruct,
    ### updated models
    "claude-sonnet-4.5": ClaudeSonnet4_5,
    "command-a-translate-08-2025": CommandATranslateModel,
    "deepseek-v3.2": DeepSeekV3Model,
    "exaone-4.0-32b": LGExaoneModel,
    "k-exaone-236b-a23b": K_ExaoneModel,
    "gemini-2.5-pro": GeminiModel,
    "gemma-3-27b-it": Gemma3Model,
    "gpt-4o": GPT4o,
    "gpt-5.2-thinking": GPT52Thinking,
    "grok-4.1-fast": GrokFast41Model,
    "grok-4.20-beta-reasoning": Grok420BetaReasoningModel,
    "hcx-seed-thinking-32b": HyperClovaXModel,
    "hunyuan-mt-7b": HunyuanMTModel,
    "llama-3.3-70b-instruct": Llama3Model,
    "qwen-mt-plus": Qwen3MTModel,
    "qwen3-235b-thinking-2507": Qwen3Thinking,
    "qwen3.5-27b": Qwen3_5Model,
    "solar-open-100b": SolarOpenModel,
    "tower-plus-72b": TowerPlusModel,
}