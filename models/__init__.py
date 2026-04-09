import importlib
import warnings

_MODEL_MAP = {
    "claude-sonnet-4.5":            (".claude",        "ClaudeSonnet4_5"),
    "command-a-translate-08-2025":  (".commandTrans",  "CommandATranslateModel"),
    "deepseek-v3.2":                (".deepseekV3",    "DeepSeekV3Model"),
    "exaone-4.0-32b":               (".exaone",        "LGExaoneModel"),
    "k-exaone-236b-a23b":           (".exaone",        "K_ExaoneModel"),
    "gemini-2.5-pro":               (".gemini25",      "GeminiModel"),
    "gemma-3-27b-it":               (".gemma",         "Gemma3Model"),
    "gpt-4o":                       (".gpts",          "GPT4o"),
    "gpt-5.2-thinking":             (".gpts",          "GPT5_2Thinking"),
    "gpt-5.4-2026-03-05":           (".gpts",          "GPT5_4"),
    "grok-4.1-fast":                (".grok4",         "GrokFast41Model"),
    "grok-4.20-beta-reasoning":     (".grok4",         "Grok420BetaReasoningModel"),
    "hcx-seed-thinking-32b":        (".hcx",           "HyperClovaXModel"),
    "hunyuan-mt-7b":                (".hunyuan",       "HunyuanMTModel"),
    "llama-3.3-70b-instruct":       (".llama3",        "Llama3Model"),
    "qwen-mt-plus":                 (".qwens",         "Qwen3MTModel"),
    "qwen3-235b-thinking-2507":     (".qwens",         "Qwen3Thinking"),
    "qwen3.5-27b":                  (".qwens",         "Qwen3_5Model"),
    "solar-open-100b":              (".solar",         "SolarOpenModel"),
    "tower-plus-72b":               (".tower_plus",    "TowerPlusModel"),
    "gemma-4-31b-it":               (".gemma",         "Gemma4Model"),
}

REGISTRY = {}
for _key, (_mod, _cls) in _MODEL_MAP.items():
    try:
        _module = importlib.import_module(_mod, package=__name__)
        REGISTRY[_key] = getattr(_module, _cls)
    except (ImportError, ModuleNotFoundError) as e:
        warnings.warn(f"Skipping '{_key}': {e}")
