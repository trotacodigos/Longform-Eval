import time
import random
import os
from typing import Any, Tuple, Dict


def get_keys(name: str) -> list[str]:
    keys = os.getenv(name)
    if not keys:
        raise RuntimeError(f"Environment variable {name} not found.")
    keys = [k.strip() for k in keys.split(',') if k.strip()]
    random.shuffle(keys)
    return keys


def rough_token_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def extract_token_usage(usage_data: Any) -> Tuple[int | None, int | None]:
    if not usage_data:
        return None, None

    # 1) dict type
    if isinstance(usage_data, dict):
        in_tok = (usage_data.get("prompt_tokens") or 
                  usage_data.get("input_tokens") or 
                  usage_data.get("prompt_eval_count"))
        
        out_tok = (usage_data.get("completion_tokens") or 
                   usage_data.get("output_tokens") or 
                   usage_data.get("eval_count"))
        return in_tok, out_tok

    # 2) object type
    in_tok = (getattr(usage_data, "prompt_tokens", None) or 
              getattr(usage_data, "input_tokens", None))
    
    out_tok = (getattr(usage_data, "completion_tokens", None) or 
               getattr(usage_data, "output_tokens", None))
    
    return in_tok, out_tok