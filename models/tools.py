import time
from pathlib import Path
import random
import os


def timed(call):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = call(*args, **kwargs)
        latency = time.perf_counter() - t0
        return out, latency
    return wrapper

def rough_token_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())

def get_keys(name: str) -> list[str]:
    keys = os.getenv(name)
    if not keys:
        raise RuntimeError(f"Environment variable {name} not found.")
    keys = [k.strip() for k in keys.split(',') if k.strip()]
    random.shuffle(keys)
    return keys