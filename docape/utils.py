import json

ALLOWED_DOMAINS = {"literary", "news", "social"}
LANG_MAP = {"en": "English", "zh_CN": "Chinese", "ko_KR": "Korean (South Korea)"}


def read_jsonl(file: str):
    with open(file, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def write_jsonl(path: str, records: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")