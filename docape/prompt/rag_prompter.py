from docape.prompt.ape_prompter import _get_templates
from docape.utils import read_jsonl

import json
import os

LEVEL = "doc"

def build_prompt(entry: dict, src_lang="English", tgt_lang="Korean", context="high"):
    assert context in ("high", "low", "full"), f"Unsupported context: {context}"

    record = {
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "src_seg": entry["src_seg"],
        "tgt_seg": entry["tgt_seg"],
        "src_ctx": entry[context]["src_ctx"],
        "tgt_ctx": entry[context]["tgt_ctx"],
    }
    prompt_dic = _get_templates().get(LEVEL)
    user = prompt_dic["user"].format(**record)
    system = prompt_dic["system"].format(tgt_lang=tgt_lang)
    return system, user
