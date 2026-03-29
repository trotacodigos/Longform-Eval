import os
import json
from typing import Dict
from .utils import LANG_MAP


_CACHED_TEMPLATES = None

def _get_templates() -> Dict[str, Dict]:
    global _CACHED_TEMPLATES
    if _CACHED_TEMPLATES is None:
        with open("data/template.json", "r", encoding="utf-8") as f:
            _CACHED_TEMPLATES = json.load(f)
    return _CACHED_TEMPLATES


def load_doc_from_json(wmt_year: str, lp: str, direction: str, doc_id: str, system: str = None):
    doc_path = os.path.join("data", f"{wmt_year}/{lp}/{direction}_doc.json")
    with open(doc_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    if direction == "src":
        return doc.get(doc_id, "")
    elif direction == "tgt":
        if system is None:
            raise ValueError("system must be provided for direction='tgt'")
        return (doc.get(doc_id) or {}).get(system, "")
    else:
        raise ValueError(f"Unknown direction: {direction!r}")


def build_prompt(entry: dict, level="seg-as-input"):
    """src and tgt documents as additional context"""
    missing = [k for k in ("src_lang", "tgt_lang", "src_seg", "tgt_seg") if k not in entry]
    if missing:
        raise KeyError(f"Missing keys in entry: {missing}")

    prompt_dic = _get_templates().get(level, None)
    if prompt_dic is None:
        raise ValueError(f"No template found for the given level={level}")

    params = {k: entry[k] for k in ("src_lang", "tgt_lang", "src_seg", "tgt_seg")}
    lang_map = {v: k for k, v in LANG_MAP.items()}
    lp = f'{lang_map[entry["src_lang"]]}-{lang_map[entry["tgt_lang"]]}'
    doc_id = str(entry["new_doc_id"])

    src_doc = load_doc_from_json(entry["wmt_year"], lp, "src", doc_id)
    tgt_doc = load_doc_from_json(entry["wmt_year"], lp, "tgt", doc_id, system=entry["system"])

    if level == "seg-as-input":
        params["src_seg"] = entry["src_seg"]
        params["tgt_seg"] = entry["tgt_seg"]
    elif level == "doc-as-context":
        params["src_doc"] = src_doc
        params["tgt_doc"] = tgt_doc
    elif level == "doc-as-input":
        params["src_seg"] = src_doc
        params["tgt_seg"] = tgt_doc
    else:
        raise ValueError

    user = prompt_dic["user"].format(**params)
    system = prompt_dic["system"].format(tgt_lang=entry["tgt_lang"])
    return system, user