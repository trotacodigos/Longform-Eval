import os
import json

from .utils import read_jsonl, LANG_MAP


_CACHED_TEMPLATES = None

def _get_templates():
    global _CACHED_TEMPLATES
    template = "data/template.jsonl"
    if _CACHED_TEMPLATES is None:
        _CACHED_TEMPLATES = read_jsonl(template)
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


def build_prompt(entry: dict, has_doc=True):
    """src and tgt documents as additional context"""
    missing = [k for k in ("src_lang", "tgt_lang", "src_seg", "tgt_seg") if k not in entry]
    if missing:
        raise KeyError(f"Missing keys in entry: {missing}")

    cur_template = next((t for t in _get_templates() if t["has_doc"] == has_doc), None)
    if cur_template is None:
        raise ValueError(f"No template found for has_doc={has_doc}")

    params = {k: entry[k] for k in ("src_lang", "tgt_lang", "src_seg", "tgt_seg")}
    lang_map = {v: k for k, v in LANG_MAP.items()}
    lp = f'{lang_map[entry["src_lang"]]}-{lang_map[entry["tgt_lang"]]}'
    doc_id = str(entry["new_doc_id"])

    src_doc = load_doc_from_json(entry["wmt_year"], lp, "src", doc_id)
    tgt_doc = load_doc_from_json(entry["wmt_year"], lp, "tgt", doc_id, system=entry["system"])

    if has_doc:
        params |= {"src_doc": src_doc, "tgt_doc": tgt_doc}
    else:
        params["src_seg"] = src_doc
        params["tgt_seg"] = tgt_doc

    user = cur_template["user"].format(**params)
    system = cur_template["system"].format(tgt_lang=entry["tgt_lang"])
    return system, user