import json
from pathlib import Path
from typing import Tuple

TEMPLATE = "prompts/template.jsonl"
LANG_CODE = {"en": "English", "ko": "Korean"}
_CACHED_TEMPLATES = None

def _get_templates():
    global _CACHED_TEMPLATES
    if _CACHED_TEMPLATES is None:
        with open(TEMPLATE, "r", encoding="utf-8") as f:
            _CACHED_TEMPLATES = [json.loads(l) for l in f.readlines()]
    return _CACHED_TEMPLATES

def load_document(doc_id) -> Tuple[str, str]:
    if not doc_id: return "", ""
    src_path = Path("data/inputs/src_docs") / f"{doc_id}.txt"
    tgt_path = Path("data/inputs/tgt_docs") / f"{doc_id}.txt"
    src_doc = src_path.read_text(encoding="utf-8") if src_path.exists() else ""
    tgt_doc = tgt_path.read_text(encoding="utf-8") if tgt_path.exists() else ""
    return src_doc, tgt_doc

def build_prompt(row: dict, has_doc=True):
    # Schema Validation
    required = ["src_lang", "tgt_lang", "src_seg", "tgt_seg"]
    for k in required:
        if k not in row: raise KeyError(f"Missing key in data: {k}")

    src_lang = LANG_CODE[row["src_lang"]]
    tgt_lang = LANG_CODE[row["tgt_lang"]]
    
    src_doc, tgt_doc = "", ""
    if has_doc:
        src_doc, tgt_doc = load_document(row.get('doc_id'))
    
    templates = _get_templates()
    template_row = next((t for t in templates if t["has_doc"] == has_doc), None)
    
    user_kwargs = {
        "src_lang": src_lang, "tgt_lang": tgt_lang,
        "src_seg": row["src_seg"], "tgt_seg": row["tgt_seg"]
    }
    if has_doc:
        user_kwargs.update({"src_doc": src_doc, "tgt_doc": tgt_doc})
        
    user = template_row["user"].format(**user_kwargs)
    system = template_row["system"].format(tgt_lang=tgt_lang)
    
    return system, user