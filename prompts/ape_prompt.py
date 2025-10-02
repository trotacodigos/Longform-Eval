import json
from pathlib import Path
from typing import Tuple


TEMPLATE = "prompts/template.jsonl"

LANG_CODE = {
    "en": "English",
    "ko": "Korean",
    
}

def load_document(doc_id) -> Tuple[str, str]:
    if not doc_id:
        return "", ""

    src_path = Path("data/inputs/src_docs") / f"{doc_id}.txt"
    tgt_path = Path("data/inputs/tgt_docs") / f"{doc_id}.txt"

    src_doc = src_path.read_text(encoding="utf-8") if src_path.exists() else ""
    tgt_doc = tgt_path.read_text(encoding="utf-8") if tgt_path.exists() else ""
    return src_doc, tgt_doc


def build_prompt(row: dict, has_doc=True):
    src_lang = row["src_lang"]
    src_lang = LANG_CODE[src_lang]
    tgt_lang = row["tgt_lang"]
    tgt_lang = LANG_CODE[tgt_lang]
    
    if has_doc:
        # Open document
        doc_id = row['doc_id']
        src_doc, tgt_doc = load_document(doc_id)
    
    # Open prompt
    with open(TEMPLATE, "r", encoding="utf-8") as f:
        template = [json.loads(l) for l in f.readlines()]
    template = [row for row in template if row["has_doc"] == has_doc][0]
    
    # Create user prompt
    if has_doc:
        user = template["user"].format(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            src_seg=row["src_seg"],
            tgt_seg=row["tgt_seg"],
            src_doc=src_doc,
            tgt_doc=tgt_doc,
        )
    else:
        user = template["user"].format(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            src_seg=row["src_seg"],
            tgt_seg=row["tgt_seg"],
        )
    system = template["system"].format(tgt_lang=tgt_lang)
    
    return system, user