from docape.prompt.ape_prompter import _get_templates
from docape.utils import LANG_MAP, read_json

import os

def build_prompt(entry: dict, context: str, ctx_dir: str = None):
    assert context in ("seg", "seq", "rel", "exp", "full"), f"Unsupported context: {context}"
    record = {
        "src_lang": entry["src_lang"],
        "tgt_lang": entry["tgt_lang"],
        "src_seg": entry["src_seg"],
        "tgt_seg": entry["tgt_seg"],
        }
    if context == "full":
        src_doc = read_json(os.path.join(ctx_dir, "src_doc.json"))
        tgt_doc = read_json(os.path.join(ctx_dir, "tgt_doc.json"))
        record = {**record, "src_doc": src_doc[entry["new_doc_id"]], "tgt_doc": tgt_doc[entry["new_doc_id"]]}

    elif context == "exp":
        doc_info = read_json(os.path.join(ctx_dir, "doc_info.json"))
        record = {**record, "doc_summary": doc_info["doc_id"]}

    elif context in ("rel", "seq"):
        record = {**record, "src_ctx": entry[context]["src_ctx"], "tgt_ctx": entry[context]["tgt_ctx"]}
    else:
        pass

    prompt_dic = _get_templates().get(f"ape-{context}")
    if prompt_dic is None:
        raise ValueError(f"No template found for context='{context}'")
    user = prompt_dic["user"].format(**record) + f"\n\nMUST output in this format:\n<pe>{entry['tgt_lang']} corrected sentence only</pe>"
    system = prompt_dic["system"].format(tgt_lang=entry["tgt_lang"])
    return system, user


def set_lp_dir(src_lang, tgt_lang):
    lang_map = {v: k for k, v in LANG_MAP.items()}
    return f"{lang_map[src_lang]}-{lang_map[tgt_lang]}"