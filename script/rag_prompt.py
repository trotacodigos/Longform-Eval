from .ape_prompt import _get_templates

import json
import os

def load_ctx_from_json(k: int, direction: str, doc_id: str, seg_id: str, system: str):
    """{doc_id_seg_id: ctx}
        or
        {system: {doc_id_seg_id: ctx}}
    """
    fname = os.path.join("data", f"wmt25/preliminary/{direction}/k{k}.json")
    with open(fname, "r", encoding="utf-8") as f:
        ctx_data = json.load(f)

    key = f"{doc_id}_{seg_id}"
    if direction == "src":
        ctx_line = ctx_data.get(key, "")
    elif direction == "tgt":
        ctx_data = ctx_data.get(system, {})
        if not ctx_data:
            raise ValueError(f"No context found for system={system} in {fname}")
        ctx_line = ctx_data.get(key, "")
    else:
        raise ValueError(f"Unknown direction: {direction!r}")

    if not ctx_line:
        raise ValueError(f"No context found for key={key} in {fname}")
    return ctx_line


def build_rag_prompt(entry: dict, level: str, k: int = 3):
    assert level in ("seg", "doc"), f"Unsupported level: {level}"

    missing = [key for key in ("src_lang", "tgt_lang", "src_seg", "tgt_seg") if key not in entry]
    if missing:
        raise KeyError(f"Missing keys in entry: {missing}")

    prompt_dic = _get_templates().get(level, None)
    if prompt_dic is None:
        raise ValueError(f"No template found for the given level={level}")

    params = {key: entry[key] for key in ("src_lang", "tgt_lang", "src_seg", "tgt_seg")}

    if level == "doc":
        assert "src_ctx" in entry and "tgt_ctx" in entry, "Context keys missing in template for doc-level prompt"
        params["src_ctx"] = load_ctx_from_json("src", str(entry["new_doc_id"]), entry["seg_id"], entry["system"])
        params["tgt_ctx"] = load_ctx_from_json("tgt", str(entry["new_doc_id"]), entry["seg_id"], entry["system"])

    params["src_seg"] = entry["src_seg"]
    params["tgt_seg"] = entry["tgt_seg"]

    user = prompt_dic["user"].format(**params)
    system = prompt_dic["system"].format(tgt_lang=entry["tgt_lang"])
    return system, user
