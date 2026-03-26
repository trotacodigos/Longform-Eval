import json
import os
from collections import defaultdict

from utils import write_jsonl, ALLOWED_DOMAINS


ALLOWED_TGT_LANGS = {"zh_CN", "ko_KR"}


def main():
    root = "../../data/wmt25-general-mt/data/"
    out_dir = "data/wmt25/"

    hyp_dir = os.path.join(root, "systems")
    hypothesis = []
    for file in os.listdir(hyp_dir):
        if not file.endswith(".jsonl"):
            continue
        sys_name = file.split(".")[0]
        with open(os.path.join(hyp_dir, file), "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                entry = json.loads(line)
                hypothesis.append({
                    "sample_id": i,
                    "doc_id": entry["doc_id"],
                    "tgt_seg": entry["hypothesis"],
                    "system": sys_name,
                })

    src_file = os.path.join(root, "wmt25-genmt.jsonl")
    id2data = {}
    with open(src_file, encoding="utf-8") as f:
        for line in f:
            l = json.loads(line)
            if (l["collection_id"] == "general"
                    and l["domain"] in ALLOWED_DOMAINS
                    and l["src_lang"] == "en"
                    and l["tgt_lang"] in ALLOWED_TGT_LANGS):
                id2data[l["doc_id"]] = l

    doc_map: dict[str, dict] = defaultdict(dict)
    for doc_id, data in id2data.items():
        lang = data["tgt_lang"]
        doc_map[lang][doc_id] = len(doc_map[lang])

    inputs = []
    outputs = []
    src_docs: dict = {}
    tgt_docs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for sample_id, record in enumerate(r for r in hypothesis if r["doc_id"] in id2data):
        src_line = id2data[record["doc_id"]]
        tgt_lang = src_line["tgt_lang"]
        new_doc_id = doc_map[tgt_lang][record["doc_id"]]
        src_seg = src_line["src_text"]
        if new_doc_id not in src_docs:
            src_docs[new_doc_id] = src_seg
        tgt_docs[tgt_lang][record["system"]][new_doc_id].append(record["tgt_seg"])

        ref_seg = src_line.get("refs", {}).get("refA", {}).get("ref", "")

        inputs.append({
            "sample_id": sample_id,
            "doc_id": new_doc_id,
            "domain": src_line["domain"],
            "system": record["system"],
            "src_lang": src_line["src_lang"],
            "tgt_lang": tgt_lang,
            "src_seg": src_seg,
            "tgt_seg": record["tgt_seg"],
        })
        outputs.append({
            "sample_id": sample_id,
            "doc_id": new_doc_id,
            "src_seg": src_seg,
            "tgt_seg": record["tgt_seg"],
            "ref_seg": ref_seg,
            "human_pe_seg": "",
            "model_pe_seg": "",
            "manual": None,
            "auto": {}
        })

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "doc_id.json"), "w", encoding="utf-8") as f:
        per_lang = {lang: {v: k for k, v in lmap.items()} for lang, lmap in doc_map.items()}
        f.write(json.dumps(per_lang, ensure_ascii=False) + "\n")

    src_dir = os.path.join(out_dir, "src_docs")
    os.makedirs(src_dir, exist_ok=True)
    for doc_id, doc in src_docs.items():
        with open(os.path.join(src_dir, f"{doc_id}.txt"), "w", encoding="utf-8") as f:
            f.write(doc + "\n")

    tgt_dir = os.path.join(out_dir, "tgt_docs")
    for lang, v in tgt_docs.items():
        for sys, vv in v.items():
            sys_dir = os.path.join(tgt_dir, f"en-{lang[:2]}", sys)
            os.makedirs(sys_dir, exist_ok=True)
            for doc_id, doc in vv.items():
                with open(os.path.join(sys_dir, f"{doc_id}.txt"), "w", encoding="utf-8") as f:
                    for line in doc:
                        f.write(line + "\n")

    write_jsonl(os.path.join(out_dir, "input.jsonl"), inputs)
    write_jsonl(os.path.join(out_dir, "output.jsonl"), outputs)


if __name__ == "__main__":
    main()
