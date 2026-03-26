import json
import os
from collections import defaultdict

from utils import read_jsonl, write_jsonl, ALLOWED_DOMAINS, LANG_MAP

LP = "en-ko_KR"

def preprocess():
    root = "../../data/wmt24pp"

    # 1. metadata
    records = read_jsonl(os.path.join(root, "metadata", LP, "src.jsonl"))
    records = [{**rec, "sample_id": i} for i, rec in enumerate(records)]

    # 2. domain, doc_id
    with open(os.path.join(root, "documents", f"{LP}.docs"), "r", encoding="utf-8") as f:
        doc = [l.strip().split("\t") for l in f]
    records = [{**meta, "domain": domain, "doc_id": doc_id}
               for meta, (domain, doc_id) in zip(records, doc)]

    # 3. references
    ref_dir = os.path.join(root, "references")
    for file in os.listdir(ref_dir):
        if LP not in file:
            continue
        ref_data = read_jsonl(os.path.join(ref_dir, file))
        field = "ref_seg" if "ref" in file else "pe_seg"
        records = [{**meta, field: ref["target"].strip()}
                   for meta, ref in zip(records, ref_data)]

    # 4. source
    src_data = read_jsonl(os.path.join(root, "sources", f"{LP}.jsonl"))
    records = [{**meta, "src_seg": src["source"].strip()}
               for meta, src in zip(records, src_data)]

    # 5. hypotheses — expand records once per system
    sys_dir = os.path.join(root, "system-outputs", LP)
    final_records = []
    for file in os.listdir(sys_dir):
        if not file.endswith(".jsonl") or "refA" in file or "postedit" in file:
            continue
        sys_data = read_jsonl(os.path.join(sys_dir, file))
        sys_name = file.split(".")[0]
        final_records.extend(
            {**meta, "system": sys_name, "tgt_seg": hyp["hypothesis"].strip()}
            for meta, hyp in zip(records, sys_data)
        )
    return final_records


def main():
    out_dir = "data/wmt24pp/"
    src_lang, tgt_lang = LP.split("-")

    filtered = [
        rec for rec in preprocess()
        if not rec["is_bad_source"] and rec["domain"] in ALLOWED_DOMAINS
    ]

    doc_map = {doc_id: i for i, doc_id in enumerate(dict.fromkeys(rec["doc_id"] for rec in filtered))}

    src_docs = defaultdict(list)
    tgt_docs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    inputs = []
    outputs = []
    seen_samples = set()

    for rec in filtered:
        new_doc_id = doc_map[rec["doc_id"]]
        if rec["sample_id"] not in seen_samples:
            src_docs[new_doc_id].append(rec["src_seg"])
            seen_samples.add(rec["sample_id"])
        tgt_docs[tgt_lang][rec["system"]][new_doc_id].append(rec["tgt_seg"])
        inputs.append({
            "sample_id": rec["sample_id"],
            "doc_id": new_doc_id,
            "domain": rec["domain"],
            "system": rec["system"],
            "src_lang": LANG_MAP[src_lang],
            "tgt_lang": LANG_MAP[tgt_lang],
            "src_seg": rec["src_seg"],
            "tgt_seg": rec["tgt_seg"],
        })
        outputs.append({
            "sample_id": rec["sample_id"],
            "src_seg": rec["src_seg"],
            "tgt_seg": rec["tgt_seg"],
            "ref_seg": rec["ref_seg"],
            "human_pe_seg": rec["pe_seg"],
            "model_pe_seg": "",
            "manual": None,
            "auto": {},
        })

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "doc_id.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps({v: k for k, v in doc_map.items()}, ensure_ascii=False) + "\n")

    src_dir = os.path.join(out_dir, "src_docs")
    os.makedirs(src_dir, exist_ok=True)
    for doc_id, doc in src_docs.items():
        with open(os.path.join(src_dir, f"{doc_id}.txt"), "w", encoding="utf-8") as f:
            for line in doc:
                f.write(line + "\n")

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
