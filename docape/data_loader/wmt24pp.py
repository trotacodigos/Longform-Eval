import os
import pandas as pd
import argparse
from docape.utils import read_jsonl, write_jsonl, ALLOWED_DOMAINS, LANG_MAP
from bucketing import create_bucket_id
from wmt25 import save_dummy


_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, "../data/wmt24pp/")

def preprocess(lp: str):
    src_lang, tgt_lang = lp.split("-")
    root = os.path.join(_HERE, "../../../corpora/wmt24pp")

    # 1. metadata
    records = read_jsonl(os.path.join(root, "metadata", lp, "src.jsonl"))
    records = [{**rec, "seg_id": i} for i, rec in enumerate(records)]

    # 2. domain, doc_id
    with open(os.path.join(root, "documents", f"{lp}.docs"), "r", encoding="utf-8") as f:
        doc = [l.strip().split("\t") for l in f]
    records = [{**meta, "domain": domain, "doc_id": doc_id}
               for meta, (domain, doc_id) in zip(records, doc)]

    # 3. references
    ref_dir = os.path.join(root, "references")
    for file in os.listdir(ref_dir):
        if lp not in file:
            continue
        ref_data = read_jsonl(os.path.join(ref_dir, file))
        field = "ref_seg" if "ref" in file else "human_pe_seg"
        records = [{**meta, field: ref["target"].strip()}
                   for meta, ref in zip(records, ref_data)]

    # 4. source
    src_data = read_jsonl(os.path.join(root, "sources", f"{lp}.jsonl"))
    records = [{**meta, "src_seg": src["source"].strip(), 
                "src_lang": LANG_MAP.get(src_lang), "tgt_lang": LANG_MAP.get(tgt_lang)}
               for meta, src in zip(records, src_data)]

    # 5. hypotheses — expand records once per system
    sys_dir = os.path.join(root, "system-outputs", lp)
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

def merge_data(lp: str):
    data = [
        rec for rec in preprocess(lp)
        if not rec["is_bad_source"] and rec["domain"] in ALLOWED_DOMAINS
    ]
    df = pd.DataFrame(data)

    # Generate new_doc_id
    new_doc_id = {doc: i for i, doc in enumerate(df["doc_id"].value_counts().keys())}
    df["new_doc_id"] = df.doc_id.apply(lambda x: new_doc_id.get(x))

    # Generate doc_id map to fit into WMT'25 style
    src = df.drop_duplicates(subset="src_seg").sort_values(["doc_id", "seg_id"])
    src_dic = {}
    for g, frame in src.groupby("new_doc_id"):
        f = frame.sort_values("seg_id")
        src_dic[g] = "\n\n".join(f["src_seg"].tolist())

    df["src_doc"] = df["new_doc_id"].apply(lambda x: src_dic.get(x))

    tgt_dic = {}
    for g, frame in df.groupby(["new_doc_id", "system"]):
        f = frame.sort_values("seg_id")
        tgt_dic[g] = "\n\n".join(f["tgt_seg"].tolist())

    df["tgt_doc"] = df.apply(lambda x: tgt_dic.get((x["new_doc_id"], x["system"])), axis=1)
    create_bucket_id(df, lp, OUT_DIR)

    for col in ("src_doc", "tgt_doc"):
        del df[col]
    return df


def main(lp: str):
    out_dir = os.path.join(OUT_DIR, lp)
    os.makedirs(out_dir, exist_ok=True)

    df = merge_data(lp)
    inputs = df[["seg_id", "new_doc_id", "doc_id", "domain", "system",
                "src_lang", "tgt_lang", "src_seg", "tgt_seg"]]
    outputs = df[["seg_id", "doc_id", "bucket_id", "system", "ref_seg", "human_pe_seg"]]

    write_jsonl(os.path.join(out_dir, "input.jsonl"), inputs.to_dict(orient="records"))
    write_jsonl(os.path.join(out_dir, "output.jsonl"), outputs.to_dict(orient="records"))

    # save samples
    save_dummy(inputs, outputs, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lp", type=str, default="en-ko_KR", help="Language pair to process")
    args = parser.parse_args()
    lp = args.lp
    main(lp)
