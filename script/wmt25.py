import pandas as pd
import os, re
import json
import numpy as np               
from utils import LANG_MAP, ALLOWED_DOMAINS, read_jsonl, write_jsonl

LP = ("en-ko_KR", "en-zh_CN")
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(_HERE, "../../../data/wmt25-general-mt/data/")
OUT_DIR = os.path.join(_HERE, "../data/wmt25/")

def read_hypotheses(lp: str):
    hyp_dir = os.path.join(ROOT, "systems")
    hypothesis = []
    for file in os.listdir(hyp_dir):
        if not file.endswith(".jsonl"):
            continue
        sys_name = file.split(".")[0]
        with open(os.path.join(hyp_dir, file), "r", encoding="utf-8") as f:
            i = 0
            for line in f:
                entry = json.loads(line)
                if entry["doc_id"].startswith(lp):
                    hypothesis.append({
                        "new_doc_id": i,
                        "doc_id": entry["doc_id"],
                        "tgt_lang": LANG_MAP.get(lp.split("-")[-1]),
                        "tgt_doc": entry["hypothesis"],
                        "system": sys_name,
                    })
                    i += 1
    return hypothesis

def read_source(lp: str, src_cache: list):
    source = []
    for entry in src_cache:
        if entry["collection_id"] != "general":
            continue
        if entry["doc_id"].startswith(lp):
            source.append({
                "doc_id": entry["doc_id"],
                "domain": entry["domain"],
                "src_lang": LANG_MAP.get(entry["src_lang"]),
                "src_doc": entry["src_text"],
                "ref_doc": entry.get("refs", {}).get("refA", {}).get("ref", ""),
            })
    return source

def merge_data(lp: str, src_cache: list):
    hyp = read_hypotheses(lp)
    src = read_source(lp, src_cache)

    hyp_ = pd.DataFrame(hyp)
    src_ = pd.DataFrame(src)

    df = pd.merge(src_, hyp_, on=["doc_id"])

    # filter out
    df = df[df["domain"].isin(ALLOWED_DOMAINS)]
    df["doc_id"] = df["doc_id"].apply(lambda x: x.split("_#_")[-1])
    return df[["new_doc_id", "doc_id", "domain", "src_lang", "tgt_lang",
               "system", "src_doc", "tgt_doc", "ref_doc"]]


def create_bucket_id(df, lp: str):
    def bucketing(doc_len):
        p33, p66 = np.percentile(doc_len, [33, 66])
        def bucket(n):
            if n <= p33: return "short"
            if n <= p66: return "medium"
            return "long"
        return [bucket(n) for n in doc_len]
    
    df_ = df.drop_duplicates(subset="doc_id")[["new_doc_id", "src_doc"]].copy()
    df_["token_len"] = df_["src_doc"].apply(
        lambda x: len([w for w in x.split() if re.sub(r"[^\w]", "", w, flags=re.UNICODE)])
    )
    df_["bucket_id"] = bucketing(df_["token_len"].tolist())

    doc2bucket = df_.set_index("new_doc_id")["bucket_id"].to_dict()
    bucket_ids = df["new_doc_id"].map(doc2bucket)
    df.insert(2, "bucket_id", bucket_ids)

    src_doc = df_.set_index("new_doc_id")["src_doc"].to_dict()

    tgt_dedup = df.drop_duplicates(subset=["new_doc_id", "system"])[["new_doc_id", "system", "tgt_doc"]]
    tgt_doc = {
        str(doc_id): grp.set_index("system")["tgt_doc"].to_dict()
        for doc_id, grp in tgt_dedup.groupby("new_doc_id")
    }

    # Save documents
    lp_dir = os.path.join(OUT_DIR, lp)
    os.makedirs(lp_dir, exist_ok=True)
    with open(os.path.join(lp_dir, "src_doc.json"), "w", encoding="utf-8") as f:
        json.dump(src_doc, f, ensure_ascii=False)

    with open(os.path.join(lp_dir, "tgt_doc.json"), "w", encoding="utf-8") as f:
        json.dump(tgt_doc, f, ensure_ascii=False)

    return df

def save_dummy(df_in, df_out, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    sample_idx = df_in.sample(3).index

    write_jsonl(os.path.join(out_dir, "dummy_in.jsonl"), df_in.loc[sample_idx].to_dict(orient="records"))
    write_jsonl(os.path.join(out_dir, "dummy_out.jsonl"), df_out.loc[sample_idx].to_dict(orient="records"))


def main():
    src_cache = read_jsonl(os.path.join(ROOT, "wmt25-genmt.jsonl"))

    for lp in LP:
        out_dir = os.path.join(OUT_DIR, lp)
        os.makedirs(out_dir, exist_ok=True)

        org_df = merge_data(lp, src_cache)
        doc_df = create_bucket_id(org_df, lp)

        # Split into segments
        df = (doc_df
            .assign(
                src_seg=lambda d: d["src_doc"].str.split("\n\n"),
                tgt_seg=lambda d: d["tgt_doc"].str.split("\n\n"),
                ref_seg=lambda d: d["ref_doc"].str.split("\n\n"),
            )
            .explode(["src_seg", "tgt_seg", "ref_seg"])
            .reset_index(drop=True)
        )
        df.index.name = "seg_id"
        df = df.reset_index()

        inputs = df[["seg_id", "new_doc_id", "doc_id", "domain", "system",
                    "src_lang", "tgt_lang", "src_seg", "tgt_seg"]]
        outputs = df[["seg_id", "doc_id", "bucket_id", "system", "ref_seg"]]

        write_jsonl(os.path.join(out_dir, "input.jsonl"), inputs.to_dict(orient="records"))
        write_jsonl(os.path.join(out_dir, "output.jsonl"), outputs.to_dict(orient="records"))

        # save samples
        save_dummy(inputs, outputs, os.path.join("../data/dummy"))

if __name__ == "__main__":
    main()
    

