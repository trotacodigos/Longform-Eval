import os
import numpy as np
import re
import json


def create_bucket_id(df, lp: str, out_dir: str):
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
    lp_dir = os.path.join(out_dir, lp)
    os.makedirs(lp_dir, exist_ok=True)
    with open(os.path.join(lp_dir, "src_doc.json"), "w", encoding="utf-8") as f:
        json.dump(src_doc, f, ensure_ascii=False)

    with open(os.path.join(lp_dir, "tgt_doc.json"), "w", encoding="utf-8") as f:
        json.dump(tgt_doc, f, ensure_ascii=False)

    return df