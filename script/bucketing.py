import os
import numpy as np
import re
import json

from utils import read_jsonl, write_jsonl


def bucketing(doc_len):
    p33, p66 = np.percentile(doc_len, [33, 66])
    def bucket(n):
        if n <= p33: return "short"
        if n <= p66: return "medium"
        return "long"
    return [bucket(n) for n in doc_len]


def doc_bucket_mapping():
    years = {"24": "data/wmt24pp", "25": "data/wmt25"}

    doc_len = {}
    for y, year_dir in years.items():
        src_dir = os.path.join(year_dir, "src_docs")
        for file in os.listdir(src_dir):
            if file.endswith(".txt"):
                with open(os.path.join(src_dir, file), "r", encoding="utf-8") as f:
                    doc = [l.strip() for l in f]
                tokens = [re.sub(r"[^\w]", "", w, flags=re.UNICODE) for l in doc for w in l.split()]
                doc_len[f"{y}_{file}"] = len([t for t in tokens if t and not t.isdigit()])

    buckets = bucketing(list(doc_len.values()))
    doc_bucket_map = dict(zip(doc_len.keys(), buckets))

    for y, year_dir in years.items():
        year_map = {k.removeprefix(f"{y}_"): v for k, v in doc_bucket_map.items() if k.startswith(f"{y}_")}
        with open(os.path.join(year_dir, "bucket_map.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(year_map, ensure_ascii=False) + "\n")

        data = read_jsonl(os.path.join(year_dir, "output.jsonl"))
        for line in data:
            line["bucket_id"] = doc_bucket_map[f"{y}_{line['doc_id']}.txt"]
        write_jsonl(os.path.join(year_dir, "output.jsonl"), data)


if __name__ == "__main__":
    doc_bucket_mapping()