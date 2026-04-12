"""
Sequential context extraction for document-level APE.

For each segment S_i in a document, extracts the preceding N sentences
as context (src_ctx, tgt_ctx).

Usage:
    python docape/retrieval/doc_seq.py --input data.jsonl --output output.jsonl --n 5
"""
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from docape.utils import write_jsonl


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def group_by_doc(entries: List[dict]) -> dict:
    """Group flat per-segment entries into {(system, doc_id): [{seg fields}, ...]}."""
    groups = defaultdict(list)
    for e in entries:
        groups[(e["system"], e["doc_id"])].append({
            "seg_id": e["seg_id"], "system": e["system"],
            "src_lang": e["src_lang"], "tgt_lang": e["tgt_lang"],
            "src_seg": e["src_seg"], "tgt_seg": e["tgt_seg"],
            })
    return groups


def extract_sequential_context(
    doc_segments: List[dict],
    n: int,
) -> List[dict]:
    """
    doc_segments: list of {"src": str, "tgt": str}
    Returns one record per segment that has at least n preceding segments.
    """
    results = []
    for i, seg in enumerate(doc_segments):
        if i < n:
            continue
        preceding = doc_segments[i - n : i]
        src_ctx = "\n".join(s["src_seg"] for s in preceding)
        tgt_ctx = "\n".join(s["tgt_seg"] for s in preceding)
        results.append({
            **seg, "seq": {"src_ctx": src_ctx, "tgt_ctx": tgt_ctx},
        })
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Sequential context extraction for APE"
    )
    parser.add_argument("--input", required=True,
        help="Input JSONL — either grouped {doc_id, segments:[{src,tgt}]} "
             "or flat per-segment {doc_id, src_seg, tgt_seg} format.")
    parser.add_argument("--output", required=True,
        help="Output JSONL file with sequential contexts per segment.")
    parser.add_argument("--n", type=int, default=5,
        help="Number of preceding segments to use as context (default: 5).")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    raw = read_jsonl(args.input)

    # Auto-detect format
    if "segments" in raw[0]:
        documents = {doc["doc_id"]: doc["segments"] for doc in raw}
    else:
        documents = group_by_doc(raw)

    output_records = []
    for (system, doc_id), doc_segments in documents.items():
        print(f"Processing system={system} doc_id={doc_id} ({len(doc_segments)} segments)...")
        contexts = extract_sequential_context(doc_segments, n=args.n)
        for ctx in contexts:
            record = {"doc_id": doc_id}
            record.update(ctx)
            output_records.append(record)

    write_jsonl(args.output, output_records)
    print(f"Saved {len(output_records)} records to {args.output}")


if __name__ == "__main__":
    main()
