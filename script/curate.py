from pathlib import Path
import argparse
import json
import os
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from docape.utils import read_jsonl, LANG_MAP
from docape.prompt.cur_prompter import build_prompt
from models import REGISTRY


def build_batch(in_data, context: str, ctx_dir: str = None):
    """Returns list of (entry, system_prompt, user_prompt)."""
    batch_items = []
    for entry in tqdm(in_data, desc="Building prompts"):
        try:
            system_prompt, user_prompt = build_prompt(entry, context=context, ctx_dir=ctx_dir)
            batch_items.append((entry, system_prompt, user_prompt))
        except Exception as e:
            print(f"Prompt error for seg_id={entry.get('seg_id')}: {e}")
    return batch_items


def write_results(batch_items, results, out_file):
    with open(out_file, "w", encoding="utf-8") as f:
        for (entry, _, _), result in zip(batch_items, results):
            if result is None:
                continue
            text, usage = result
            record = {
                "doc_id": entry["doc_id"],
                "seg_id": entry["seg_id"],
                "collection_id": entry["collection_id"],
                "mt_pe_seg": text,
                "tgt_seg": entry["tgt_seg"],
                "usage": {k: usage.get(k) for k in ("input_token", "output_token", "latency_sec")},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--context", default=None, help="Context for the RAG prompt: 'seq', 'seg', 'exp', 'rel', or 'full'")
    ap.add_argument("--collection_id", help="wmt24pp or wmt25")
    ap.add_argument("--lang_pair", help="en-ko_KR or en-zh_CN")
    ap.add_argument("--output_dir", default="data/outputs")
    ap.add_argument("--model_params", default="", help="modeling parameters saved in .json")
    args = ap.parse_args()

    # Set dataset paths based on year and language pair
    if args.context == "seg":
        input_file = Path(f"data/{args.collection_id}/{args.lang_pair}/input.jsonl")
    else:
        input_file = Path(f"data/{args.collection_id}/{args.lang_pair}/input_{args.context}.jsonl")

    # Set context directiory
    if args.context in ["full", "exp"]:
        ctx_dir = Path(f"data/{args.collection_id}/{args.lang_pair}/")
    else:
        ctx_dir = None

    src_lang, tgt_lang = args.lang_pair.split("-")
    src_lang = LANG_MAP.get(src_lang)
    tgt_lang = LANG_MAP.get(tgt_lang)

    # set path
    out_dir = Path(args.output_dir) / "ctx_curation" / args.context
    os.makedirs(out_dir, exist_ok=True)

    if args.model not in REGISTRY:
        raise ValueError(f"Unknown model '{args.model}'. Available: {list(REGISTRY.keys())}")
    model = REGISTRY[args.model]()

    in_data = read_jsonl(input_file)
    batch_items = build_batch(in_data, args.context, ctx_dir=ctx_dir)
    results = model.generate_batch([(sp, up) for _, sp, up in batch_items])
    write_results(batch_items, results, out_dir / f"{args.collection_id}_{args.model}.jsonl")


if __name__ == "__main__":
    main()
