from pathlib import Path
import argparse
import json
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from script.utils import read_jsonl
from script.ape_prompt import build_prompt
from models import REGISTRY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--level", required=True, default="seg-as-input")
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_params", default="", help="modeling parameters saved in .json")
    args = ap.parse_args()

    input_path = Path(args.input_file)
    out_dir = Path(args.output_dir)

    if args.model not in REGISTRY:
        raise ValueError(f"Unknown model '{args.model}'. Available: {list(REGISTRY.keys())}")
    model = REGISTRY[args.model]()

    in_data = read_jsonl(input_path)
    wmt_year = input_path.parts[1]

    out_data = read_jsonl(input_path.parent / "output.jsonl")
    out_data = {(l["seg_id"], l["doc_id"], l["system"]): l for l in out_data}

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{wmt_year}_{args.model}.jsonl"

    # Build prompts for all valid entries
    batch_items = []  # list of (out_row, system_prompt, user_prompt)
    for entry in tqdm(in_data, desc="Building prompts"):
        entry["wmt_year"] = wmt_year
        out_row = out_data.get((entry["seg_id"], entry["doc_id"], entry["system"]))
        if out_row is None:
            continue
        try:
            system, user = build_prompt(entry, level=args.level)
            batch_items.append((out_row, system, user))
        except Exception as e:
            print(f"Prompt error for seg_id={entry.get('seg_id')}: {e}")

    # Submit as a batch and write results
        # Submit as a batch and write results
    prompts = [(system, user) for _, system, user in batch_items]

    # Optimization for doc-as-input: many prompts share the same input since doc 
    # is the same for all segments in a document. 
    if args.level == "doc-as-input":
        seen: dict[tuple, int] = {}
        unique_prompts: list[tuple] = []
        prompt_to_idx: list[int] = []
        for p in prompts:
            if p not in seen:
                seen[p] = len(unique_prompts)
                unique_prompts.append(p)
            prompt_to_idx.append(seen[p])
        unique_results = model.generate_batch(unique_prompts)
        results = [unique_results[i] for i in prompt_to_idx]
    else:
        results = model.generate_batch(prompts)

    with open(out_path, "w", encoding="utf-8") as f:
        for (out_row, _, _), result in zip(batch_items, results):
            if result is None:
                continue
            text, usage = result
            out_row["mt_pe_seg"] = text
            out_row["level"] = args.level
            out_row["usage"] = {k: usage.get(k) for k in ("input_token", "output_token", "latency_sec")}
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
