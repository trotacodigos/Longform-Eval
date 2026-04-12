"""
Llama-4-Maverick inference script for explicit context annotation.
Extracts genre, participant relationships, and register/formality
from source documents for APEexp context construction.
"""

import json
import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from docape.prompt.rag_prompter import build_prompt
from docape.utils import read_json, read_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


MAX_INPUT_TOKENS = 120_000 


def load_model(model_path: str):
    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    logger.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    logger.info("Model loaded.")
    return tokenizer, model


def run_inference(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    # Check input length and log a warning if it exceeds the model's safe limit
    input_len = inputs.shape[-1]
    if input_len > MAX_INPUT_TOKENS:
        logger.warning(
            f"Input length {input_len} tokens exceeds safe limit "
            f"({MAX_INPUT_TOKENS}). Document may be truncated by the model."
        )
    else:
        logger.debug(f"Input length: {input_len} tokens.")

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=512,
            do_sample=False, # greedy decoding for deterministic output
        )

    response = tokenizer.decode(
        outputs[0][inputs.shape[-1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def parse_json_output(raw: str) -> dict:
    """Strip markdown fences and parse JSON."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        )
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e}\nRaw output:\n{raw}")
        return {"_raw": raw, "_error": str(e)}


# Main
def process_dataset(
    tokenizer,
    model,
    input_path: str,
    output_path: str,
):
    input_path  = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc_dic = read_json(input_path / "src_doc.json")
    entries = read_jsonl(input_path / "input.jsonl") # get domain information
    id2entry = {str(entry["new_doc_id"]): entry for entry in entries}

    logger.info(f"Loaded {len(entries)} entries and {len(doc_dic)} docs from {input_path}")

    with open(output_path / "doc_info.json", "w", encoding="utf-8") as out_f:
        for i, (doc_id, doc) in enumerate(doc_dic.items()):
            entry = {"doc_id": doc_id, "src_doc": doc, "domain": id2entry.get(str(doc_id), {}).get("domain", "unknown")}

            logger.info(
                f"[{i+1}/{len(doc_dic)}] Processing doc_id={doc_id} "
                f"(genre={entry['domain']})"
            )

            system_prompt, user_prompt = build_prompt(entry)
            raw = run_inference(model, tokenizer, system_prompt, user_prompt)
            parsed = parse_json_output(raw)

            out_f.write(json.dumps({entry["doc_id"]: parsed}, ensure_ascii=False) + "\n")

    logger.info(f"Done. Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Llama-4-Maverick inference for explicit context annotation."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to wmt year/language pair/",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to wmt year/language pair/",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        help="HuggingFace model path or ID",
    )
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_path)

    process_dataset(
        tokenizer=tokenizer,
        model=model,
        input_path=args.input,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()