#!/bin/bash

lp="en-ko_KR"

/home/ssu/Documents/3_docAPE_low/.venv/bin/python infer.py \
  --model "gemini-2.5-pro" \
  --input_file data/wmt24pp/$lp/dummy/dummy_in.jsonl \
  --output_dir "data/outputs/dummy" \
  --has_doc