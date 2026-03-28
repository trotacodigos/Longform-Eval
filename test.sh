#!/bin/bash

lp="en-ko_KR"

python infer.py \
  --model "command-a-translate-08-2025" \
  --input_file data/wmt24pp/$lp/dummy/dummy_in.jsonl \
  --output_dir "data/outputs/dummy" \
  --has_doc