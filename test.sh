#!/bin/bash

lp="en-ko_KR"
level="doc-as-context"
dataset="wmt25"

/home/ssu/Documents/3_docAPE_low/.venv/bin/python infer.py \
  --model "hunyuan-mt-7b" \
  --level $level \
  --input_file  data/$dataset/$lp/input.jsonl \
  --output_dir data/outputs/$level