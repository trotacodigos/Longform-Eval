#!/bin/bash

lp="en-ko_KR"
level="seg-as-input"
dataset="wmt24pp"

#/home/ssu/Documents/3_docAPE_low/.venv/bin/python infer.py \
python infer.py \
  --model "hunyuan-mt-7b" \
  --level $level \
  --input_file  data/$dataset/$lp/input.jsonl \
  --output_dir data/outputs/$level