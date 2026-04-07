#!/bin/bash

lp="en-ko_KR"
level="seg-as-input"
dataset="wmt24pp"

#/home/ssu/Documents/3_docAPE_low/.venv/bin/python infer.py \
python infer.py \
  --model "exaone-4.0-32b" \
  --level $level \
  --input_file data/$dataset/$lp/dummy/dummy_in.jsonl \
  --output_dir data/outputs/dummy