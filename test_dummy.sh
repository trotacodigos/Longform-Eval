#!/bin/bash

lp="en-ko_KR"
level="seg-as-input"
dataset="wmt24pp"

#/home/ssu/Documents/3_docAPE_low/.venv/bin/python infer.py \
python infer.py \
  --model "tower-plus-72b" \
  --level $level \
  --input_file data/$dataset/$lp/dummy/dummy_in.jsonl \
  --output_dir data/outputs/dummy
  #--input_file data/$dataset/$lp/input.jsonl \
  #--output_dir data/outputs/$level