#!/bin/bash

model="tower-plus-72b"
for lp in "en-ko_KR" "en-zh_CN"; do
  for dataset in "wmt24pp" "wmt25"; do
    for level in "seg-as-input" "doc-as-input" "doc-as-context"; do
      if [ "$lp" = "en-zh_CN" ] && [ "$dataset" = "wmt24pp" ]; then
        continue
      fi

      /home/ssu/Documents/3_docAPE_low/.venv/bin/python infer.py \
        --model $model \
        --level "$level" \
        --input_file "data/$dataset/$lp/input.jsonl" \
        --output_dir "data/outputs/$level"
    done
  done
done