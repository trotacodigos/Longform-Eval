#!/bin/bash

for year in "wmt24pp" "wmt25"; do
  for lp in "en-ko_KR" "en-zh_CN"; do
    echo "Extracting sequential context for $year / $lp ..."
    python docape/retrieval/doc_seq.py --input data/$year/$lp/input.jsonl \
                                    --output data/$year/$lp/input_seq.jsonl --n 5
  done
done