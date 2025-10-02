#!/bin/bash

python generate.py \
  --config config/models.yaml \
  --input_file unittest/dummy_in.jsonl \
  --output_dir unittest \
  --models "gpt-4o" \
  --with_doc \
  --max_workers 1