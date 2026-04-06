#!/bin/bash

for k in 1 3 5 10; do
    for context in "high" "low"; do
        echo "Running inference for k=$k ..."
        /home/ssu/Documents/3_docAPE_low/.venv/bin/python infer_ctx.py \
                --model "gemma-3-27b-it" \
                --system "Gemini-2" \
                --context $context \
                -k $k
    done
done