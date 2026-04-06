#!/bin/bash

fdir="data/wmt25/preliminary"
indir=$fdir/input
outdir=$fdir/output

for file in "$indir"/*.jsonl; do
    filename=$(basename "$file")
    echo "Processing $filename ..."
    for k in 1 3 5 10; do
        mkdir -p "$outdir/k${k}"
        /home/ssu/Documents/3_docAPE_low/.venv/bin/python script/labse_retrieval.py \
        --input "$file" \
        --output "$outdir/k${k}/$filename" \
        --k $k
    done
done