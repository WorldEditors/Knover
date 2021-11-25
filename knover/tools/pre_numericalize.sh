#!/bin/bash

python -m \
    knover.tools.pre_numericalize \
    --vocab_path ./package/dialog_en/vocab.txt \
    --spm_model_file ./package/dialog_en/spm.model \
    --input_file ./data/example/val_wow_seen.tsv \
    --output_file ./data/example/val_wow_seen.numerical.tsv
