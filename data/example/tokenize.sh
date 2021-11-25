#!/bin/bash

python -m \
    knover.tools.pre_tokenize \
    --vocab_path ../../package/dialog_en/vocab.txt \
    --spm_model_file ../../package/dialog_en/spm.model \
    --input_file ./val_wow_seen_unseen.tsv \
    --output_file ./val_wow_seen_unseen.tokenized.tsv
