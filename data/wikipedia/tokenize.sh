#!/bin/bash

nohup python -m \
    knover.tools.pre_tokenize \
    --vocab_path ../../package/dialog_en/vocab.txt \
    --spm_model_file ../../package/dialog_en/spm.model \
    --input_file ./$1 \
    --output_file ./$1.tokenized > logs/log_$1 &
