RUN_FILE=$0
PROJECT_DIR=$(readlink -f "$(dirname "$0")")
ROOTPATH=$(readlink -f "$(dirname "$0")/../../")
nohup bash $ROOTPATH/scripts/local/job.sh $RUN_FILE > $PROJECT_DIR/run/log.train.rnn.512 &

# job settings
job_script="$ROOTPATH/scripts/single_gpu/train.sh"

# task settings
export CUDA_VISIBLE_DEVICES=0
model=RecursiveModels
task=DialogGeneration

vocab_path="$ROOTPATH/package/dialog_en/vocab.txt"
spm_model_file="$ROOTPATH/package/dialog_en/spm.model"
train_file="$ROOTPATH/data/example/train_wow.tokenized.tsv"
valid_file="$ROOTPATH/data/example/val_wow_seen_unseen.tokenized.tsv"
data_format="tokenized"
#raw, tokenized, numerical
file_format="file"
config_path="$ROOTPATH/projects/PRNN/rnn.json"

# training settings
in_tokens="false"
batch_size=16
lr=1e-4
warmup_steps=500
weight_decay=0.01
num_epochs=10000

train_args="--max_knowledge_len 128 --max_src_len 256 --max_tgt_len 128 --max_seq_len 512 --knowledge_position pre_src --sort_pool_size 0"

log_steps=10
validation_steps=1000
save_steps=10000

log_dir="./log"
save_path="./output"
