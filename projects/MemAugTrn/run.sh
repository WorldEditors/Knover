RUN_FILE=$1
SUFFIX=$2
CUDA_DEVICES=$3

PROJECT_DIR=./projects/MemAugTrn
ROOTPATH=./
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
bash $ROOTPATH/scripts/local/job.sh $RUN_FILE 
#nohup bash $ROOTPATH/scripts/local/job.sh $RUN_FILE > $PROJECT_DIR/results/log.train.$SUFFIX &
