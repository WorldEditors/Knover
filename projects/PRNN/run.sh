RUN_FILE=$1
SUFFIX=$2
CUDA_DEVICES=$3

PROJECT_DIR=/mnt/wangfan/worldEditors/Knover/projects/PRNN
ROOTPATH=/mnt/wangfan/worldEditors/Knover
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
nohup bash $ROOTPATH/scripts/local/job.sh $RUN_FILE > $PROJECT_DIR/results/log.train.$SUFFIX &
