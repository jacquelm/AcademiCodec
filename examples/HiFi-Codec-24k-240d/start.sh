
#!/bin/bash
source path.sh
set -e

log_root="logs"

input_training_file="train.lst" # .lst save the wav path.
input_validation_file="valid_256.lst"

#mode=debug
mode=train

if [ "${mode}" == "debug" ]; then
  ## debug
  echo "Debug"
  log_root=${log_root}_debug
  export CUDA_VISIBLE_DEVICES=0
  python ${BIN_DIR}/train.py \
    --config config_24k_240d.json \
    --checkpoint_path ${log_root} \
    --input_training_file ${input_training_file} \
    --input_validation_file ${input_validation_file} \
    --checkpoint_interval 100 \
    --summary_interval 10 \
    --validation_interval 100 \

elif [ "$mode" == "train" ]; then
  ## train
  echo "Train model..."
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  python ${BIN_DIR}/train.py \
    --config config_24k_240d.json \
    --checkpoint_path ${log_root} \
    --input_training_file ${input_training_file} \
    --input_validation_file ${input_validation_file} \
    --checkpoint_interval 5000 \
    --summary_interval 100 \
    --validation_interval 5000
fi
