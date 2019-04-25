#!/usr/bin/env bash

#BERT_BASE_DIR=/Users/alchemistlee/projects/alchemistlee/MagiRobe/bert/uncased_L-12_H-768_A-12
BERT_BASE_DIR=/root/bert_data/data/bert_model/bert/uncased_L-12_H-768_A-12
#GLUE_DIR=/Users/alchemistlee/Personal/projects/tigerye/py-toolkit/glue_data
GLUE_DIR=/root/bert_data/data/glue_data

TRAINED_CLASSIFIER=$BERT_BASE_DIR/bert_model.ckpt
OUTPUT_DIR=/root/bert_out/

#SCRIPT_DIR=/Users/alchemistlee/Personal/projects/tigerye/bert
SCRIPT_DIR=/home/yifan/anywhere/bert

python3 $SCRIPT_DIR/run_classifier.py \
  --task_name=MRPC \
  --do_predict=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=$OUTPUT_DIR