#!/usr/bin/env bash

#BERT_BASE_DIR=/Users/alchemistlee/projects/alchemistlee/MagiRobe/bert/uncased_L-12_H-768_A-12
#GLUE_DIR=/Users/alchemistlee/Personal/projects/tigerye/py-toolkit/glue_data
#OUT_DIR=/tmp/mrpc_output/train/

BERT_BASE_DIR=/root/bert_data/data/bert_model/bert/uncased_L-12_H-768_A-12
GLUE_DIR=/root/bert_data/data/glue_data
OUTPUT_DIR=/root/bert_out/

SCRIPT_DIR=/home/yifan/anywhere/bert


python3 $SCRIPT_DIR/run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUT_DIR