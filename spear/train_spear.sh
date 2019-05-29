#!/usr/bin/env bash

#BERT_BASE_DIR=/Users/alchemistlee/projects/alchemistlee/MagiRobe/bert/uncased_L-12_H-768_A-12
#GLUE_DIR=/Users/alchemistlee/Personal/projects/tigerye/py-toolkit/glue_data
#OUT_DIR=/tmp/mrpc_output/train/

BERT_BASE_DIR=/root/bert_data/data/bert_model/bert/uncased_L-12_H-768_A-12
DATA_DIR=/root/bert_data/data/news_multi_label/news_title/
OUTPUT_DIR=/root/bert_out/news_title

SCRIPT_DIR=/root/workspace/bert/spear


python3 $SCRIPT_DIR/multi_lable_classifier_v2.py \
  --do_train=True \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_ckpt=$BERT_BASE_DIR/bert_model.ckpt \
  --output_dir=$OUTPUT_DIR