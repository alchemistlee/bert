import os




BERT_BASE_DIR="/root/bert_data/data/bert_model/bert/uncased_L-12_H-768_A-12"
DATA_DIR="/root/bert_data/data/news_multi_label/news_title/"
OUTPUT_DIR="/root/bert_out/news_title"



VOCAB_FILE=os.path.join(BERT_BASE_DIR, "vocab.txt")
BERT_CONFIG_FILE=os.path.join(BERT_BASE_DIR, "bert_config.json") 
INIT_CKPT=OUTPUT_DIR
MAX_SEQ_LENGTH=40
LABEL_SIZE = 5
DO_LOWER_CASE=True 