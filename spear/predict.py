# coding = utf-8

# @time    : 2019/5/22 10:56 AM
# @author  : alchemistlee
# @fileName: multi_lable_classifier_v2.py
# @abstract:

import sys

sys.path.append('../')

import modeling
import os
import tensorflow as tf
import numpy as np
import utils
import collections
import tokenization


flags = tf.flags

FLAGS=tf.app.flags.FLAGS

#input-data
flags.DEFINE_string("data_dir", None,
                    "The input data dir. Should contain the .tsv files (or other data files) for the task.")
flags.DEFINE_string("bert_config_file", None,
                    "The config json file corresponding to the pre-trained BERT model. This specifies the model architecture.")
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("sentence", None, "")

#other-parameter
flags.DEFINE_string("init_ckpt", None,"Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_bool("do_lower_case", True,"Whether to lower case the input text. Should be True for uncased models and False for cased models.")

#hyper-parameter
flags.DEFINE_integer("max_seq_length", 40,
  "max total input sequence length after WordPiece tokenization. Seq longer than this will be truncated, and seq shorter than this will be padded.")



def create_model(bert_config, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  is_training = tf.cast(False, tf.bool)
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    probabilities = tf.nn.sigmoid(logits)  # tf.nn.softmax(logits, axis=-1)
    return probabilities


def convert_single_example(sample, max_seq_length,tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  tokens_a = tokenizer.tokenize(sample)
  if len(tokens_a) > max_seq_length - 2:
    tokens_a = tokens_a[0:(max_seq_length - 2)]

  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)


  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length


  return input_ids,input_mask,segment_ids


def get_input_mask_segment_ids(sentence, max_seq_length, tokenizer):
  batch_input_ids = []
  batch_input_mask = []
  batch_segment_ids = []

  tmp_input_ids, tmp_input_mask, tmp_segment_ids = convert_single_example(sentence, max_seq_length, tokenizer)
  batch_input_ids.append(tmp_input_ids)
  batch_input_mask.append(tmp_input_mask)
  batch_segment_ids.append(tmp_segment_ids)

  return batch_input_ids, batch_input_mask, batch_segment_ids



def predict_it(valid_data, tokenizer):
  """
  evalution on model using validation data
  """
  label_size = 5

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name="input_ids")  # FLAGS.batch_size
  input_mask = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name="input_mask")
  segment_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name="segment_ids")
  label_ids = tf.placeholder(tf.float32, [None, label_size], name="label_ids")

  use_one_hot_embeddings = None
  probabilities = create_model(bert_config, input_ids, input_mask, segment_ids, label_ids, label_size, use_one_hot_embeddings)

  global_step = tf.Variable(0, trainable=False, name="Global_Step")

  gpu_config = tf.ConfigProto()
  gpu_config.gpu_options.allow_growth = True
  sess = tf.Session(config=gpu_config)
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()

  if os.path.exists(FLAGS.init_ckpt):
    print("Checkpoint Exists. Restoring Variables from Checkpoint.")
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.init_ckpt))

  tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)


  num_examples = len(valid_data)

  #batch_size=1 # TODO
  batch_input_ids_, batch_input_mask_, batch_segment_ids = get_input_mask_segment_ids(valid_data[0], FLAGS.max_seq_length,tokenizer)

  feed_dict = {input_ids: batch_input_ids_, input_mask: batch_input_mask_, segment_ids: batch_segment_ids}
  prob = sess.run([probabilities], feed_dict)
  print(prob)
  #predict_labels = utils.get_label_using_logits_batch(prob)
  #print(predict_lables)



def main(_):
  sentence = FLAGS.sentence
  tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
  predict_it([sentence], tokenizer)


if __name__ == '__main__':
  tf.app.run()





