# coding = utf-8

# @time    : 2019/6/4 3:15 PM
# @author  : alchemistlee
# @fileName: Spear.py
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
import config
import time


class Spear(object):

  def __init__(self):
    self.tokenizer = tokenization.FullTokenizer(vocab_file=config.VOCAB_FILE, do_lower_case=config.DO_LOWER_CASE)
    self.bert_config = modeling.BertConfig.from_json_file(config.BERT_CONFIG_FILE)
    self.is_training = tf.cast(False, tf.bool)
    self._initialize_gpu_config()

    self.input_ids = tf.placeholder(tf.int32, [None, config.MAX_SEQ_LENGTH], name="input_ids")  # FLAGS.batch_size
    self.input_mask = tf.placeholder(tf.int32, [None, config.MAX_SEQ_LENGTH], name="input_mask")
    self.segment_ids = tf.placeholder(tf.int32, [None, config.MAX_SEQ_LENGTH], name="segment_ids")

    self.probabilities = self._create_model(self.input_ids, self.input_mask, self.segment_ids)

    self.sess = tf.Session(config=self.gpu_config)
    self.sess.run(tf.global_variables_initializer())
    self.saver = tf.train.Saver()

    if os.path.exists(config.INIT_CKPT):
      print("Checkpoint Exists. Restoring Variables from Checkpoint.")
      self.saver.restore(self.sess, tf.train.latest_checkpoint(config.INIT_CKPT))

  def _initialize_gpu_config(self):

    self.gpu_config = tf.ConfigProto()
    self.gpu_config.gpu_options.allow_growth = True

  def _create_model(self, input_ids, input_mask, segment_ids):

    """Creates a classification model."""
    model = modeling.BertModel(
      config=self.bert_config,
      is_training=self.is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=None)

    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
      "output_weights", [config.LABEL_SIZE, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
      "output_bias", [config.LABEL_SIZE], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
      logits = tf.matmul(output_layer, output_weights, transpose_b=True)
      logits = tf.nn.bias_add(logits, output_bias)

      probabilities = tf.nn.sigmoid(logits)  # tf.nn.softmax(logits, axis=-1)
      return probabilities

  def _convert_single_example(self, sample, max_seq_length, tokenizer):
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

    return input_ids, input_mask, segment_ids

  def _get_input_mask_segment_ids(self, sentence, max_seq_length, tokenizer):

    tmp_input_ids, tmp_input_mask, tmp_segment_ids = self._convert_single_example(sentence, max_seq_length, tokenizer)
    return [tmp_input_ids], [tmp_input_mask], [tmp_segment_ids]

  def predict_it(self, text):

    # input_ids = tf.placeholder(tf.int32, [None, config.MAX_SEQ_LENGTH], name="input_ids")  # FLAGS.batch_size
    # input_mask = tf.placeholder(tf.int32, [None, config.MAX_SEQ_LENGTH], name="input_mask")
    # segment_ids = tf.placeholder(tf.int32, [None, config.MAX_SEQ_LENGTH], name="segment_ids")

    # use_one_hot_embeddings = None
    # probabilities = self._create_model(input_ids, input_mask, segment_ids)
    #
    # sess = tf.Session(config=self.gpu_config)
    # sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()
    #
    # if os.path.exists(config.INIT_CKPT):
    #   print("Checkpoint Exists. Restoring Variables from Checkpoint.")
    #   saver.restore(sess, tf.train.latest_checkpoint(config.INIT_CKPT))

    batch_input_ids, batch_input_mask, batch_segment_ids = self._get_input_mask_segment_ids(text, config.MAX_SEQ_LENGTH,
                                                                                            self.tokenizer)

    feed_dict = {self.input_ids: batch_input_ids, self.input_mask: batch_input_mask, self.segment_ids: batch_segment_ids}

    prob = self.sess.run([self.probabilities], feed_dict)
    return prob


def main():
  bertTag = Spear()
  text = "economy hottest in world, socialism should be convicted: Larry Kudlow"
  start = time.time()
  probs = bertTag.predict_it(text)
  print(time.time() - start)
  print(probs)


if __name__ == '__main__':
  main()
  # tf.app.run()
