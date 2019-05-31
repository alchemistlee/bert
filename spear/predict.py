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
flags.DEFINE_string("ckpt_dir", None,"Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_string("init_ckpt", None,"Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_bool("do_lower_case", True,"Whether to lower case the input text. Should be True for uncased models and False for cased models.")
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_bool("do_train", True, "Whether to run training.")


#hyper-parameter
flags.DEFINE_integer("max_seq_length", 40,
  "max total input sequence length after WordPiece tokenization. Seq longer than this will be truncated, and seq shorter than this will be padded.")
flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")
flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")
flags.DEFINE_float("learning_rate", 2e-5, "The initial learning rate for Adam.")
flags.DEFINE_float("num_train_epochs", 2.0,"Total number of training epochs to perform.")
flags.DEFINE_float("warmup_proportion", 0.1,"Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
flags.DEFINE_integer("save_checkpoints_steps", 1000,"How often to save the model checkpoint.")
flags.DEFINE_integer("iterations_per_loop", 1000,"How many steps to make in each estimator call.")





def load_vocab(vocab_file):
  return tokenization.load_vocab(vocab_file)


def load_label(label_file):
  label_dict = collections.OrderedDict()
  index = 0
  with tf.gfile.GFile(label_file, "r") as reader:
    while True:
      token = tokenization.convert_to_unicode(reader.readline())
      if not token:
        break
      token = token.strip()
      label_dict[token] = index
      index += 1

  print('label-dict = '+str(label_dict))
  return label_dict


def load_samples(sample_file):
  res = []
  index = 0
  with tf.gfile.GFile(sample_file, "r") as reader:
    while True:
      line = reader.readline()
      if not line:
        break
      line = line.strip()
      tmp_data = line.split('\t')

      if len(tmp_data) != 2:
        continue
      res.append(tmp_data)
      index += 1
  print('load file [ %s ] , size = [ %s ]' % (sample_file,str(index+1)))
  return res



def trans_multilabel_as_multihot(label_dict,origin_label):
  """
  convert to multi-hot style
  :param label_list: e.g.[0,1,4], here 4 means in the 4th position it is true value(as indicate by'1')
  :param label_size: e.g.199
  :return:e.g.[1,1,0,1,0,0,........]
  """
  label_size = len(label_dict)
  result = np.zeros(label_size)
  label_list = list()
  for item_label in origin_label:
    label_list.append(label_dict[item_label])

  # set those location as 1, all else place as 0.
  result[label_list] = 1
  return result


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

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    # if is_training:
    #   # I.e., 0.1 dropout
    #   output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    def apply_dropout_last_layer(output_layer):
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
      return output_layer

    def not_apply_dropout(output_layer):
      return output_layer

    #output_layer=tf.cond(is_training, lambda: apply_dropout_last_layer(output_layer), lambda:not_apply_dropout(output_layer))
    # if is_training:
    #   output_layer = apply_dropout_last_layer(output_layer)
    # else :
    #   output_layer = not_apply_dropout(output_layer)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    # probabilities = tf.nn.softmax(logits, axis=-1)
    # log_probs = tf.nn.log_softmax(logits, axis=-1)
    #
    # one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    #
    # per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    # loss = tf.reduce_mean(per_example_loss)

    probabilities = tf.nn.sigmoid(logits)  # tf.nn.softmax(logits, axis=-1)
    #per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)  # shape=(?, 1999)
    #loss_batch = tf.reduce_sum(per_example_loss, axis=1)  # (?,)
    #loss = tf.reduce_mean(loss_batch)
    #loss = tf.reduce_mean(per_example_loss)
    return probabilities, model


def convert_single_example(sample, label_dict, max_seq_length,tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  # if isinstance(example, PaddingInputExample):
  #   return InputFeatures(
  #       input_ids=[0] * max_seq_length,
  #       input_mask=[0] * max_seq_length,
  #       segment_ids=[0] * max_seq_length,
  #       label_id=0,
  #       is_real_example=False)
  text_a = sample

  # label_map = {}
  # for (i, label) in enumerate(label_list):
  #   label_map[label] = i

  tokens_a = tokenizer.tokenize(text_a)
  # tokens_b = None
  # if text_b:
  #   tokens_b = tokenizer.tokenize(text_b)
  #
  # if tokens_b:
  #   # Modifies `tokens_a` and `tokens_b` in place so that the total
  #   # length is less than the specified length.
  #   # Account for [CLS], [SEP], [SEP] with "- 3"
  #   _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  # else:
  # Account for [CLS] and [SEP] with "- 2"
  if len(tokens_a) > max_seq_length - 2:
    tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  # if tokens_b:
  #   for token in tokens_b:
  #     tokens.append(token)
  #     segment_ids.append(1)
  #   tokens.append("[SEP]")
  #   segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
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


def get_input_mask_segment_ids(sentence, label_dict, max_seq_length, tokenizer):
  batch_input_ids = []
  batch_input_mask = []
  batch_segment_ids = []

  tmp_input_ids, tmp_input_mask, tmp_segment_ids = convert_single_example(sentence, label_dict, max_seq_length, tokenizer)
  batch_input_ids.append(tmp_input_ids)
  batch_input_mask.append(tmp_input_mask)
  batch_segment_ids.append(tmp_segment_ids)

  return batch_input_ids, batch_input_mask, batch_segment_ids



def predict_it(valid_data, label_dict, tokenizer):
  """
  evalution on model using validation data
  """
  label_size = len(label_dict)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name="input_ids")  # FLAGS.batch_size
  input_mask = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name="input_mask")
  segment_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name="segment_ids")
  label_ids = tf.placeholder(tf.float32, [None, label_size], name="label_ids")
  # is_training = FLAGS.do_train

  use_one_hot_embeddings = None
  probabilities, model = create_model(bert_config, input_ids, input_mask, segment_ids, label_ids, label_size,use_one_hot_embeddings)

  global_step = tf.Variable(0, trainable=False, name="Global_Step")

  gpu_config = tf.ConfigProto()
  gpu_config.gpu_options.allow_growth = True
  sess = tf.Session(config=gpu_config)
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()

  if os.path.exists(FLAGS.init_ckpt):
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
    print("Checkpoint Exists. Restoring Variables from Checkpoint.")
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.init_ckpt))

  tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)


  num_examples = len(valid_data)

  calc_label_dict = utils.init_label_dict(len(label_dict))
  batch_size=1 # TODO
  batch_input_ids_, batch_input_mask_, batch_segment_ids = get_input_mask_segment_ids(valid_data[0], label_dict, FLAGS.max_seq_length,tokenizer)

  feed_dict = {input_ids: batch_input_ids_, input_mask: batch_input_mask_, segment_ids: batch_segment_ids}
  prob = sess.run([probabilities], feed_dict)
  print(prob)
  #predict_labels = utils.get_label_using_logits_batch(prob)
  #print(predict_lables)



def main(_):
  sentence = FLAGS.sentence
  label_file = FLAGS.data_dir + 'label.txt'

  label_dict = load_label(label_file)

  tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  #batch_input_ids, batch_input_mask, batch_segment_ids, batch_label_ids = get_input_mask_segment_ids(sentence,label_dict,FLAGS.max_seq_length,tokenizer)
  predict_it([sentence], label_dict, tokenizer)


if __name__ == '__main__':
  tf.app.run()





