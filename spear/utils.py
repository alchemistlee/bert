# coding = utf-8

# @time    : 2019/5/22 1:31 PM
# @author  : alchemistlee
# @fileName: utils.py
# @abstract:

import numpy as np
import random

random_number=300


def get_target_label_short_batch(eval_y_big):  # tested.
  eval_y_short_big = []  # will be like:[22,642,1391]
  for ind, eval_y in enumerate(eval_y_big):
    eval_y_short = []
    for index, label in enumerate(eval_y):
      if label > 0:
        eval_y_short.append(index)
    eval_y_short_big.append(eval_y_short)
  return eval_y_short_big


def get_origin_label_from_origin_sample(samples,label_dict):
  label_res = []
  for item in samples:
    tmp_labels = item[0].split('|')
    tmp_label_index = [ label_dict[k] for k in tmp_labels ]
    label_res.append(tmp_label_index)
  return label_res


#get top5 predicted labels
def get_label_using_prob(prob,top_number=5):
    y_predict_labels = [i for i in range(len(prob)) if prob[i] >= 0.50]  # TODO 0.5PW e.g.[2,12,13,10]
    if len(y_predict_labels) < 1:
        y_predict_labels = [np.argmax(prob)]
    return y_predict_labels


def get_label_using_logits_batch(prob,top_number=5): # tested.
    result_labels=[]
    for i in range(len(prob)):
        single_prob=prob[i]
        labels=get_label_using_prob(single_prob)
        result_labels.append(labels)
    return result_labels


def compute_confuse_matrix(target_y, predict_y, label_dict, name='default'):
  """
  compute true postive(TP), false postive(FP), false negative(FN) given target lable and predict label
  :param target_y:
  :param predict_y:
  :param label_dict {label:(TP,FP,FN)}
  :return: macro_f1(a scalar),micro_f1(a scalar)
  """
  # 1.get target label and predict label
  if random.choice([x for x in range(random_number)]) == 1:
    print(name + ".target_y:", target_y, ";predict_y:", predict_y)  # debug purpose

  # 2.count number of TP,FP,FN for each class
  y_labels_unique = []
  y_labels_unique.extend(target_y)
  y_labels_unique.extend(predict_y)
  y_labels_unique = list(set(y_labels_unique))
  for i, label in enumerate(y_labels_unique):  # e.g. label=2
    TP, FP, FN = label_dict[label]
    if label in predict_y and label in target_y:  # predict=1,truth=1 (TP)
      TP = TP + 1
    elif label in predict_y and label not in target_y:  # predict=1,truth=0(FP)
      FP = FP + 1
    elif label not in predict_y and label in target_y:  # predict=0,truth=1(FN)
      FN = FN + 1
    label_dict[label] = (TP, FP, FN)
  return label_dict


def compute_confuse_matrix_batch(y_targetlabel_list, y_logits_array, label_dict, name='default'):
  """
  compute confuse matrix for a batch
  :param y_targetlabel_list: a list; each element is a mulit-hot,e.g. [1,0,0,1,...]
  :param y_logits_array: a 2-d array. [batch_size,num_class]
  :param label_dict:{label:(TP, FP, FN)}
  :param name: a string for debug purpose
  :return:label_dict:{label:(TP, FP, FN)}
  """
  for i, y_targetlabel_list_single in enumerate(y_targetlabel_list):
    label_dict = compute_confuse_matrix(y_targetlabel_list_single, y_logits_array[i], label_dict, name=name)
  return label_dict


def compute_micro_macro(label_dict):
  """
  compute f1 of micro and macro
  :param label_dict:
  :return: f1_micro,f1_macro: scalar, scalar
  """
  f1_micro = compute_f1_micro_use_TFFPFN(label_dict)
  f1_macro = compute_f1_macro_use_TFFPFN(label_dict)
  return f1_micro, f1_macro


def compute_TF_FP_FN_micro(label_dict):
  """
  compute micro FP,FP,FN
  :param label_dict_accusation: a dict. {label:(TP, FP, FN)}
  :return:TP_micro,FP_micro,FN_micro
  """
  TP_micro, FP_micro, FN_micro = 0.0, 0.0, 0.0
  for label, tuplee in label_dict.items():
    TP, FP, FN = tuplee
    TP_micro = TP_micro + TP
    FP_micro = FP_micro + FP
    FN_micro = FN_micro + FN
  return TP_micro, FP_micro, FN_micro


def compute_f1_micro_use_TFFPFN(label_dict):
  """
  compute f1_micro
  :param label_dict: {label:(TP,FP,FN)}
  :return: f1_micro: a scalar
  """
  TF_micro_accusation, FP_micro_accusation, FN_micro_accusation = compute_TF_FP_FN_micro(label_dict)
  f1_micro_accusation = compute_f1(TF_micro_accusation, FP_micro_accusation, FN_micro_accusation, 'micro')
  return f1_micro_accusation


def compute_f1_macro_use_TFFPFN(label_dict):
  """
  compute f1_macro
  :param label_dict: {label:(TP,FP,FN)}
  :return: f1_macro
  """
  f1_dict = {}
  num_classes = len(label_dict)
  for label, tuplee in label_dict.items():
    TP, FP, FN = tuplee
    f1_score_onelabel = compute_f1(TP, FP, FN, 'macro')
    f1_dict[label] = f1_score_onelabel
  f1_score_sum = 0.0
  for label, f1_score in f1_dict.items():
    f1_score_sum = f1_score_sum + f1_score
  f1_score = f1_score_sum / float(num_classes)
  return f1_score


small_value = 0.00001


def compute_f1(TP, FP, FN, compute_type):
  """
  compute f1
  :param TP_micro: number.e.g. 200
  :param FP_micro: number.e.g. 200
  :param FN_micro: number.e.g. 200
  :return: f1_score: a scalar
  """
  precison = TP / (TP + FP + small_value)
  recall = TP / (TP + FN + small_value)
  f1_score = (2 * precison * recall) / (precison + recall + small_value)

  if random.choice([x for x in range(500)]) == 1: print(compute_type, "precison:", str(precison), ";recall:",
                                                        str(recall), ";f1_score:", f1_score)
  return f1_score


def init_label_dict(num_classes):
  """
  init label dict. this dict will be used to save TP,FP,FN
  :param num_classes:
  :return: label_dict: a dict. {label_index:(0,0,0)}
  """
  label_dict = {}
  for i in range(num_classes):
    label_dict[i] = (0, 0, 0)
  return label_dict
