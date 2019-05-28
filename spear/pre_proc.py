# coding = utf-8

# @time    : 2019/5/24 2:07 PM
# @author  : alchemistlee
# @fileName: pre_proc.py
# @abstract:

import os, sys, getopt
import random
import math


def get_resp_file(base_path,base_name):
  train_name = base_path + base_name+'-train'
  val_name = base_path + base_name+'-val'
  test_name = base_path + base_name+'-test'
  return train_name, val_name, test_name


def get_resp_index(data_size , val_prop, test_prop):
  val_beg_end = list()
  test_beg_end = list()
  train_beg_end = list()

  val_size = math.floor(data_size * val_prop)
  test_size = math.floor(data_size * test_prop)

  test_beg_end[0] = 0
  test_beg_end[1] = test_size

  val_beg_end[0] = test_beg_end[1]
  val_beg_end[1] = val_beg_end[0]+val_size

  train_beg_end[0] = val_beg_end[1]
  train_beg_end[1] = data_size

  return test_beg_end, val_beg_end, train_beg_end


def write_into_file(data,fh):
  for item in data :
    tmp_line = '%s\t%s\n' % (item[0],item[1])
    fh.write(tmp_line)


def prod_train_val_test_label_in_file(base_path,data_name):
  '''
  在文件里面生成train / val / test 和 label
  :param base_path:
  :param data_name:
  :return:
  '''
  data_file = base_path+data_name
  train_file, val_file, test_file = get_resp_file(base_path,data_name)
  label_set = dict()

  with open(train_file,'w') as train_fh, open(val_file,'w') as val_fh, open(test_file,'w') as test_fh :
    with open(data_file,'r') as data_fh:
      for cont in data_fh.read().split("\n!@#$%^&***********\n"):
        line = cont.strip().split("\t", 1)
        if len(line) != 2:
          continue
        tmp_label = eval(line[0])
        tmp_info = line[1]
        tmp_label_txt = '|'.join(tmp_label)

        tmp_line_txt = '%s\t%s\n' % (tmp_label_txt,tmp_info)

        seed = random.random()

        if 0.0 <= seed < 0.1:
          val_fh.write(tmp_line_txt)
        elif 0.1<= seed < 0.2 :
          test_fh.write(tmp_line_txt)
        else:
          train_fh.write(tmp_line_txt)

        for item_label in tmp_label:
          if not item_label in label_set.keys():
            label_set[item_label] = 1
          else:
            tmp_cnt = label_set[item_label]
            tmp_cnt += 1
            label_set[item_label] = tmp_cnt

  print('finish write train / val / test ')

  lable_file = base_path + 'label.txt'
  with open(lable_file,'w') as lable_fh:
    for sub_label in label_set.keys():
      lable_fh.write(sub_label+'\n')

  print('finish write lable ')


def prod_train_val_test_label_in_mem(base_path, data_name):
  '''
  在内存里面直接进行shuffle，分组test 、train、val
  :param base_path:
  :param data_name:
  :return:
  '''
  data_file = base_path + data_name
  train_file, val_file, test_file = get_resp_file(base_path, data_name)
  label_set = dict()

  all_data = []
  with open(train_file,'w') as train_fh, open(val_file,'w') as val_fh, open(test_file,'w') as test_fh :
    with open(data_file,'r') as data_fh:
      for cont in data_fh.read().split("\n!@#$%^&***********\n"):
        line = cont.strip().split("\t", 1)
        if len(line) != 2:
          continue
        tmp_label = eval(line[0])
        tmp_info = line[1]
        tmp_label_txt = '|'.join(tmp_label)
        tmp_data = [tmp_label_txt,tmp_info]
        all_data.append(tmp_data)

        for item_label in tmp_label:
          if not item_label in label_set.keys():
            label_set[item_label] = 1
          else:
            tmp_cnt = label_set[item_label]
            tmp_cnt += 1
            label_set[item_label] = tmp_cnt

      random.shuffle(all_data)

      test_idxs, val_idxs, train_idxs = get_resp_index(len(all_data), 0.01, 0.01)

      test_data = all_data[test_idxs[0]:test_idxs[1]]
      val_data = all_data[val_idxs[0]:val_idxs[1]]
      train_data = all_data[train_idxs[0]:train_idxs[1]]

      write_into_file(test_data, test_fh)
      write_into_file(train_data, train_fh)
      write_into_file(val_data, val_fh)

  print('finish write train / val / test ')

  lable_file = base_path + 'label.txt'
  with open(lable_file, 'w') as lable_fh:
    for sub_label in label_set.keys():
      lable_fh.write(sub_label + '\n')

  print('finish write lable ')


if __name__ == '__main___':
  origin_data_file = None
  try:
    opts, args = getopt.getopt(sys.argv[1:],'hi:t:')
  except getopt.GetoptError:
    print('xxx.py -i <input-data-file> -t <file or mem>')
    sys.exit(2)

  is_file_proc = True

  for opt, arg in opts:
    if opt == '-h':
      print('xxx.py -i <input-data-file> -t <file or mem>')
      sys.exit(2)
    elif opt == '-i':
      origin_data_file = arg
    elif opt == '-t':
      if arg == 'mem':
        is_file_proc = False

  if not origin_data_file is None:
    dirname, filename = os.path.split(origin_data_file)
    print('input path and file , [ %s ] [ %s ]' % (dirname, filename))

    if is_file_proc :
      print('produce in file !')
      prod_train_val_test_label_in_file(dirname,filename)
    else:
      print('produce in memory !')
      prod_train_val_test_label_in_mem(dirname,filename)
  else:
    print('need input-file !!~')

