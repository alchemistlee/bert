# coding = utf-8

# @time    : 2019/6/4 6:24 PM
# @author  : alchemistlee
# @fileName: spear_interface.py
# @abstract:


from flask import Flask
from flask import request
from flask import render_template

from connect import MyConnManager

import config

import sys
import json
import time


app = Flask(__name__)

MyConnManager.register(config.REMOTE_PUSH_FUNC)
MyConnManager.register(config.REMOTE_RESULT_FUNC)
# MyConnManager.register(config.REMOTE_PREDICT_FUNC)
app.config['myConn'] = MyConnManager(hosts=config.REMOTE_ADDRESS, authkey=config.REMOTE_AUTHKEY)


def decode_predict(string):
  try:
    myConn = app.config['myConn']
    result_dict = myConn.result(string)
    print('result-dict = %s ' % str(result_dict))
    # result_dict = dispatch.predict(string)
    status = result_dict.get('status')
    res=result_dict.get('res')
    if status == -999:
      raise Exception('识别超时')
  except Exception as e:
    raise e
  else:
    return status, res


app.config['decode_predict_func'] = decode_predict


@app.route("/bert/multi_label",methods=['GET','POST'])
def multi_label_bert():

  result_dict = dict(error_code=1)
  try:

    if request.method == 'GET':
      input_str = request.args.get('in')
    else :
      # post
      input_str = request.form['in']

    decode_predict_func = app.config['decode_predict_func']
    start_time = time.time()
    status, res = decode_predict_func(input_str)
    elapsed_time = time.time() - start_time
    print(' cost time =  %s ' % str(elapsed_time))
    #res_str = str(res)
    res_str = res
  except Exception as e:
    result_dict['message'] = str(e)
    print('views action text error: %s' % e)
  else:
    result_dict = {
      'error_code': 0,
      'message': 'success',
      'data': {
        'res': res_str,
        'status': status,
      }
    }

  return json.dumps(result_dict)


if __name__ == "__main__":
  app.debug = True
  # logging.basicConfig(stream=sys.stdout)
  # handler = logging.FileHandler("/home/yifan.li/data/logs/my-tf-flask.log", encoding="UTF-8")
  # handler.setLevel(logging.DEBUG)
  # logging_format = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s')
  # handler.setFormatter(logging_format)
  # app.logger.addHandler(handler)
  app.run(host='0.0.0.0')


