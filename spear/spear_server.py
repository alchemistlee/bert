# coding = utf-8

# @time    : 2019/6/4 10:02 AM
# @author  : alchemistlee
# @fileName: spear_server.py
# @abstract:


import threading
from queue import Queue
from connect import ConnectManager
import config
import hashlib
from multiprocessing import freeze_support
import fire
import Spear
import time
import asyncio
import uuid


class SpearServer(object):

  def __init__(self,host, port, authkey):
    self._host = host
    self._port = port
    self._address = '%s:%s' % (self._host, self._port)
    self._authkey = str(authkey).encode('utf-8')
    self._task_queue = Queue()
    self._task_result = dict()
    self._manager = ConnectManager(address=(self._host, self._port), authkey=self._authkey)
    self._initialize()
    self._spear = Spear.MultiLabelSpear()


  def _initialize(self):
    ConnectManager.register(config.REMOTE_PUSH_FUNC, callable=self.push)
    ConnectManager.register(config.REMOTE_RESULT_FUNC, callable=self.result)
    # ConnectManager.register(config.REMOTE_PREDICT_FUNC, callable=self.predict)
    threading.Thread(target=self._manager.server).start()
    print('host: %s start success!' % self._address)

  def push(self, uid, content):
    # md5=self.md5(content)
    print('uid: %s, string: %s' % (uid, content))
    self._task_queue.put((uid, content))

  def result(self, uid, timeout=None):
    result_data = dict()
    begin_time = time.time()
    while True:
      if uid not in self._task_result:
        if timeout and (time.time() - begin_time) >= timeout:
          print('host: %s, uid: %s, timeout: %s' % (self._address, uid, time.time() - begin_time))
          result_data['status'] = -999
          return result_data
        continue
      result_data['res'] = self._task_result.pop(uid)
      result_data['status']=0
      print('host: %s, uid: %s, result: %s' % (self._address, uid, result_data))
      return result_data

  def entry(self):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(self.my_loop())

  async def my_loop(self):
    while True:
      if self._task_queue.empty():
        continue
      try:
        uid, content = self._task_queue.get()
        prob = self._spear.predict_it(content)
        self._task_result[uid]=prob
      except Exception as e:
        print('ERR , host: %s, run error: %s' % (self._address, e))


def main(host, port, authkey=None):
    SpearServer(host=host, port=port, authkey=authkey).entry()


if __name__ == '__main__':
  freeze_support()
  fire.Fire(main)