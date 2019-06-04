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


