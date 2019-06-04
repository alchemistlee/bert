# coding = utf-8

# @time    : 2019/6/4 10:41 AM
# @author  : alchemistlee
# @fileName: connect.py
# @abstract:


from multiprocessing.managers import BaseManager
import time
import uuid
import random
import config
import threading


class ConnectManager(BaseManager):

    def __init__(self, **kwargs):
        super(ConnectManager, self).__init__(**kwargs)
        self.host = kwargs['address']

    def server(self):
        _server = self.get_server()
        _server.serve_forever()

    def client(self):
        self.connect()

    @property
    def state(self):
        state = getattr(self, '_state', None)
        if not state:
            return
        return state.value

    def get(self, function_name):
        return getattr(self, function_name, None)


class MyConnManager(object):

  def __init__(self, hosts=None, authkey=None):
    self._hosts = hosts
    self._authkey = str(authkey).encode('utf-8')
    self._managers = dict()
    self._initialize()

  def _initialize(self):
    for host in self._hosts:
      self._managers[uuid.uuid4()] = ConnectManager(address=host, authkey=self._authkey)
    for idx in self._managers.keys():
      self.connect(idx)
    threading.Thread(target=self._listen).start()

  def _listen(self):
    while True:
      for idx in self._managers.keys():
        self.connect(idx)
      time.sleep(config.REMOTE_LISTEN_TIME)

  def connect(self, idx):
    if idx not in self._managers:
      return
    manager = self._managers[idx]
    # logger.info('uid: %s, host: %s connect...' % (idx, manager.host))
    try:
      manager.connect()
    except Exception as e:
      print('uid: %s, host: %s. connect error: %s' % (idx, manager.host, e))
    else:
      print('uid: %s, host: %s. connect success!' % (idx, manager.host))

  @staticmethod
  def register(name):
    ConnectManager.register(name)

  def _get_random_manager(self):
    items = [(k, v) for (k, v) in self._managers.items() if v.state == 1]
    if not items:
      print('not found connect manager ')
      raise Exception
    return random.choice(items)

  def _get_manager(self, idx):
    return self._managers.get(idx)

  def predict(self, string, idx=None):
    if idx is None:
      idx, manager = self._get_random_manager()
    else:
      manager = self._get_manager(idx)
    try:
      predict_func = manager.get(config.REMOTE_PREDICT_FUNC)
      return predict_func(string)
    except (ConnectionRefusedError, EOFError):
      self.connect(idx)
      return self.predict(string)

  def result(self, string, idx=None):
    try:
      if idx is None:
        idx, manager = self._get_random_manager()
      else:
        manager = self._get_manager(idx)
      uid = uuid.uuid3(uuid.NAMESPACE_DNS,string)
      push_func = manager.get(config.REMOTE_PUSH_FUNC)
      push_func(uid,string)
      result_func = manager.get(config.REMOTE_RESULT_FUNC)
      res = result_func(uid, timeout=config.REMOTE_LISTEN_TIME)
      # print(str(res))
      return res
    except (ConnectionRefusedError, EOFError):
      self.connect(idx)
      return self.result(string)
    except Exception as e:
      print('DispatchManager result error: %s' % e)
    return {}