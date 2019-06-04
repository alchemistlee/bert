# coding = utf-8

# @time    : 2019/6/4 10:41 AM
# @author  : alchemistlee
# @fileName: connect.py
# @abstract:


# from multiprocessing.managers import DictProxy
from multiprocessing.managers import BaseManager


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

