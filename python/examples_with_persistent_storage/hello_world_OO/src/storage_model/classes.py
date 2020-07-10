try:
    # dataClay and Redis
    from storage.api import StorageObject
except:
    # Hecuba
    from hecuba.storageobj import StorageObj as StorageObject

try:
    from dataclay import dclayMethod
except ImportError:
    def dclayMethod(*args, **kwargs):
        return lambda f: f


class hello(StorageObject):
    """
    @ClassField message str
    """
    @dclayMethod(message='anything')
    def __init__(self, message):
        super(hello, self).__init__()
        self.message = message

    @dclayMethod(return_='anything')
    def get(self):
        return self.message
