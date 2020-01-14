try:
    # dataClay and Hecuba
    from storage.api import StorageObject
except:
    # Redis
    from storage.storage_object import StorageObject

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
