try:
    # dataClay
    from storage.api import StorageObject
except:
    try:
        # Hecuba
        from hecuba.storageobj import StorageObj as StorageObject
    except:
        # Redis
        from storage.storage_object import StorageObject

try:
    from dataclay import dclayMethod
except ImportError:
    def dclayMethod(*args, **kwargs):
        return lambda f: f


class Words(StorageObject):
    """
    @ClassField text str
    """

    @dclayMethod(text='str')
    def __init__(self, text):
        super(Words, self).__init__()
        self.text = text

    @dclayMethod(return_='str')
    def get_text(self):
        return self.text

    @dclayMethod(text='str')
    def set_text(self, text):
        self.text = text
