from collections import defaultdict

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
    from pycompss.api.task import task
    from pycompss.api.parameter import FILE_IN
except ImportError:
    # Required since the pycompss module is not ready during the registry
    from dataclay.contrib.dummy_pycompss import task, FILE_IN

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
    def __init__(self, text=''):
        super(Words, self).__init__()
        self.text = text

    @dclayMethod(return_='str')
    def get_text(self):
        return self.text

    @dclayMethod(text='str')
    def set_text(self, text):
        self.text = text

    @task(returns=1, file=FILE_IN, priority=True)
    @dclayMethod(file_path='str')
    def populate_block(self, file_path):
        """
        Reads a file and stores its content within the object.
        :param file_path: Absolute path of the file to process.
        :return: None
        """
        fp = open(file_path)
        data = fp.read()
        fp.close()
        self.set_text(data)

    @task(returns=defaultdict, priority=True)
    @dclayMethod(return_='anything')
    def wordcount(self):
        """
        Wordcount over a Words object.
        :param block: Block with text to perform word counting.
        :return: dictionary with the words and the number of appearances.
        """
        data = self.get_text().split()
        result = defaultdict(int)
        for word in data:
            result[word] += 1
        return result
