import storage.api
from storage.storage_object import StorageObject

class PSCO(StorageObject):
    def __init__(self, mat = "Content"):
        super(PSCO, self).__init__()
        self.mat = mat

    def get_mat(self):
        return self.mat

    def set_mat(self, mat):
        self.mat = mat

