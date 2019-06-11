from storage.storage_object import StorageObject
import storage.api

class Block(StorageObject):
    def __init__(self, block):
        super(Block, self).__init__()
        self.block = block

    def get_block(self):
        return self.block

    def set_block(self, new_block):
        self.block = new_block
