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


class Block(StorageObject):
    """
    @ClassField block numpy.ndarray
    """

    # # Unsupported with Hecuba
    # def __init__(self, block):
    #     super(Block, self).__init__()
    #     self.block = block

    # # Unsupported with Hecuba
    # def __mul__(self, other):
    #     return self.block * other.block
