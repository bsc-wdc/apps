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


class Block(StorageObject):
    """
    @ClassField block numpy.matrix
    """

    @dclayMethod(block='anything')
    def __init__(self, block):
        super(Block, self).__init__()
        self.block = block

    @dclayMethod(other='classes.block.Block', return_='classes.block.Block')
    def __mul__(self, other):
        return Block(self.block * other.block)

    @dclayMethod(other='classes.block.Block', return_='classes.block.Block')
    def __iadd__(self, other):
        self.block += other.block
        return self
