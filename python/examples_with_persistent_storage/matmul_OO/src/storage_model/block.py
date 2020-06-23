try:
    # dataClay and Redis
    from storage.api import StorageObject
except:
    # Hecuba
    from hecuba.storageobj import StorageObj as StorageObject

try:
    from pycompss.api.task import task
    from pycompss.api.parameter import INOUT
except ImportError:
    # Required since the pycompss module is not ready during the registry
    from dataclay.contrib.dummy_pycompss import task, INOUT

try:
    from dataclay import dclayMethod
except ImportError:
    def dclayMethod(*args, **kwargs):
        return lambda f: f

import numpy as np


class Block(StorageObject):
    """
    @ClassField block numpy.ndarray

    @dclayImport numpy as np
    """

    @dclayMethod(block='anything')
    def __init__(self, block=None):
        super(Block, self).__init__()
        self.block = block

    @dclayMethod(size='int', num_blocks='int', seed='int', set_to_zero='bool')
    def generate_block(self, size, num_blocks, seed=0, set_to_zero=False):
        """
        Generate a square block of given size.
        :param size: <Integer> Block size
        :param num_blocks: <Integer> Number of blocks
        :param seed: <Integer> Random seed
        :param set_to_zero: <Boolean> Set block to zeros
        :return: None
        """
        np.random.seed(seed)
        if not set_to_zero:
            b = np.random.random((size, size))
            # Normalize matrix to ensure more numerical precision
            b /= np.sum(b) * float(num_blocks)
        else:
            b = np.zeros((size, size))
        self.block = b

    @dclayMethod(other='storage_model.block.Block', return_='storage_model.block.Block')
    def __mul__(self, other):
        return Block(np.dot(self.block, other.block))

    @dclayMethod(other='storage_model.block.Block', return_='storage_model.block.Block')
    def __iadd__(self, other):
        self.block += other.block
        return self
