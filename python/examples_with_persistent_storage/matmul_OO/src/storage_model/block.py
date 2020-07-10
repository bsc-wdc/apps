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

    @dclayMethod(a='storage_model.block.Block', b='storage_model.block.Block')
    def fused_multiply_add(self, a, b):
        """Accumulate a product.

        This FMA operation multiplies the two operands (parameters a and b) and
        accumulates its result onto self.

        Note that the multiplication is the matrix multiplication (aka np.dot)
        """
        self.block += np.dot(a.block, b.block)
