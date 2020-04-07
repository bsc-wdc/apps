from pycompss.api.task import task
from pycompss.api.parameter import INOUT

import numpy as np


class Block(object):

    def __init__(self, block=None):
        self.block = block

    @task(target_direction=INOUT)
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

    def __mul__(self, other):
        return Block(self.block * other.block)

    def __iadd__(self, other):
        self.block += other.block
        return self
