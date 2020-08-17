#!/usr/bin/python
#
#  Copyright 2002-2020 Barcelona Supercomputing Center (www.bsc.es)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

# -*- coding: utf-8 -*-

# For better print formatting
from __future__ import print_function

# Regular imports
import sys
import time

# PyCOMPSs imports
from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.api.api import compss_wait_on
from pycompss.api.api import compss_barrier

# Project imports
from block import Block

# Constants
DEBUG = 1


#
# Define auxiliar functions
#

def init_matrix(width_num_blocks, length_num_blocks, block_size):
    res = []
    for i in range(0, width_num_blocks):
        a = []
        for j in range(0, length_num_blocks):
            a.append(init_block(block_size))
        res.append(a)
    return res


def init_matrix_void(width_num_blocks, length_num_blocks):
    res = []
    for i in range(0, width_num_blocks):
        a = []
        for j in range(0, length_num_blocks):
            a.append(None)
        res.append(a)
    return res


#
# TASKS DEFINITION
#

@task(returns=Block)
def init_block(blockSize):
    return Block(block_size=blockSize, rand=1)


@task(returns=Block, b00=IN, b01=IN, b02=IN, b10=IN, b11=IN, b12=IN, b20=IN, b21=IN, b22=IN)
def update_block(b00, b01, b02, b10, b11, b12, b20, b21, b22, aFactor, blockSize):
    # sub_state_a
    sub_state_a = []
    for i in range(0, 3):
        sub_state_a.append([])

    for i in range(0, 3):
        for j in range(0, 3):
            sub_state_a[i].append(Block(blockSize))

    # sub_state_b
    sub_state_b = [[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]]

    # iterations
    for t in range(aFactor, -1, -1):
        sub_state_c = sub_state_a
        sub_state_a = sub_state_b
        sub_state_b = sub_state_c

        for i in range(blockSize - t, 2 * blockSize + t):
            for j in range(blockSize - t, 2 * blockSize + t):

                count = 0

                # count
                for off_i in range(-1, 2):
                    for off_j in range(-1, 2):
                        if off_i != 0 or off_j != 0:
                            p = sub_state_a[(i + off_i) / blockSize][(j + off_j) / blockSize]
                            if p.get((i + off_i) % blockSize, (j + off_j) % blockSize) == 1:
                                count = count + 1

                # Rules
                p = sub_state_a[i / blockSize][j / blockSize]
                q = sub_state_b[i / blockSize][j / blockSize]
                mod_i = i % blockSize
                mod_j = j % blockSize

                if p.get(mod_i, mod_j) == 1:
                    if count == 2 or count == 3:
                        q.set(mod_i, mod_j, 1)
                    else:
                        q.set(mod_i, mod_j, 0)
                else:
                    if count == 3:
                        q.set(mod_i, mod_j, 1)
                    else:
                        q.set(mod_i, mod_j, 0)

    return Block(ref=sub_state_b[1][1])


#
# MAIN
#

def main_program():
    def usage():
        print("[ERROR] Bad number of parameters.")
        print("    Usage: conway.py <W, L, num_iterations, block_size, a_factor>")

    # Initialize constants
    if len(sys.argv) != 6:
        usage()
        exit(-1)

    width_elements = int(sys.argv[1])
    length_elements = int(sys.argv[2])
    num_iterations = int(sys.argv[3])
    block_size = int(sys.argv[4])
    a_factor = int(sys.argv[5])

    width_num_blocks = width_elements / block_size
    length_num_blocks = length_elements / block_size

    if DEBUG == 1:
        print("Application parameters:")
        print("- Elements Width: {}".format(width_elements))
        print("- Elements Length: {}".format(length_elements))
        print("- Num. Iterations: {}".format(num_iterations))
        print("- Block size: {}".format(block_size))
        print("- A factor: {}".format(a_factor))

    # Timing
    start_time = time.time()

    # Initialize state
    state_a = init_matrix(width_num_blocks, length_num_blocks, block_size)
    # Initialize swap state (only structure, blocks will be copied)
    state_b = init_matrix_void(width_num_blocks, length_num_blocks)

    # Iterations
    for iter in range(0, num_iterations / (a_factor + 1)):
        if DEBUG == 1:
            print("Running iteration {}".format(iter))

        # Swap states
        if iter != 0:
            if DEBUG == 1:
                print("- Swapping starting states...")
            for i in range(0, width_num_blocks):
                for j in range(0, length_num_blocks):
                    state_a[i][j] = state_b[i][j]

        # Update blocks
        if DEBUG == 1:
            print("- Updating block states...")

        for i in range(0, width_num_blocks):
            for j in range(0, length_num_blocks):
                state_b[i][j] = update_block(
                    state_a[(i - 1 + width_num_blocks) % width_num_blocks][
                        (j - 1 + length_num_blocks) % length_num_blocks],
                    state_a[(i - 1 + width_num_blocks) % width_num_blocks][
                        (j + 0 + length_num_blocks) % length_num_blocks],
                    state_a[(i - 1 + width_num_blocks) % width_num_blocks][
                        (j + 1 + length_num_blocks) % length_num_blocks],
                    state_a[(i + 0 + width_num_blocks) % width_num_blocks][
                        (j - 1 + length_num_blocks) % length_num_blocks],
                    state_a[(i + 0 + width_num_blocks) % width_num_blocks][
                        (j + 0 + length_num_blocks) % length_num_blocks],
                    state_a[(i + 0 + width_num_blocks) % width_num_blocks][
                        (j + 1 + length_num_blocks) % length_num_blocks],
                    state_a[(i + 1 + width_num_blocks) % width_num_blocks][
                        (j - 1 + length_num_blocks) % length_num_blocks],
                    state_a[(i + 1 + width_num_blocks) % width_num_blocks][
                        (j + 0 + length_num_blocks) % length_num_blocks],
                    state_a[(i + 1 + width_num_blocks) % width_num_blocks][
                        (j + 1 + length_num_blocks) % length_num_blocks],
                    a_factor, block_size)

    # Results
    for i in range(0, width_num_blocks):
        for j in range(0, length_num_blocks):
            state_b[i][j] = compss_wait_on(state_b[i][j])

    if DEBUG == 1:
        print("Results:")

        for i in range(0, width_num_blocks):
            for j in range(0, length_num_blocks):
                print("Block [{},{}] = {}".format(i, j, state_b[i][j]))

    # Timing
    compss_barrier()
    end_time = time.time()
    print("Total execution time: {} s".format(end_time - start_time))


if __name__ == "__main__":
    main_program()
