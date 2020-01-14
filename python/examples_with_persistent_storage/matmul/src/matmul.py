"""
A matrix multiplication implementation with (optionally) PSCOs
author: Sergio Rodriguez Guasch < sergio rodriguez at bsc dot es >
"""
from pycompss.api.task import task
from pycompss.api.parameter import *

from classes.block import Block

@task(returns=1)
def generate_block(size, num_blocks, seed=0, psco=False, set_to_zero=False):
    """
    Generate a square block of given size.
    :param size: <Integer> Block size
    :param num_blocks: <Integer> Number of blocks
    :param seed: <Integer> Random seed
    :param psco: <Boolean> If psco
    :param set_to_zero:<Boolean> Set block to zeros
    :return: Block (persisted if psco)
    """
    import numpy as np
    np.random.seed(seed)
    if not set_to_zero:
        b = np.matrix(np.random.random((size, size)))
    else:
        b = np.zeros((size, size))
    # Normalize matrix to ensure more numerical precision
    if not set_to_zero:
        b /= np.sum(b) * float(num_blocks)
    ret = Block(b)
    if psco:
        ret.make_persistent()
    return ret


@task(C=INOUT)
def multiply(A, B, C):
    """
    Multiplies two Blocks and accumulates the result in an INOUT Block
    :param A: Block A
    :param B: Block B
    :param C: Result Block
    :return: None
    """
    C += A * B


def dot(A, B, C, set_barrier=False):
    """
    A COMPSs-PSCO blocked matmul algorithm.
    A and B (blocks) are persistent PSCOs, while C (blocks) are non
    persistent blocks == objects
    :param A: Block A
    :param B: Block B
    :param C: Result Block
    :param set_barrier: Set barrier at the end
    :return: None
    """
    n, m = len(A), len(B[0])
    # as many rows as A, as many columns as B
    for i in range(n):
        for j in range(m):
            for k in range(n):
                multiply(A[i][k], B[k][j], C[i][j])
    if set_barrier:
        from pycompss.api.api import compss_barrier
        compss_barrier()


@task()
def persist_result(b):
    """
    Persist results of block b.
    :param b: Block to persist
    :return: None
    """
    from classes.block import Block
    bl = Block(b)
    bl.make_persistent()


def main(num_blocks, elems_per_block, check_result, seed):
    """
    Matmul main.
    :param num_blocks: <Integer> Number of blocks
    :param elems_per_block: <Integer> Number of elements per block
    :param check_result: <Boolean> Check results against sequential version
                         of matmul
    :param seed: <Integer> Random seed
    :return: None
    """
    from pycompss.api.api import compss_barrier
    import time

    start_time = time.time()

    # Generate the dataset in a distributed manner
    # i.e: avoid having the master a whole matrix
    A, B, C = [], [], []
    for i in range(num_blocks):
        for l in [A, B, C]:
            l.append([])
        # Keep track of blockId to initialize with different random seeds
        bid = 0
        for j in range(num_blocks):
            for l in [A, B]:
                l[-1].append(generate_block(elems_per_block,
                                            num_blocks,
                                            seed=seed + bid,
                                            psco=True))
                bid += 1
            C[-1].append(generate_block(elems_per_block,
                                        num_blocks,
                                        psco=False,
                                        set_to_zero=True))

    compss_barrier()
    initialization_time = time.time()

    # Do matrix multiplication
    dot(A, B, C, False)

    # Persist the result in a distributed manner (i.e: exploit data locality
    # & avoid memory flooding)
    for i in range(num_blocks):
        for j in range(num_blocks):
            persist_result(C[i][j])
            # If we are not going to check the result, we can safely delete
            # the Cij intermediate matrices
            if not check_result:
                from pycompss.api.api import compss_delete_object
                compss_delete_object(C[i][j])

    compss_barrier()
    multiplication_time = time.time()

    # Check if we get the same result if multiplying sequentially (no tasks)
    # Note that this implies having the whole A and B matrices in the master,
    # so it is advisable to set --check_result only with small matrices
    # Explicit correctness (i.e: an actual dot product is performed) must be
    # checked manually
    if check_result:
        from pycompss.api.api import compss_wait_on
        for i in range(num_blocks):
            for j in range(num_blocks):
                A[i][j] = compss_wait_on(A[i][j])
                B[i][j] = compss_wait_on(B[i][j])
        for i in range(num_blocks):
            for j in range(num_blocks):
                Cij = compss_wait_on(C[i][j])
                Dij = generate_block(elems_per_block,
                                     num_blocks,
                                     psco=False,
                                     set_to_zero=True)
                Dij = compss_wait_on(Dij)
                for k in range(num_blocks):
                    Dij += A[i][k] * B[k][j]
                import numpy as np
                if not np.allclose(Cij.block, Dij.block):
                    print('Block %d-%d gives different products!' % (i, j))
                    return
        print('Distributed and sequential results coincide!')

    print("-----------------------------------------")
    print("-------------- RESULTS ------------------")
    print("-----------------------------------------")
    print("Initialization time: %f" % (initialization_time - start_time))
    print("Multiplication time: %f" % (multiplication_time - initialization_time))
    print("Total time: %f" % (multiplication_time - start_time))
    print("-----------------------------------------")


def parse_args():
    """
    Arguments parser.
    Code for experimental purposes.
    :return: Parsed arguments.
    """
    import argparse
    parser = argparse.ArgumentParser(description='A COMPSs-PSCO blocked matmul implementation')
    parser.add_argument('-b', '--num_blocks', type=int, default=1,
                        help='Number of blocks (N in NxN)'
                        )
    parser.add_argument('-e', '--elems_per_block', type=int, default=2,
                        help='Elements per block (N in NxN)'
                        )
    parser.add_argument('--check_result', action='store_true',
                        help='Check obtained result'
                        )
    parser.add_argument('--seed', type=int, default=0,
                        help='Pseudo-Random seed'
                        )
    return parser.parse_args()


if __name__ == "__main__":
    opts = parse_args()
    main(**vars(opts))
