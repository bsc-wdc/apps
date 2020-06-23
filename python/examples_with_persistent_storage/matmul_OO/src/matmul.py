import time
import numpy as np
from pycompss.api.task import task
from pycompss.api.parameter import CONCURRENT
from pycompss.api.api import compss_barrier
from pycompss.api.api import compss_wait_on
from pycompss.api.api import compss_delete_object


@task(returns=1)
def generate_block(size, num_blocks, seed=0, use_storage=False,
                   set_to_zero=False, psco_name=''):
    """
    Generate a square block of given size.
    :param size: <Integer> Block size
    :param num_blocks: <Integer> Number of blocks
    :param seed: <Integer> Random seed
    :param use_storage: <Boolean> If use_storage
    :param set_to_zero: <Boolean> Set block to zeros
    :param psco_name: <String> Persistent object name if use_storage
    :return: Block (persisted if use_storage)
    """
    if use_storage:
        from storage_model.block import Block
        ret = Block()
        ret.make_persistent(psco_name)  # here better for dataClay
        ret.generate_block(size,
                           num_blocks,
                           seed=seed)
        # ret.make_persistent(psco_name)  # here is a need for redis
    else:
        from model.block import Block
        ret = Block()
        ret.generate_block(size,
                           num_blocks,
                           set_to_zero=True)
    return ret


@task(C=CONCURRENT)
def fused_multiply_add(A, B, C):
    """
    Multiplies two Blocks and accumulates the result in an INOUT Block (FMA).
    :param A: Block A
    :param B: Block B
    :param C: Result Block
    :return: None
    """
    C.fused_multiply_add(A, B)


def dot(A, B, C):
    """
    A COMPSs-PSCO blocked matmul algorithm.
    A and B (blocks) can be persistent PSCOs, while C (blocks) are non
    persistent blocks == objects
    :param A: Block A
    :param B: Block B
    :param C: Result Block
    :return: None
    """
    n, m = len(A), len(B[0])
    # as many rows as A, as many columns as B
    for i in range(n):
        for k in range(n):
            for j in range(m):
                # We want to exploit the concurrentness of C[i][j]
                # but avoid it becoming a choke point. That's why we 
                # don't iterate [i, j, k] but [i, k, j] instead.
                fused_multiply_add(A[i][k], B[k][j], C[i][j])


@task()
def persist_result(b, psco_name=''):
    """
    Persist results of block b.
    :param b: Block to persist
    :param psco_name: Persistent object name
    :return: None
    """
    from storage_model.block import Block
    bl = Block(b)
    bl.make_persistent(psco_name)


def main(num_blocks, elems_per_block, check_result, seed, use_storage):
    """
    Matmul main.
    :param num_blocks: <Integer> Number of blocks
    :param elems_per_block: <Integer> Number of elements per block
    :param check_result: <Boolean> Check results against sequential version
                         of matmul
    :param seed: <Integer> Random seed
    :param use_storage: <Boolean> Use storage
    :return: None
    """
    print("Starting application")
    start_time = time.time()

    # Generate the dataset in a distributed manner
    # i.e: avoid having the master a whole matrix
    A, B, C = [], [], []
    matrix_name = ["A", "B"]
    for i in range(num_blocks):
        for l in [A, B, C]:
            l.append([])
        # Keep track of blockId to initialize with different random seeds
        bid = 0
        for j in range(num_blocks):
            for ix, l in enumerate([A, B]):
                psco_name = "%s%02dg%02d" % (matrix_name[ix], i, j)
                l[-1].append(generate_block(elems_per_block,
                                            num_blocks,
                                            seed=seed + bid,
                                            use_storage=use_storage,
                                            psco_name=psco_name))
                bid += 1
            C[-1].append(generate_block(elems_per_block,
                                        num_blocks,
                                        set_to_zero=True,
                                        use_storage=use_storage,
                                        psco_name="C%02dg%02d" % (i, j)))
    compss_barrier()
    print("Data generated; proceeding to do matrix multiplication")
    initialization_time = time.time()

    # Do matrix multiplication
    dot(A, B, C)

    compss_barrier()
    multiplication_time = time.time()

    print("Multiplication finished")

    # Check if we get the same result if multiplying sequentially (no tasks)
    # Note that this implies having the whole A and B matrices in the master,
    # so it is advisable to set --check_result only with small matrices
    # Explicit correctness (i.e: an actual dot product is performed) must be
    # checked manually
    if check_result:
        print("Checking result")
        for i in range(num_blocks):
            for j in range(num_blocks):
                A[i][j] = compss_wait_on(A[i][j])
                B[i][j] = compss_wait_on(B[i][j])
        for i in range(num_blocks):
            for j in range(num_blocks):
                Cij_obj = compss_wait_on(C[i][j])
                Cij = Cij_obj.block
                Dij_obj = generate_block(elems_per_block,
                                     num_blocks,
                                     use_storage=False,
                                     set_to_zero=True)
                Dij_obj = compss_wait_on(Dij_obj)
                Dij = Dij_obj.block
                import numpy as np
                for k in range(num_blocks):
                    Dij += np.dot(A[i][k].block, B[k][j].block)
                if not np.allclose(Cij, Dij):
                    print('Block %d-%d gives different products!' % (i, j))
                    return
        print('Distributed and sequential results coincide!')

    print("-----------------------------------------")
    print("-------------- RESULTS ------------------")
    print("-----------------------------------------")
    print("Initialization time: %f" % (initialization_time -
                                       start_time))
    print("Multiplication time: %f" % (multiplication_time -
                                       initialization_time))
    print("Total time: %f" % (multiplication_time - start_time))
    print("-----------------------------------------")


def parse_args():
    """
    Arguments parser.
    Code for experimental purposes.
    :return: Parsed arguments.
    """
    import argparse
    description = 'Object Oriented COMPSs-PSCO blocked matmul implementation'
    parser = argparse.ArgumentParser(description=description)
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
    parser.add_argument('--use_storage', action='store_true',
                        help='Use storage?'
                        )
    return parser.parse_args()


if __name__ == "__main__":
    opts = parse_args()
    main(**vars(opts))
