"""
A matrix multiplication implementation with (optionally) PSCOs
author: Sergio Rodriguez Guasch < sergio rodriguez at bsc dot es >
"""
from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.api.constraint import constraint

#@constraint(ComputingUnits="${ComputingUnits}")
#@task(returns=1)
def generate_block(size, num_blocks, seed=0, psco=False, use_storage=True, set_to_zero=False, p_name=''):
    """
    Generate a square block of given size.
    :param size: <Integer> Block size
    :param num_blocks: <Integer> Number of blocks
    :param seed: <Integer> Random seed
    :param psco: <Boolean> If psco
    :param use_storage: <Boolean> Use storage
    :param set_to_zero:<Boolean> Set block to zeros
    :return: Block (with storage) or np.matrix (otherwise)
    """
    import numpy as np
    np.random.seed(seed)
    b = np.matrix(np.random.random((size, size)) if not set_to_zero else np.zeros((size, size)))
    # Normalize matrix to ensure more numerical precision
    if not set_to_zero:
        b /= np.sum(b) * float(num_blocks)
    if psco:
        if use_storage:
            from classes.block import Block
        else:
            from classes.fake_block import Block
        ret = Block()
        ret.block = b  # Hecuba assignment
        if use_storage:
            ret.make_persistent(p_name)
    else:
        ret = b
    return ret


@task(C=INOUT)
def multiply(A, B, C):
    """
    Multiplies two blocks and accumulates the result in an INOUT matrix
    :param A: Block A
    :param B: Block B
    :param C: Result Block
    :return: None
    """
    import numpy as np
    #print "GREP ME PLEASE"
    #print np.__version__
    #print np.__file__
    C += np.dot(A.block, B.block)
    # C += A.block * B.block  # will not work (multiply element by element)
    # C += A * B  # This would work if __mult__ was supported.


def dot(A, B, C, set_barrier=False):
    """
    A COMPSs-PSCO blocked matmul algorithm.
    A and B (blocks) are PSCOs, while C (blocks) are objects
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

# 28 as computing units worked fine
@constraint(ComputingUnits="${ComputingUnits}")
@task()
def persist_result(b, p_name=''):
    """
    Persist results of block b.
    :param b: Block to persist
    :return: None
    """
    from classes.block import Block
    bl = Block()
    bl.block = b  # Hecuba assignment
    print "YYY " + p_name
    bl.make_persistent(p_name)


def main(num_blocks, elems_per_block, check_result, seed, use_storage):
    """
    Matmul main.
    :param num_blocks: <Integer> Number of blocks
    :param elems_per_block: <Integer> Number of elements per block
    :param check_result: <Boolean> Check results against sequential version of matmul
    :param seed: <Integer> Random seed
    :param use_storage: <Boolean> Use storage
    :return: None
    """
    from pycompss.api.api import compss_barrier
    import time

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
                l[-1].append(generate_block(elems_per_block, num_blocks, seed=seed + bid, psco=True, use_storage=use_storage, p_name=matrix_name[ix] + str(i) + 'g' + str(j)))
                bid += 1
            #C[-1].append(generate_block(elems_per_block, num_blocks, psco=False, set_to_zero=True, use_storage=use_storage, p_name=''))
    compss_barrier()

    initialization_time = time.time()
    '''
    # Do matrix multiplication
    dot(A, B, C, False)

    # Persist the result in a distributed manner (i.e: exploit data locality & avoid memory flooding)
    for i in range(num_blocks):
        for j in range(num_blocks):
            if use_storage:
                persist_result(C[i][j], 'C' + str(i) + 'g' + str(j))
            # If we are not going to check the result, we can safely delete the Cij intermediate matrices
            if not check_result:
                from pycompss.api.api import compss_delete_object
                #compss_delete_object(C[i][j])

    compss_barrier()
    '''
    multiplication_time = time.time()
 
    # Check if we get the same result if multiplying sequentially (no tasks)
    # Note that this implies having the whole A and B matrices in the master,
    # so it is advisable to set --check_result only with small matrices
    # Explicit correctness (i.e: an actual dot product is performed) must be checked
    # manually
    if check_result:
        from pycompss.api.api import compss_wait_on
        for i in range(num_blocks):
            for j in range(num_blocks):
                A[i][j] = compss_wait_on(A[i][j])
                B[i][j] = compss_wait_on(B[i][j])
                #print "ELOY WAS HERE"
                #print 'Aij' + str(i) + str(j) + ":"
                #print A[i][j].block
                #print 'Bij' + str(i) + str(j) + ":"
                #print B[i][j].block
                
        for i in range(num_blocks):
            for j in range(num_blocks):
                Cij = compss_wait_on(C[i][j])
                Dij = generate_block(elems_per_block, num_blocks, psco=False, set_to_zero=True)
                Dij = compss_wait_on(Dij)
                import numpy as np
                for k in range(num_blocks):
                    Dij += np.dot(A[i][k].block, B[k][j].block)
                #print "ELOY WAS HERE"
                #print 'Cij' + str(i) + str(j) + ":"
                #print Cij
                #print 'Dij' + str(i) + str(j) + ":"
                #print Dij
                if not np.allclose(Cij, Dij):
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
    compss_barrier()

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
    parser.add_argument('--use_storage', action='store_true',
                        help='Use storage?'
                        )
    return parser.parse_args()


if __name__ == "__main__":
    opts = parse_args()
    main(**vars(opts))
