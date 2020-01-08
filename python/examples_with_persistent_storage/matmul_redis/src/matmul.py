'''A matrix multiplication implementation with (optionally) PSCOs
author: Sergio Rodriguez Guasch < sergio rodriguez at bsc dot es >
'''

import numpy as np
from pycompss.api.task import task
from pycompss.api.parameter import *

@task(C = INOUT)
def multiply(A, B, C):
  '''Multiplies two blocks and acumulates the result in an INOUT
  matrix
  '''
  C += np.dot(A.block, B.block)

def dot(A, B, C, set_barrier = False):
  '''A COMPSs-PSCO blocked matmul algorithm
  A and B (blocks) are PSCOs, while C (blocks) are objects
  '''
  n, m = len(A), len(B[0])
  # as many rows as A, as many columns as B
  for i in range(n):
    for j in range(m):
      for k in range(n):
        multiply(A[i][k], B[k][j], C[i][j])
  if set_barrier:
    from pycompss.api.api import compss_barrier
    compss_barrier()


'''Code for experimental purposes.
'''
def parse_args():
  import argparse
  parser = argparse.ArgumentParser(description = 'A COMPSs-PSCO blocked matmul implementation')
  parser.add_argument('-b', '--num_blocks', type = int, default = 1,
                     help = 'Number of blocks (N in NxN)'
                     )
  parser.add_argument('-e', '--elems_per_block', type = int, default = 2,
                     help = 'Elements per block (N in NxN)'
                     )
  parser.add_argument('--check_result', action = 'store_true',
                     help = 'Check obtained result'
                     )
  parser.add_argument('--seed', type = int, default = 0,
                     help = 'Pseudo-Random seed'
                     )
  parser.add_argument('--use_storage', action = 'store_true',
                     help = 'Use storage?'
                     )
  return parser.parse_args()

@task(returns = 1)
def generate_block(size, num_blocks, seed = 0, psco = False, use_storage = True, set_to_zero = False):
  '''Generate a square block of given size.
  '''
  np.random.seed(seed)
  b = np.matrix(
    np.random.random((size, size)) if not set_to_zero else np.zeros((size, size))
  )
  # Normalize matrix to ensure more numerical precision
  if not set_to_zero:
    b /= np.sum(b) * float(num_blocks)
  if psco:
    if use_storage:
      from block import Block
    else:
      from fake_block import Block
    ret = Block(b)
    if use_storage:
      ret.make_persistent()
  else:
    ret = b
  return ret

@task()
def persist_result(b):
  from block import Block
  bl = Block(b)
  bl.make_persistent()

def main(num_blocks, elems_per_block, check_result, seed, use_storage):
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
        l[-1].append(generate_block(elems_per_block, num_blocks, seed = seed + bid, psco = True, use_storage = use_storage))
        bid += 1
      C[-1].append(generate_block(elems_per_block, num_blocks, psco = False, set_to_zero = True, use_storage = use_storage))
  dot(A, B, C, True)
  # Persist the result in a distributed manner (i.e: exploit data locality &
  # avoid memory flooding)
  for i in range(num_blocks):
    for j in range(num_blocks):
      if use_storage:
        persist_result(C[i][j])
      # If we are not going to check the result, we can safely delete the Cij intermediate
      # matrices
      if not check_result:
        from pycompss.api.api import compss_delete_object as del_obj
        del_obj(C[i][j])
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
    for i in range(num_blocks):
      for j in range(num_blocks):
        Cij = compss_wait_on(C[i][j])
        Dij = generate_block(elems_per_block, num_blocks, psco = False, set_to_zero = True)
        Dij = compss_wait_on(Dij)
        for k in range(num_blocks):
          Dij += np.dot(A[i][k].block, B[k][j].block)
        if not np.allclose(Cij, Dij):
          print('Block %d-%d gives different products!' % (i, j))
          return
    print('Distributed and sequential results coincide!')


if __name__ == "__main__":
  opts = parse_args()
  main(**vars(opts))
