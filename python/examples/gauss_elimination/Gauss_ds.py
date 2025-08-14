#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.api.api import compss_wait_on, compss_barrier
import dislib as ds

# ----------------------------
# 1) PARALLEL TASK DEFINITIONS
# ----------------------------

@task(returns=list)
def map_block(block, row_index):
    """
    For a given matrix block and its starting row index in the global matrix:
    - Find the row index (inside the block) of the maximum element in each column
      using np.argmax over axis=0.
    - Return a list of tuples (max_value, global_row_index) for each column.
    This is used for pivoting in Gaussian elimination.
    """
    return [(block[row, col], row + row_index) 
            for col, row in enumerate(np.argmax(block, axis=0))]


@task(returns=list, mapped_results=COLLECTION_IN)
def reduce_max_indices(mapped_results):
    """
    Given results from multiple map_block calls:
    - Each mapped_result contains max values and corresponding row indices.
    - For each column, select the row index of the maximum value across all blocks.
    Returns: List of global row indices (one per column) with the largest pivot.
    """
    columns = zip(*mapped_results)
    return [max(col, key=lambda x: x[0])[1] for col in columns]


@task(blocks_needed=COLLECTION_INOUT, b_blocks_needed=COLLECTION_INOUT)
def swap_rows_if_needed(max_rows, blocks_needed, b_blocks_needed, k, block_height):
    """
    Swaps rows in the distributed matrix (A) and right-hand side vector (b) if needed.
    - max_rows: list of row indices where the maximum pivot element is located
    - blocks_needed: matrix A blocks involved in the swap
    - b_blocks_needed: vector b blocks involved in the swap
    - k: current pivot block index
    - block_height: number of rows in each block

    Handles both:
    - Swapping between different block rows
    - Swapping within the same block
    """
    for diag_row, max_row in enumerate(max_rows):
        max_block_row = int(max_row / block_height)
        max_row_in_block = max_row % block_height

        # Only swap if pivot row is not already the current diagonal row
        if (max_row_in_block != diag_row or max_block_row != k):
            if (max_block_row != k):
                # Swap between two different block rows
                # Swap in matrix A
                for i in range(len(blocks_needed[k])):
                    blocks_needed[k][i][diag_row], blocks_needed[max_block_row][i][max_row_in_block] = \
                        blocks_needed[max_block_row][i][max_row_in_block], np.copy(blocks_needed[k][i][diag_row])
                # Swap in vector b
                for i in range(len(b_blocks_needed[k])):
                    b_blocks_needed[k][i][diag_row], b_blocks_needed[max_block_row][i][max_row_in_block] = \
                        b_blocks_needed[max_block_row][i][max_row_in_block], np.copy(b_blocks_needed[k][i][diag_row])
            else:
                # Swap rows inside the same block row
                for block in blocks_needed[k]:
                    block[diag_row], block[max_row_in_block] = block[max_row_in_block], np.copy(block[diag_row])
                for block in b_blocks_needed[k]:
                    block[diag_row], block[max_row_in_block] = block[max_row_in_block], np.copy(block[diag_row])


@task(pivot_block_row=COLLECTION_IN, block_row=COLLECTION_INOUT, b_pivot_block=IN, b_block_row=INOUT)
def reduce_block_row(pivot_block_row, block_row, b_pivot_block, b_block_row):
    """
    Performs Gaussian elimination on one block row (below the pivot).
    - pivot_block_row: the pivot row of A
    - block_row: the row of A being reduced
    - b_pivot_block: pivot block of vector b
    - b_block_row: row of vector b being reduced
    """
    for k in range(pivot_block_row[0].shape[1]):
        multipliers = []
        pivot = pivot_block_row[0][k][k]  # pivot element

        # Compute elimination multipliers for each row
        for row in range(block_row[0].shape[0]):
            multipliers.append((row, block_row[0][row][k] / pivot))

        # Apply elimination to matrix blocks
        for pivot_block, block in zip(pivot_block_row, block_row):
            for row, multiplier in multipliers:
                block[row] -= multiplier * pivot_block[k]

        # Apply elimination to vector b
        for row, multiplier in multipliers:
            b_block_row[row] -= multiplier * b_pivot_block[k]


def pivot_block_gauss(pivot_block_row, b_pivot_block):
    """
    Performs Gaussian elimination only on the pivot block row.
    This function handles the top-left pivot block and modifies the rest of the pivot row.
    """
    multipliers = reduce_pivot_block(pivot_block_row[0])
    for block in pivot_block_row[1:]:
        reduce_pivot_row_block(block, multipliers)
    reduce_pivot_row_block(b_pivot_block, multipliers)


@task(block=INOUT)
def reduce_pivot_row_block(block, multipliers):
    """
    Apply previously computed multipliers to a block in the pivot row.
    """
    for k, multipliers_k in enumerate(multipliers):
        for row, multiplier in multipliers_k:
            block[row] -= multiplier * block[k]


@task(returns=1, block=INOUT)
def reduce_pivot_block(block):
    """
    Perform elimination on the pivot block itself and compute the multipliers.
    Returns: multipliers for use on the rest of the pivot row.
    """
    multipliers = [[] for _ in range(block.shape[1])]
    for k in range(block.shape[1]):
        pivot = block[k][k]
        for row in range(k + 1, block.shape[1]):
            multipliers[k].append((row, block[row][k] / pivot))
        for row, multiplier in multipliers[k]:
            block[row] -= multiplier * block[k]
    return multipliers


@task(block=IN, x_block=IN)
def back_subs_non_diagonal(block, x_block):
    """
    Computes partial results for back substitution for a non-diagonal block.
    Returns: list of values to subtract from the diagonal solution.
    """
    subtractors = [0 for _ in range(len(block))]
    for i in range(len(block)):
        row = block[i]
        for row_value, x_value in zip(row, x_block):
            subtractors[i] += row_value * x_value
    return subtractors


@task(block=IN, b_block=IN, x_block=INOUT, all_subtractors=COLLECTION_IN)
def back_subs_diagonal(block, b_block, x_block, all_subtractors):
    """
    Performs back substitution on the diagonal block to solve for x.
    Subtracts contributions from already-solved variables.
    """
    for i in range(len(block) - 1, -1, -1):
        x_block[i] = b_block[i]
        for subtractor in all_subtractors:
            x_block[i] -= subtractor[i]
        for j in range(i + 1, len(block), 1):
            x_block[i] -= block[i][j] * x_block[j]
        x_block[i] /= block[i][i]

# ----------------------------
# 2) DATASET READING FUNCTIONS
# ----------------------------

def read_simple_matrix():
    """
    Returns a small fixed system Ax = b for debugging.
    Block size is small (2x2 for A, 2x1 for b).
    """
    x = ds.array(np.array((
        [4, 2, 4, 1, 19],
        [15, 1, 3, 2, 1],
        [1, 19, 1, 5, 1],
        [4, 2, 12, 3, 1],
        [2, 0, 5, 15, 4]), dtype=float), block_size=(2, 2))
    y = ds.array(np.array((
        [2], [4], [5], [1], [5]), dtype=float), block_size=(2, 1))
    return x, y


def read_random_matrix(n, block_size, seed=0):
    """
    Generates a random, diagonally dominant system Ax = b that is guaranteed to be invertible.
    - n: size of the matrix
    - block_size: tuple (rows_per_block, cols_per_block)
    - seed: RNG seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    # Random integer solution (small integers)
    x = rng.integers(-9, 10, size=n, dtype=np.int64)
    print(x)

    # Random float matrix, then make diagonally dominant
    A = rng.random((n, n), dtype=np.float64)
    A[np.diag_indices(n)] += 10.0  # ensure diagonal dominance

    # Compute b = A @ x
    b = A @ x

    # Convert to dislib distributed arrays
    A = ds.array(A, block_size=block_size)
    b = ds.array(b, block_size=(block_size[0], 1))
    return A, b

# ----------------------------
# 3) OUTPUT TASK
# ----------------------------

@task(file_path=FILE_OUT)
def write_solution(file_path, x):
    """
    Writes solution vector x to a file.
    """
    with open(file_path, 'w') as f:
        for value in x:
            f.write(str(value) + " ")

# ----------------------------
# 4) MAIN SCRIPT
# ----------------------------

if __name__ == "__main__":
    solutionPathDataset = os.getcwd() + '/gauss_dataset_sol'

    systems = []
    # Example with a large random system (10,000 x 10,000)
    systems.append((read_random_matrix(10_000, (500, 500)), "exemple_random.txt"))
    # systems.append((read_simple_matrix(), "exemple_simple.txt"))

    for sys_index in range(len(systems)):
        A = systems[sys_index][0][0]  # Matrix A
        b = systems[sys_index][0][1]  # Vector b
        n = A.shape[0]
        block_height, block_width = A._reg_shape
        x = ds.array(np.zeros(n), block_size=(block_height, 1))  # Solution vector

        # --- FORWARD ELIMINATION ---
        for k in range(A._n_blocks[1]):
            block_col = A._get_col_block(k)._blocks[k:]
            pivot_block = block_col[0][0]

            # Step 1: Find pivot rows
            max_rows = []
            for index, block in enumerate(block_col):
                max_rows.append(map_block(block[0], (k + index) * block_height))
            max_rows = reduce_max_indices(max_rows)
            max_rows = compss_wait_on(max_rows)

            # Step 2: Gather blocks needed for possible row swap
            blocks_needed = [0 for _ in range(A._n_blocks[0])]
            b_blocks_needed = [0 for _ in range(A._n_blocks[0])]
            blocks_needed[k] = A._blocks[k]
            b_blocks_needed[k] = b._blocks[k]

            for max_row in max_rows:
                max_block_row = int(max_row / block_height)
                if blocks_needed[max_block_row] == 0:
                    blocks_needed[max_block_row] = A._blocks[max_block_row]
                    b_blocks_needed[max_block_row] = b._blocks[max_block_row]

            # Step 3: Swap rows if needed
            swap_rows_if_needed(max_rows, blocks_needed, b_blocks_needed, k, block_height)

            # Step 4: Pivot block Gaussian elimination
            pivot_block_row = A._blocks[k][k:]
            b_pivot_block = b._blocks[k][0]
            pivot_block_gauss(pivot_block_row, b_pivot_block)

            # Step 5: Reduce rows below pivot
            for i in range(len(A._blocks[k+1:])):
                block_row = A._blocks[k + 1 + i][k:]
                b_block_row = b._blocks[k + 1 + i][0]
                reduce_block_row(pivot_block_row, block_row, b_pivot_block, b_block_row)

        # --- BACK SUBSTITUTION ---
        for i in range(len(A._blocks) - 1, -1, -1):
            block_row = A._blocks[i]
            b_block = b._blocks[i][0]

            all_subtractors = []
            for j in range(len(A._blocks) - 1, i, -1):
                all_subtractors.append(back_subs_non_diagonal(block_row[j], x._blocks[j][0]))

            back_subs_diagonal(block_row[i], b_block, x._blocks[i][0], all_subtractors)

        # Wait for solution and write to file
        x = compss_wait_on(x)
        file_path_sol = os.path.join(solutionPathDataset, systems[sys_index][1])
        write_solution(file_path_sol, x.collect())
