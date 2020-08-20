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

# PyCOMPSs imports
from pycompss.api.mpi import mpi
from pycompss.api.constraint import constraint
from pycompss.api.task import task
from pycompss.api.parameter import *


#
# TASKS DEFINITION: USAGE EXAMPLES
#
@mpi(runner="mpirun",
     binary="${BIN_DIR}/hello_world.x",
     processes="${MPI_TASK_NUM_NODES}",
     scale_by_cu=True)
@constraint(computing_units="${MPI_TASK_NUM_CUS}")
@task()
def mpi_example_empty():
    pass


@mpi(runner="mpirun",
     binary="${BIN_DIR}/parameters.x",
     processes="${MPI_TASK_NUM_NODES}",
     scale_by_cu=True)
@constraint(computing_units="${MPI_TASK_NUM_CUS}")
@task(num=IN, string=IN, file_path=FILE_IN)
def mpi_example_params(num, string, file_path):
    pass


@mpi(runner="mpirun",
     binary="${BIN_DIR}/hello_world.x",
     processes="${MPI_TASK_NUM_NODES}",
     scale_by_cu=True)
@constraint(computing_units="${MPI_TASK_NUM_CUS}")
@task(returns=1)
def mpi_example_with_ev():
    pass


@mpi(runner="mpirun",
     binary="${BIN_DIR}/hello_world.x",
     processes="${MPI_TASK_NUM_NODES}",
     scale_by_cu=True)
@constraint(computing_units="${MPI_TASK_NUM_CUS}")
@task(returns=1, stdout=FILE_OUT_STDOUT, stderr=FILE_OUT_STDERR)
def mpi_example_std(stdout, stderr):
    pass


@mpi(runner="mpirun",
     binary="${BIN_DIR}/hello_world.x",
     processes="${MPI_TASK_NUM_NODES}",
     working_dir="${OUTPUT_DIR}",
     scale_by_cu=True)
@constraint(computing_units="${MPI_TASK_NUM_CUS}")
@task()
def mpi_example_wd():
    pass


#
# TASKS DEFINITION: COMPLEX EXAMPLE
#
@task(file_path=FILE_OUT, block_size=IN)
def init_data_block_in_file(file_path, block_size):
    import random
    min_rand = 0
    max_rand = 256

    with open(file_path, 'w') as f:
        for _ in range(block_size):
            val = random.randint(min_rand, max_rand)
            f.write(str(val))
            f.write("\n")


@mpi(runner="mpirun",
     binary="${BIN_DIR}/complex.x",
     processes="${MPI_TASK_NUM_NODES}",
     working_dir="${OUTPUT_DIR}",
     scale_by_cu=True)
@constraint(computing_units="${MPI_TASK_NUM_CUS}")
@task(output=FILE_OUT, varargs_type=FILE_IN)
def mpi_compact(output, *args):
    pass


#
# MAIN METHODS
#
def main_usage_examples():
    # Imports
    from pycompss.api.api import compss_barrier
    from pycompss.api.api import compss_wait_on
    from pycompss.api.api import compss_open
    import os

    # LAUNCH SOME MPI EXAMPLES
    if __debug__:
        print("Launching MPI usage examples")

    # Launch empty MPI
    if __debug__:
        print("- Launching empty MPI")
    mpi_example_empty()
    compss_barrier()

    # Launch MPI with parameters
    if __debug__:
        print("- Launching MPI with parameters")
    num = 5
    string = "hello world"
    file_path = os.environ["OUTPUT_DIR"] + "/file.in"
    with open(file_path, 'w') as f:
        f.write("Hello World!\n")
    mpi_example_params(num, string, file_path)

    # Launch MPI with exit value
    if __debug__:
        print("- Launching MPI with EV")
    ev = mpi_example_with_ev()
    ev = compss_wait_on(ev)
    print("MPI EXIT VALUE = " + str(ev))
    if ev != 0:
        raise Exception("ERROR: MPI binary exited with non-zero value")

    # Launch MPI with STDOUT and STDERR
    if __debug__:
        print("- Launching MPI with STDOUT/STDERR")
    stdout = os.environ["OUTPUT_DIR"] + "/mpi_task_example.stdout"
    stderr = os.environ["OUTPUT_DIR"] + "/mpi_task_example.stderr"
    ev = mpi_example_std(stdout, stderr)
    ev = compss_wait_on(ev)
    print("MPI EXIT VALUE = " + str(ev))
    print("MPI STDOUT:")
    with compss_open(stdout) as f:
	print(f.read())
    print("MPI STDERR:")
    with compss_open(stderr) as f:
        print(f.read())

    if __debug__:
        print("- Launching MPI with working_dir")
    mpi_example_wd()
    compss_barrier()

    # DONE
    if __debug__:
        print("DONE Launching MPI usage examples")


def main_complex_example():
    # Imports
    from pycompss.api.api import compss_open

    # LAUNCH COMPLEX EXAMPLE
    if __debug__:
        print("Launching MPI complex example")

        # Constants
    n = 4
    m = 8
    block_size = 16

    # Generate data
    data = [[None for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            data[i][j] = "file_" + str(i) + "_" + str(j)
            init_data_block_in_file(data[i][j], block_size)

    # Process data
    compact_data = [None for _ in range(n)]
    for i in range(n):
        compact_data[i] = "compact_" + str(i)
        mpi_compact(compact_data[i], *data[i])

    final_data = "final_data"
    mpi_compact(final_data, *compact_data)

    # Synchronise
    with compss_open(final_data, "r") as f:
        print(f.read())

    # DONE
    if __debug__:
        print("DONE Launching MPI complex example")


#
# ENTRY POINT
#
if __name__ == '__main__':
    # Import libraries
    import time
    from pycompss.api.api import compss_barrier

    # Parse arguments
    # args = sys.argv[1:]

    # Log arguments if required
    if __debug__:
        print("Running MPI manager")

    # Start timers
    start_time = time.time()

    # Execute examples
    main_usage_examples()
    main_complex_example()

    # Log time
    compss_barrier()
    end_time = time.time()
    total_time = end_time - start_time
    print("TIME: " + str(total_time))
