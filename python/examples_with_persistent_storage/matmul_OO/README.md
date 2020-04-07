# Object Oriented Matrix Multiplication Application

This application presents the Matrix multiplication algorithm parallelized with
PyCOMPSs and using persistent storage backend to deal with the points fragments.

This application is composed of two main files:

```
src
  |- classes
  |    |- __init__.py
  |    |- block.py
  |
  |- matmul.py
```

The ```src/matmul.py``` file contains the main of the Matrix Multiplication
algorithm, while the ```src/classes/block.py``` contains the declaration of
the Block class with its necessary methods for block multiplication and
addition.

In addition, this application also contains a set of scripts to submit the
```matmul.py``` application within the <ins>MN4 supercomputer</ins>
queuing system for the three currently supported persistent storage frameworks,
which are: **Redis**, **dataClay** and **Hecuba**.
The following commands submit the execution *without a job dependency*,
requesting *2 nodes*, with *5 minutes* walltime and with *tracing and graph
generation disabled* to perform the matrix multiplication of two matrices of
4 blocks with 4 elements per block, checking the result.

> Please, check the **[REQUIREMENTS](../README.md)** before using the following commands.

* Launch with dataClay:
```bash
./launch_with_dataClay.sh None 2 5 false $(pwd)/src/matmul.py -b 4 -e 4 --check_result
```

* Launch with Hecuba:
```bash
./launch_with_Hecuba.sh None 2 5 false $(pwd)/src/matmul.py -b 4 -e 4 --check_result
```
* Launch with Redis:
```bash
./launch_with_redis.sh None 2 5 false $(pwd)/src/matmul.py -b 4 -e 4 --check_result
```

And also, contains a script to run the ```matmul.py``` application
<ins>locally</ins> with **Redis** to perform the matrix multiplication of two
matrices of 4 blocks with 4 elements per block, checking the result.:

```bash
./run_with_redis.sh
```

Furthermore, it can also be executed without persistent storage backend with
the same parameters:
```bash
./run.sh
```

## Issues

If any issue is found, please contact <support-compss@bsc.es>
