This is the Readme for:  
gaussian-elimination-ds

[Name]: gaussian-elimination-ds  
[Contact Person]: support-compss@bsc.es  
[License Agreement]: Apache2  
[Platform]: COMPSs  

[Body]  
== Description ==  
Distributed Gaussian Elimination using COMPSs and dislib.  
This application solves large dense linear systems **A·x = b** by performing pivoted Gaussian elimination in a **block-distributed** fashion, using PyCOMPSs tasks for parallelism.  

== Execution instructions ==  

* Usage in supercomputer:  

    ./launch.sh

    This script is already configured to:
    - Use Python version 3.12.1
    - Load COMPSs 3.3.3 and dislib master
    - Submit the job with:
        - 1 node
        - 60 minutes walltime
        - Tracing enabled
        - GPU flag (-g)
        - Project name: bsc19
        - Log level: debug
        - Master and worker working directories set to the current directory
        - Python path set to the current directory
        - Python interpreter set to python3
        - Language: python
        - Main program: Gauss_ds.py

    If you want to change the number of nodes, execution time, tracing option, or other parameters, edit the corresponding flags in `launch.sh` before running it.

    Example run (as provided):
    ```bash
    ./launch.sh
    ```

== Build ==  
No build is required.

