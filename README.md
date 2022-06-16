# COMPSs applications folder

This folder contains example applications developed for [py]COMPSs.

Its purpose is to share the applications among the COMPSs users community to ease
the implementation of new applications and share them with the rest of users.

## Datasets

The datasets for the applications can be downloaded using the following link:

* http://compss.bsc.es/repo/datasets

## Application's Folder Structure

Application are grouped by language.  
For instance, the matmul application that has version for PyCOMPSs, COMPSs with files,
COMPSs using objects and COMPSs with byte arrays and has the following tree directory.

```
apps
  |- python
  |    |- app_1
  |    |    |- src                 Application source code
  |    |    |- dist                Application binary or jar
  |    |    |- bin                 External binaries needed by the application
  |    |    |- lib                 External libraries needed by the application
  |    |    |- run_local.sh        Execution example for local automatic process
  |    |    |- launch.sh           Execution example for supercomputer automatic process
  |    |    |- clean.sh            Cleans the folder
  |    |    |- README              Brief description of the application
  |- java
  |    |- app_1
  |    |    | ...
  |- c
  |    |- ...
  |- datasets				Small datasets for examples

```

## The Distributed Computing Library (python only)

The Distributed Computing Library (dislib) provides distributed algorithms ready to use as a library. So far, dislib is highly focused on machine learning algorithms, and is greatly inspired by scikit-learn. However, other types of numerical algorithms might be added in the future. The main objective of dislib is to facilitate the execution of big data analytics algorithms in distributed platforms, such as clusters, clouds, and supercomputers.

You can find out more at:

* https://dislib.bsc.es/en/stable/


## Warnings and MN instructions

**REMINDER**: For each language there **must** be **one** launch.sh and **one** clean.sh script
          that executes the **stable** (base) version of that language. This script is used by
          the automatic benchmarking process.

**REMINDER**: For each version directory there **must** be a launcher script, a src directory with
the source code of that version, and a dist folder with the application binary or jar.


**WARNING**: Addtional binaries commonly used by all versions **must** be placed in the
         application's root folder.

**WARNING**: Additional input data used by all versions **must** be placed in a new folder
         inside the COMPSs_DATASETS with the **same** application name.
