# COMPSs applications folder

This folder contains example applications developed for [py]COMPSs . 

Its purpose is to share the applications among the COMPSs users community to ease
the implementation of new applications and share them with the rest of users.


## Application's Folder Structure 

Application are grouped by language.  
For instance, the matmul application that has version for PyCOMPSs, COMPSs with files,
COMPSs using objects and COMPSs with byte arrays and has the following tree directory.

```
apps
  |- python 
  |    |- app_1
  |    |    |- version_1
  |    |    |    |- src                 Application source code
  |    |    |    |- dist                Application binary or jar
  |    |    |    |- bin                 External binaries needed by the application
  |    |    |    |- lib                 External libraries needed by the application
  |    |    |    |- launch.sh           Execution example (must use enqueue_compss)
  |    |    |- ...
  |    |    |- version_n
  |    |    |- launch.sh                Execution example for automatic process
  |    |    |- clean.sh                 Cleans the folder for *ALL* versions
  |    |    |- README                   Brief description of the application
  |- java
  |    |- app_1
  |    |    | ...
  |- c
  |    |- ...
  |- datasets				Small datasets for examples 

```
## Datasets

The datasets for the applications are on the Capella VM (bsccs1000.int.bsc.es). You can access them through SSH or via browser using the link: http://http://compss.bsc.es/repo/datasets.


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

