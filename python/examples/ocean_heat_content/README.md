# Ocean Heat Content

## Description

Ocean Heat Content is a tool that analyses the heat variations in the surface
of the sea.

Uses Numba to improve the performance of the analysis, and it is also able to
exploit the usage of GPUs.

* **Access Level**: private
* **License Agreement**: Private
* **Platform**: COMPSs
* **Language**: Python

## Versions

### 1.- per_layer
Ocean heat content sequential code paralelized with PyCOMPSs considering the
computation of each layer as a task. It has multiple implementations, so the
computation can be performed in CPU and GPU if there are available resources
transparently.

### 2.- all_layers
Simpler version of *per_layer* that considers the computation of all layers
of each input file within a task. This version allows to select that the
resource where the tasks that have various implementations are expected to
be performed (e.g. only CPU or only in GPU).

                
## Requirements

* Python 2.7. or higher
* PyCOMPSs
* Numba
* Iris
* Numpy

## Execution instructions

### Usage:

* 1.- per_layer
```
per_layer/oceanheatcontent.py <DATASET_FOLDER> <MESH_FILE> <REGIONS_FILE> <OUTPUT_FOLDER>
```
* 2.- all_layers
```
all_layers/oceanheatcontent.py <DATASET_FOLDER> <MESH_FILE> <REGIONS_FILE> <OUTPUT_FOLDER> <ONLY_GPU>
```

Where:
* DATASET_FOLDER: Input dataset path.
* MESH_FILE: Mesh file path.
* REGIONS_FILE: Regions file path.
* OUTPUT_FOLDER: Output folder where to store the results.
* ONLY_GPU: Use only GPUs (True | False). If false, uses only CPUs.

### Local:

To run Ocean Heat Content locally, please use the ```run_local_per_layer.sh``` and
```run_local_all_layers.sh``` scripts.

These scripts contain the parameters described in the previous *usage* subsection and invoke
the application using the COMPSs' ```runcompss``` command. 

After checking the parameters of the desired version, run:
```
./run_local_per_layer.sh
```
or
```
./run_local_all_layers.sh
```

More information about the ```runcompss``` command available flags can be found in the 
[compss execution manual](http://compss.bsc.es/releases/compss/latest/docs/COMPSs_User_Manual_App_Exec.pdf)
 
### Supercomputer - CTE-Power9:

#### Configuration:

The user **MUST** load the following modules in order to run Ocean Heat Content:

```
module use /gpfs/projects/bsc32/software/rhel/7.4/ppc64le/POWER9/modules/all/
module load iris
module load numba
export COMPSS_PYTHON_VERSION=none
module load COMPSs/2.5 
```

#### Job submission:

To submit a Ocean Heat Content job to CTE-Power9, please use the ```launch_per_layer.sh``` and
```launch_all_layers.sh``` scripts.

These scripts contain the parameters described in the previous *usage* subsection and invoke
the application using the COMPSs' ```enqueue_compss``` command. 

After checking the parameters of the desired version, run:
```
./launch_per_layer.sh
```
or
```
./launch_all_layers.sh
```

More information about the ```enqueue_compss``` command available flags can be found in the 
[compss supercomputer manual](http://compss.bsc.es/releases/compss/latest/docs/COMPSs_Supercomputers_Manual.pdf)

## Notes and Considerations

Please, take into account to remove the output folder if before reusing it.

## Build
No build is required

## Contact
[Saskia Loosveldt](mailto:saskia.loosveldt@bsc.es)

[Javier Vegas](mailto:javier.vegas@bsc.es)

[Kim Serradell](mailto:kim.serradell@bsc.es)


