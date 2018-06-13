# DBSCAN 4 PyCOMPSs
## 1. Introduction
DBSCAN for PyCOMPSs is a distributed approach to the well known clustering algorithm proposed for the first time in 1996 [here](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf "Original Implementation"). The application is implemented in Python and its parallelisation is  performed by the [COMPSs framework](https://www.bsc.es/research-and-development/software-and-apps/software-list/comp-superscalar/ "COMPSs Homepage"). 
## 2. Files
In this repository you will find the following files and directories:
* `DBSCAN.py` contains the main algorithm and task invokation. It requires however classes included in the `/classes/` folder.
* `/classes/` contains two modules imported by `DBSCAN.py`
  * One of them is a custom-built data class.
  * The second one is a disjoint-set data structure (merge-find set) found [here]( https://github.com/imressed/python-disjoint-set "Link to the repo").
* `run.sh`shell scripts to run the algorithm both in localhost and in a cluster with COMPSs installed.
* `launchDBSCAN.py` script to run a batch of executions using `launch.sh` as launcher. 
* `launch.sh` launcher for a slurm based cluster.
* `Gen_Data_DBSCAN.py` python script to generate randomly shaped clustering datasets as the ones in `/data/`.
* `/data/`bunch of datasets to test the algorithm in.
* `/ext_versions/` contains other DBSCAN implementations that might be useful for benchmarking.
  * `DBSCAN_Seq.py`Sequential naive (all vs all) implementation of the algorithm.
* `/kmeans/`contains an implementation of the k-means algorithm, in PyCOMPSs, used as well for benchmarking.
* `script_times.py` post-processing script to gather times from a big batch of executions.
## 3. Requirements
  1. Python 2.7.x (with NumPy) **_COMPSs won't work with Python 3.x_**
  2. [COMPSs Latest](https://www.bsc.es/research-and-development/software-and-apps/software-list/comp-superscalar/downloads "Download COMPSs"), if you are trying to install it [this manual](http://compss.bsc.es/releases/compss/latest/docs/COMPSs_Installation_Manual.pdf?tracked=true "Link to COMPSs installation manual") might be useful.
  3. Pandas 0.21 (this is the one I use, older versions may work as well but they need to support callables as arguments to the `pd.skip_rows` method.
## 4. Inquires and Contact
For any inquires or problems when trying to run the algorithm or COMPSs itself, don't hesitate to contact me at:
a=carlos.segarra
b=bsc.es
mailto: a @ b
