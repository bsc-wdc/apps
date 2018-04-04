# DBSCAN-PyCOMPSs
## 1. Introduction
DBSCAN for PyCOMPSs is a distributed approach to the well known clustering algorithm proposed for the first time in 1996 [here](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf "Original Implementation"). The application is implemented in Python and its parallelisation is  performed by the [COMPSs framework](https://www.bsc.es/research-and-development/software-and-apps/software-list/comp-superscalar/ "COMPSs Homepage"). 
### How it works?
  1. Initially, the dataset is chunked according to the point density, obtaining equally loaded chunks. 
  2. A core point retrieval is performed simultaneously at chunks of each chunk.
  3. Results are synced using an adjacency matrix.
  4. For each resulting cluster, a reachable point retrieval is performed and results lastly updated.
![](https://github.com/csegarragonz/DBSCAN-pyCOMPSs/blob/master/img/animation.gif "Main Workflow")
## 2. Files
In this repository you will find the following files and directories:
* `DBSCAN.py` contains the main algorithm and task invokation. It is not however standalone.
* `launchDBSCAN.py` launcher for the DBSCAN method, contains all the argument parsing and, if desired, performs the plotting. **_All the information required about the parameters, data format, etc can be found in this script_**
* `run.sh`some shell scripts to run the algorithm both in localhost and in a cluster with COMPSs installed.
* `/classes/` contains two modules imported by `DBSCAN.py`
  * One of them is a custom-built Cluster class.
  * The second one is a disjoint-set data structure (merge-find set) found [here]( https://github.com/imressed/python-disjoint-set "Link to the repo").
* `/data/`bunch of datasets to test the algorithm in.
  * `blobs_small.txt` 1500 points distributed in randomly placed blobs (is the dataset from the animation)
  * `blobs.txt`5000 points, `blobs_small.txt`is a subset of this dataset.
  * `moons.txt` 5000 moon-shaped points. Imported from [Sklearn datasets](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html "Link to the dataset")
* `/ext_versions/` contains other DBSCAN implementations that might be useful for benchmarking.
  * `DBSCAN_Seq.py`Sequential naive (all vs all) implementation of the algorithm.
* `/img/` contains the images used for creating the gif and the gif itself.
* `/kmeans/`contains an implementation of the k-means algorithm, in PyCOMPSs, used as well for benchmarking.
## 3. Requirements
  1. Python 2.7.x (with NumPy) **_COMPSs won't work with Python 3.x_**
  2. [COMPSs 2.1.](https://www.bsc.es/research-and-development/software-and-apps/software-list/comp-superscalar/downloads "Download COMPSs"), if you are trying to install it [this manual](http://compss.bsc.es/releases/compss/latest/docs/COMPSs_Installation_Manual.pdf?tracked=true "Link to COMPSs installation manual") might be useful.
  3. Matplotlib (in case plotting is desired)
## 4. Inquires and Contact
For any inquires or problems when trying to run the algorithm or COMPSs itself, don't hesitate to contact me at:
a=carlos.segarra
b=bsc.es
mailto: a @ b
