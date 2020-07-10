# Object Oriented Kmeans Application

This application presents the Kmeans clustering algorithm parallelized with
PyCOMPSs and using persistent storage backend to deal with the points
fragments.

This application is composed of two main files:

```
src
  |- model
  |    |- __init__.py
  |    |- fragment.py
  |- storage_model
  |    |- __init__.py
  |    |- fragment.py
  |
  |- kmeans.py
```

The ```src/kmeans.py``` file contains the main of the Kmeans algorithm, while
the ```src/model/fragment.py``` contains the declaration of the Fragment class
with its necessary methods for the clustering, and the
```src/storage_model/fragment.py``` contains the declaration of the Fragment
considering persistent storage. Their methods are declared as
tasks for PyCOMPSs.

In addition, this application also contains a set of scripts to submit the
```kmeans.py``` application within the <ins>MN4 supercomputer</ins>
queuing system for the three currently supported persistent storage frameworks,
which are: **Redis**, **dataClay** and **Hecuba**.
The following commands submit the execution *without a job dependency*,
requesting *2 nodes*, with *5 minutes* walltime and with *tracing and graph
generation disabled* to perform the kmeans clustering of *1024* points
divided into *8* fragments, each point of *2* dimensions and looking for *4*
centers.

> Please, check the **[REQUIREMENTS](../README.md)** before using the following
commands.

* Launch with dataClay:
```bash
./launch_with_dataClay.sh None 2 5 false $(pwd)/src/kmeans.py -n 1024 -f 8 -d 2 -c 4
```

* Launch with Hecuba:
```bash
./launch_with_Hecuba.sh None 2 5 false $(pwd)/src/kmeans.py -n 1024 -f 8 -d 2 -c 4
```
* Launch with Redis:
```bash
./launch_with_redis.sh None 2 5 false $(pwd)/src/kmeans.py -n 1024 -f 8 -d 2 -c 4
```

And also, contains a script to run the ```kmeans.py``` application
<ins>locally</ins> with **Redis** to perform the kmeans clustering of *1024*
points divided into *8* fragments, each point of *2* dimensions and looking
for *4* centers:

```bash
./run_with_redis.sh
```

Furthermore, it can also be executed without persistent storage backend with
the same parameters:
```bash
./run.sh
```

## Available options

```
-n <NUM_POINTS>........ Number of points
 -d <DIMENSIONS>........ Number of dimensions
 -c <CENTRES>........... Number of centres
 -f <FRAGMENTS>......... Number of fragments
 -s <SEED>.............. Define a seed
                         (Default: 0)
 -m <MODE>.............. Distribution of points ( uniform | normal )
                         (Default: uniform)
 -i <ITERATIONS>........ Maximum number of iterations
                         (Default: 20)
 -e <EPSILON>........... Epsilon value. Convergence value.
                         (Deftault: 1e-9)
 -a <ARITY>............. Arity
                         (Default: 50)
 --use_storage.......... Use the available storage backend
```

## Issues

If any issue is found, please contact <support-compss@bsc.es>
