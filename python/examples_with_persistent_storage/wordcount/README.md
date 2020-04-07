# Wordcount Application

This application presents the wordcount algorithm parallelized with
PyCOMPSs and using persistent storage backend to deal with multiple blocks
of text.

This application is composed of the following files:

```
src
  |- classes
  |    |- __init__.py
  |    |- block.py
  |
  |- wordcount.py

dataset
  |- ...
```

The ```src/wordcount.py``` file contains the main of the Wordcount algorithm,
while the ```src/classes/block.py``` contains the declaration of the Words
class with its necessary methods for text addition and retrieval. The
```dataset``` folder contains a set of 4 testing text files to perform the
wordcount.

In addition, this application also contains a set of scripts to submit the
```wordcount.py``` application within the <ins>MN4 supercomputer</ins>
queuing system for the three currently supported persistent storage frameworks,
which are: **Redis**, **dataClay** and **Hecuba**.
The following commands submit the execution *without a job dependency*,
requesting *2 nodes*, with *5 minutes* walltime and with *tracing and graph
generation disabled* to perform the wordcount of the dataset stored in the
```dataset``` folder.

> Please, check the **[REQUIREMENTS](../README.md)** before using the following commands.

* Launch with dataClay:
```bash
./launch_with_dataClay.sh None 2 5 false $(pwd)/src/wordcount.py -d $(pwd)/dataset
```

* Launch with Hecuba:
```bash
./launch_with_Hecuba.sh None 2 5 false $(pwd)/src/wordcount.py -d $(pwd)/dataset
```
* Launch with Redis:
```bash
./launch_with_redis.sh None 2 5 false $(pwd)/src/wordcount.py -d $(pwd)/dataset
```

And also, contains a script to run the ```wordcount.py``` application
<ins>locally</ins> with **Redis** to perform the wordcount of the dataset stored in the
```dataset``` folder:

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
