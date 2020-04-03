# Applications using PyCOMPSs with a Persistent Storage Backend

This folder contains applications parallelized with PyCOMPSs using persistent
storage frameworks. The currently existing API implementations between PyCOMPSs
and persistent storage frameworks are for **dataClay**, **Hecuba** and
**Redis**.

Each application contains a sample script to launch the application within the
**MN4 Supercomputer** for these three storage frameworks, and also another
script to run it locally with **Redis**.

Before running any application, there are some requirements that **MUST** be
met in order to avoid issues.

## MN4 Requirements

### dataClay:

In order to use PyCOMPSs with dataClay in MN4, it is necessary to add the
following lines to your ```.bashrc```:

```bash
module load gcc/8.1.0
 export COMPSS_PYTHON_VERSION=3-ML
 module load COMPSs/2.6
 module load mkl/2018.1
 module load impi/2018.1
 module load opencv/4.1.2
 module load python/3.6.4_ML
 module load DATACLAY/2.0rc
```

### Hecuba:

In order to use PyCOMPSs with Hecuba in MN4, it is necessary to add the
following lines to your ```.bashrc```:

```bash
module load gcc/8.1.0
 export COMPSS_PYTHON_VERSION=3-ML
 module load COMPSs/2.6
 module load mkl/2018.1
 module load impi/2018.1
 module load opencv/4.1.2
 module load python/3.6.4_ML
 module load hecuba/0.1.3_ML
```

### Redis

In order to use PyCOMPSs with Redis in MN4, it is necessary to install some
packages in your ```$HOME``` folder with the following commands:

```bash
 # From your local machine, download the following packages:
 #  - redis-3.3.0.gem................. http://rubygems.org/downloads/redis-3.3.0.gem
 #  - hiredis-1.0.1.tar.gz............ https://pypi.org/project/hiredis/1.0.1/
 #  - redis-3.0.1.tar.gz.............. https://pypi.org/project/redis/3.0.1/
 #  - redis-py-cluster-2.0.0.tar.gz... https://pypi.org/project/redis-py-cluster/2.0.0/
 # Copy them into your MN4 home folder
 # Log into your MN4 account and do the following commands
 module load ruby
 gem install -l redis-3.3.0.gem --user-install
 module load python/3.6.4_ML
 pip install hiredis-1.0.1.tar.gz --user --no-index
 pip install redis-3.0.1.tar.gz --user --no-index
 pip install redis-py-cluster-2.0.0.tar.gz --user --no-index
```

And add the following line to your ```.bashrc```:

```bash
export PATH=/apps/COMPSs/Storage/Redis/bin:$PATH
```

## Local Requirements

### Redis:

Redis must be installed locally, as well as its python packages
(redis and redis-py-cluster). Hiredis is *optional*, but it is recommended
since it boosts the redis performance.

Please, **take care with the versions**.
```bash
# Opensuse 15.0 System packages:
 # - Redis 4.0.10
 # - hiredis 0.13.3
 sudo zypper install redis hiredis

 # Python packages:
 #  - redis-py-cluster 2.0.0
 #  - redis 3.0.1
 # Choose the following depending on your target Python version
 python -m pip install 'redis==3.0.1' 'redis-py-cluster==2.0.0' --user
 python3 -m pip install 'redis==3.0.1' 'redis-py-cluster==2.0.0' --user
```

<!--
## TBD If updating Redis to v5:

COMPSs relies on the usage of ```redis-trib.rb``` script.

> WARNING: redis-trib.rb is not longer available since v5!
We should use redis-cli instead.

All commands and features belonging to redis-trib.rb have been moved to ```redis-cli```.
In order to use them you should call ```redis-cli``` with the ```--cluster```
option followed by the subcommand name, arguments and options.

Use the following syntax:
```bash
redis-cli --cluster SUBCOMMAND [ARGUMENTS] [OPTIONS]
```

Example:
```bash
redis-cli --cluster create 10.1.20.60:6379 10.1.20.61:6379 10.1.20.60:6380 --cluster-replicas 0
```

To get help about all subcommands, type:
```bash
redis-cli --cluster help
```
-->

## Issues

If any issue is found, please contact <support-compss@bsc.es>
