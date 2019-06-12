##############################
# KMeans with Redis Notebook #
##############################

Instructions :
==============

Usage:

./start.sh

* In order to use this notebook it is necessary to have a Redis server running.
  The start.sh script starts a Redis server and then starts the notebook.

* While the notebook is running, it is possible to check the Redis server objects:

       export PYTHONPATH="$COMPSS_HOME/Tools/storage/redis/python/:$PYTHONPATH"
       python check_redis.py

  It retrieves the keys of the existing objects and their objects.

* When the execution has finished, please use:

      ./stop.sh

  It stops the Redis server and cleans unnecessary files.
