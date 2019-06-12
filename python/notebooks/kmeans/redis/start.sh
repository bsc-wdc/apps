# Set a standalone Redis backend
redis-server --daemonize yes

# Add Redis Storage API to CLASSPATH
for jarname in ${COMPSS_HOME}/Tools/storage/redis/*.jar; do
      cp=$cp:$jarname
done
export CLASSPATH="$cp:${CLASSPATH}"

# Add Redis Storage API to PYTHONPATH
export PYTHONPATH="$COMPSS_HOME/Tools/storage/redis/python/:$PYTHONPATH"

# Start jupyter-notebook
jupyter-notebook KMeans_redis.ipynb
