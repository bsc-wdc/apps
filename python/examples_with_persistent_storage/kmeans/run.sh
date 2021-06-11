runcompss \
  --log_level=debug \
  --pythonpath=$(pwd)/src \
  --python_interpreter=python3 \
  src/kmeans.py -n 1024 -f 4 -d 2 -c 4 -i 4
