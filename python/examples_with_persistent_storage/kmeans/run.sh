runcompss \
  -dg \
  --pythonpath=$(pwd)/src \
  --python_interpreter=python3 \
  src/kmeans.py -n 1024 -f 8 -d 2 -c 4
