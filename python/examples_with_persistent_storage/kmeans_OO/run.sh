runcompss \
  -dg \
  --python_interpreter=python3 \
  --pythonpath=$(pwd)/src \
  src/kmeans.py -n 1024 -f 8 -d 2 -c 4
