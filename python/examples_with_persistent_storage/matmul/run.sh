runcompss \
  -dg \
  --python_interpreter=python3 \
  --pythonpath=$(pwd)/src \
  src/matmul.py -b 4 -e 4 --check_result
