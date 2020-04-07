runcompss \
  -dg \
  --python_interpreter=python3 \
  src/matmul.py -b 4 -e 4 --check_result
