#!/bin/bash -e

  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  runcompss \
    -d \
    \
	-g \
	-t \
	--summary \
	 \
     --lang=python \
     --pythonpath="${SCRIPT_DIR}"/src/ \
     \
	./src/conway.py 32 32 40 16 9 
