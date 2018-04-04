#!/bin/bash

  # Clean job files
  find . -type f -name 'compss-*.out' -delete
  find . -type f -name 'compss-*.err' -delete

  # Clean temp files
  find . -type f -name '*.sh~' -delete
  find . -type f -name '*.py~' -delete

  # Clean python compiled files
  find . -type f -name '*.pyc' -delete

  # Clean trace files
  find . -type f -name '*tar.gz' -delete

  #Clean specific files
  find . -type f -name 'A*' | grep '1_matmul_files' | xargs rm -rf {}
  find . -type f -name 'B*' | grep '1_matmul_files' | xargs rm -rf {}
  find . -type f -name 'C*' | grep '1_matmul_files' | xargs rm -rf {}
  find . -type d -name 'results' -exec rm -rf "{}" \;
