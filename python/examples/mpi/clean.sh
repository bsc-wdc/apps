#!/bin/bash

  # Clean job files
  find . -type f -name 'compss-*.out' -delete
  find . -type f -name 'compss-*.err' -delete
  find . -type d -name 'output' | xargs rm -rf

  # Clean temp files
  find . -type f -name '*.sh~' -delete
  find . -type f -name '*.py~' -delete

  # Clean python compiled files
  find . -type f -name '*.pyc' -delete
  find . -type f -name '*.pyo' -delete
  find . -type d -name '__pycache__' | xargs rm -rf
  find . -type f -name '*.x' -delete

  # Clean trace files
  find . -type f -name '*tar.gz' -delete
