#!/bin/bash

  # Clean job files
  find . -type f -name 'compss-*.out' -delete
  find . -type f -name 'compss-*.err' -delete

  # Clean python compiled files
  find . -type f -name '*.pyc' -delete
  find . -type f -name '*.pyo' -delete

  # Clean trace files
  find . -type f -name '*tar.gz' -delete
