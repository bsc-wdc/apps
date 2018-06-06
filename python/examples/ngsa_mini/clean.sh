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

  # Application dependant files
  DIR0=initial
  DIR1=base
  DIR2=base_buckets
  WORK_DIR=work_compss

  rm -r $DIR0/$WORK_DIR/*
  rm -r $DIR1/$WORK_DIR/*
  rm -r $DIR2/$WORK_DIR/*

