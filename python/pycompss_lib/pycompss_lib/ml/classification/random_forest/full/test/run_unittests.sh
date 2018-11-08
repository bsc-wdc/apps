#!/usr/bin/env bash
BASE_PATH="/home/bscuser/git"

#     --qos=debug \
#     --tracing=true \
PYTHONPATH=$PYTHONPATH:"${BASE_PATH}/apps/python/pycompss_lib/pycompss_lib/ml/classification/random_forest/full/src" \

python -m unittest test_test_split.TestTestSplit test_compare_sklearn.TestCompareSKLearn
