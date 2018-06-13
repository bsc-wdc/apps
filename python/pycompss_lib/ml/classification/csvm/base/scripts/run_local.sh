BASE_PATH="/home/user"

runcompss \
    --pythonpath="${BASE_PATH}/apps/python/pycompss_lib/ml/classification/csvm/base/src" \
    csvm-driver.py \
        -i 5 \
        --convergence \
        "${BASE_PATH}/apps/python/pycompss_lib/ml/classification/csvm/base/data/agaricus.csv" \