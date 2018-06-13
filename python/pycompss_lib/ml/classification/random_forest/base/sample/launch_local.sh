#     --qos=debug \
#     --tracing=true \

BASE_PATH="/home/bscuser/git"

runcompss \
    --pythonpath="${BASE_PATH}/apps/python/pycompss_lib/ml/classification/random_forest/base/src" \
    ./main_dense.py \
        --regr=False \
        --path="${BASE_PATH}/apps/datasets/random_forest" \
        --name=1e5s1e2f3c \
        --sklearn=False \
        --n_estimators=96 \

# see sklearn docs for RandomForest arguments