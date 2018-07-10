BASE_PATH="/home/bscuser/git"

#     --qos=debug \
#     --tracing=true \
runcompss \
    --pythonpath="${BASE_PATH}/apps/python/pycompss_lib/ml/classification/random_forest/full/src" \
    main_random_forest.py\
        --path_in=/home/bscuser/datasets/dt_test_4/ \
        --n_instances=20000 \
        --n_features=25 \
        --path_out= \
        --n_estimators=10 \
