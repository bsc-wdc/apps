BASE_PATH="/home/bscuser/git"

#     --qos=debug \
#     --tracing=true \
runcompss \
    --pythonpath="${BASE_PATH}/apps/python/pycompss_lib/pycompss_lib/ml/classification/random_forest/full/src" \
    main_decision_tree.py \
        --path_in=/home/bscuser/datasets/dt_test_2/ \
        --n_instances=20 \
        --n_features=10 \
        --path_out= \
        --name_out=local_test_tree \
        --max_depth=3 \
