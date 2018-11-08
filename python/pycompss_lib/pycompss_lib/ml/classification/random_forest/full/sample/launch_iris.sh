BASE_PATH="/home/bscuser/git"

#     --qos=debug \
#     --tracing=true \
runcompss \
    --jvm_master_opts="-Dcom.sun.management.jmxremote.port=3333,-Dcom.sun.management.jmxremote.ssl=false,-Dcom.sun.management.jmxremote.authenticate=false" \
    --pythonpath="${BASE_PATH}/apps/python/pycompss_lib/pycompss_lib/ml/classification/random_forest/full/src" \
    -g iris.py\
        --n_estimators=1 \
        --distr_depth=2 \
        --try_features=3 \
