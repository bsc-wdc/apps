import sklearn

from forest import rf_sklearn_trees
import utils


def main():
    X_train, y_train, X_test, _ = utils.Dataset(prediction_type='regr').read_and_wait_all()

    random_forests = []
    random_forests.append(rf_sklearn_trees.RandomForestRegressor(oob_score=True, random_state=0))
    random_forests.append(sklearn.ensemble.RandomForestRegressor(oob_score=True, random_state=0))

    compare_list = []
    for rf in random_forests:
        rf.fit(X_train, y_train)
        compare_list.append(rf.predict(X_test))
        # Can also test: apply(X_test), decision_path(X_test), feature_importances_,
        #                oob_prediction_, oob_score_

    different = False
    for i in range(len(compare_list)):
        if not utils.are_equal(compare_list[0], compare_list[i]):
            different = True

    if different:
        print('RandomForestRegressors yield different results')
    else:
        print('Test SUCCESS')


if __name__ == "__main__":
    main()
