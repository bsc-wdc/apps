import sklearn

from ..src import forest
from ..utils import utils


def main():
    X_train, y_train, X_test, _ = utils.Dataset(prediction_type='class').read_and_wait_all()

    random_forests = []
    random_forests.append(forest.RandomForestClassifier(oob_score=True, random_state=0))
    random_forests.append(sklearn.ensemble.RandomForestClassifier(oob_score=True, random_state=0))

    compare_list = []
    for rf in random_forests:
        rf.fit(X_train, y_train)
        compare_list.append(rf.predict(X_test))
        # Can also test: apply(X_test), decision_path(X_test), feature_importances_,
        #                oob_decision_function_, oob_score_, predict_proba(X_test), predict_log_proba(X_test)

    different = False
    for res in compare_list:
        if not utils.are_equal(compare_list[0], res):
            different = True

    if different:
        print('RandomForestClassifiers yield different results')
    else:
        print('Test SUCCESS')


if __name__ == "__main__":
    main()
