from compss_decision_tree import Node


def bootstrap_sample():
    pass


class FullCompssRandomForestRegressor:
    def __init__(self,
                 n_estimators=10,
                 max_features=None,
                 criterion="mse",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.estimators_ = []

    def fit(self):

        for i in range(self.n_estimators):
            root = Node()
            sample = bootstrap_sample()
            root.compute_split(sample, kwargs)
            root.build_subnodes()
            self.estimators_.append(root)
        return self

    def predict(self):
        pass
