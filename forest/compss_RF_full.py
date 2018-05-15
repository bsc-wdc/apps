from compss_decision_tree import CompssDecisionTreeRegressor


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

    def fit(self, X, y):
        trees = []
        for i in range(self.n_estimators):
            trees.append(CompssDecisionTreeRegressor(max_features=self.max_features).fit())
        return self

    def predict(self, x):
        pass
