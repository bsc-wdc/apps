class CompssDecisionTreeRegressor(object):
    def __init__(self,
                 max_features=None,
                 criterion="mse",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2):
        self.max_features = max_features
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.tree = None
        self.feature_importances = []

    def fit(self):
        self.tree = self.__get_split()
        self.__split(self.tree, 1)
        return self

    def predict(self, x):
        return self.min_samples_split*x

    def __get_split(self):
        pass

    def __split(self, node, depth):
        pass
