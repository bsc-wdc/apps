import argparse
from sklearn.tree import DecisionTreeRegressor


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS,
                                     description='Predict regression value using a decision tree regressor.')
    parser.add_argument('--criterion', help="The function to measure the quality of a split.")
    parser.add_argument('--splitter', help="The strategy used to choose the split at each node.")
    parser.add_argument('--max_depth', type=int, help="The maximum depth of the tree.")
    parser.add_argument('--min_samples_split', type=float,
                        help="The minimum number of samples required to split an internal node.")
    parser.add_argument('--max_features',
                        help="The number of features to consider when looking for the best split")
    args = parser.parse_args()

    rf = DecisionTreeRegressor(**vars(args))
    rf.fit([[0], [1], [2], [3], [4], [5]], [0, 0, 1, 1, 0, 0])
    result = rf.predict([[0.5], [1.5], [2.5], [3.5], [4.5]])
    print(result)


if __name__ == "__main__":
    main()
