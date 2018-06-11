import argparse
import time

from pycompss.api.api import compss_barrier
from forest import RandomForestClassifier


def main():
    initial_time = time.time()
    parser = argparse.ArgumentParser(description='Random forest classifier.')

    # DecisionTree params
    parser.add_argument('--path_in', help='Path of the dataset directory.',)
    parser.add_argument('--n_instances', type=int, help='Number of instances in the sample.')
    parser.add_argument('--n_features', type=int, help='Number of attributes in the sample.')
    parser.add_argument('--path_out', help='Path of the output directory.')
    parser.add_argument('--n_estimators', type=int, help='Number of trees to build.')
    parser.add_argument('--max_depth', type=int, help='Depth of the decision tree.')

    args = parser.parse_args()

    forest = RandomForestClassifier(args.path_in, args.n_instances, args.n_features,
                                    args.path_out, args.n_estimators, args.max_depth)
    forest.fit()

    compss_barrier()

    final_time = time.time()

    print(args)
    print('Time: ' + str(final_time - initial_time))


if __name__ == "__main__":
    main()
