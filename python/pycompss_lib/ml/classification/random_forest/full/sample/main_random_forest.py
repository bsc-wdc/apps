import argparse
import time

from pycompss.api.api import compss_barrier, compss_open
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
    parser.add_argument('--max_depth', type=int, default=None, help='Depth of the decision tree.')
    parser.add_argument('--distr_depth', type=int, default=None, help='Tasks are distributed up to this depth.')

    args = parser.parse_args()

    forest = RandomForestClassifier(args.path_in, args.n_instances, args.n_features,
                                    args.path_out, args.n_estimators, args.max_depth, args.distr_depth)
    forest.fit()

    compss_barrier()

    final_time = time.time()

    print(args)
    print('Time: ' + str(final_time - initial_time))

    for i in range(args.n_estimators):
        with compss_open(args.path_out+'tree_' + str(i)):
            pass


if __name__ == "__main__":
    main()
