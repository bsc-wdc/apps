import argparse
import time

from pycompss.api.api import compss_barrier
from decision_tree import DecisionTreeClassifier


def main():
    initial_time = time.time()
    parser = argparse.ArgumentParser(description='Decision tree classifier.')

    # DecisionTree params
    parser.add_argument('--path_in', help='Path of the dataset directory.',)
    parser.add_argument('--n_instances', type=int, help='Number of instances in the sample.')
    parser.add_argument('--n_features', type=int, help='Number of attributes in the sample.')
    parser.add_argument('--path_out', help='Path of the output directory.')
    parser.add_argument('--name_out', help='Name of the output file.')
    parser.add_argument('--max_depth', type=int, help='Depth of the decision tree.')

    args = parser.parse_args()

    tree = DecisionTreeClassifier(args.path_in, args.n_instances, args.n_features, args.path_out, args.name_out, args.max_depth)
    tree.fit()

    compss_barrier()

    final_time = time.time()

    print(args)
    print('Time: ' + str(final_time - initial_time))


if __name__ == "__main__":
    main()
