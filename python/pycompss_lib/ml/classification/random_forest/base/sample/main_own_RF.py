import argparse
import time

from pycompss.api.api import compss_barrier

import os
import inspect
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(inspect.stack()[0][1])))+'/src')
from decision_tree import DecisionTree


def main():
    initial_time = time.time()
    parser = argparse.ArgumentParser(description='Own decision tree, for now.')

    # DecisionTree params
    parser.add_argument('--path_in', help='Path of the dataset directory.',)
    parser.add_argument('--n_instances', type=int, help='Number of instances in the sample.')
    parser.add_argument('--n_features', type=int, help='Number of attributes in the sample.')
    parser.add_argument('--max_depth', type=int, help='Depth of the decision tree.')
    parser.add_argument('--path_out', help='Path of the output directory.')
    parser.add_argument('--name', help='Name of the output file.')

    args = parser.parse_args()

    tree = DecisionTree(args.path_in, args.n_instances, args.n_features)
    tree.fit(args.max_depth, args.path_out, args.name)

    compss_barrier()

    final_time = time.time()

    print(args)
    print('Time: ' + str(final_time - initial_time))


if __name__ == "__main__":
    main()
