import argparse

from forest.compss_decision_tree import DecisionTree


def main():
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


if __name__ == "__main__":
    main()
