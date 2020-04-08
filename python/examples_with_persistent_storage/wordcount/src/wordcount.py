import os
import time
from collections import defaultdict

from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.api.api import compss_wait_on


@task(returns=1, file=FILE_IN, priority=True)
def populate_block(file_path, use_storage):
    """
    Perform a wordcount of a file.
    :param file_path: Absolute path of the file to process.
    :param use_storage: Use a persistent object.
    :return: dictionary with the appearance of each word.
    """
    if use_storage:
        from storage_model.block import Words
    else:
        from model.block import Words
    fp = open(file_path)
    data = fp.read()
    fp.close()
    psco = Words()
    psco.text = data
    if use_storage:
        psco.make_persistent(os.path.basename(file_path))
    return psco


@task(returns=defaultdict, priority=True)
def wordcount(block):
    """
    Wordcount over a Words object.
    :param block: Block with text to perform word counting.
    :return: dictionary with the words and the number of appearances.
    """
    data = block.text.split()
    result = defaultdict(int)
    for word in data:
        result[word] += 1
    return result


@task(returns=dict)
def reduce(dic1, dic2):
    """
    Reduce dictionaries a and b.
    :param a: dictionary.
    :param b: dictionary.
    :return: dictionary result of merging a and b.
    """
    for k, v in dic1.items():
        dic2[k] += v
    return dic2


def merge_reduce(f, data):
    """
    Apply function cumulatively to the items of data,
    from left to right in binary tree structure, so as to
    reduce the data to a single value.
    :param f: function to apply to reduce data
    :param data: List of items to be reduced
    :return: result of reduce the data to a single value
    """
    from collections import deque
    q = deque(range(len(data)))
    while len(q):
        x = q.popleft()
        if len(q):
            y = q.popleft()
            data[x] = f(data[x], data[y])
            q.append(x)
        else:
            return data[x]


def word_count(blocks):
    """
    A Wordcount from Words list algorithm.
    Given a set of Words blocks, the algorithm checks the number of appearances
    of each word.
    :param blocks: List of blocks
    :return: Final word count
    """
    partial_result = []
    for b in blocks:
        partial_result.append(wordcount(b))
    result = merge_reduce(reduce, partial_result)
    return result


def parse_arguments():
    """
    Parse command line arguments. Make the program generate
    a help message in case of wrong usage.
    :return: Parsed arguments
    """
    import argparse
    parser = argparse.ArgumentParser(
                      description='Wordcount application.')
    parser.add_argument('-d', '--dataset_path', type=str,
                        help='Dataset path')
    parser.add_argument('--use_storage', action='store_true',
                        help='Use storage?')
    return parser.parse_args()


def main(dataset_path, use_storage):
    """
    This will be executed if called as main script. Look @ wordcount for the
    Wordcount function.
    This code is used for experimental purposes.
    :param dataset_path: Dataset path
    :return: None
    """
    start_time = time.time()

    # We populate the storage with the file contents.
    blocks = []
    for fileName in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, fileName)
        blocks.append(populate_block(file_path,
                                     use_storage))

    population_time = time.time()

    # Run Wordcount
    result = word_count(blocks)

    result = compss_wait_on(result)
    wordcount_time = time.time()

    print("-----------------------------------------")
    print("-------------- RESULTS ------------------")
    print("-----------------------------------------")
    print("Population time: %f" % (population_time - start_time))
    print("Wordcount time: %f" % (wordcount_time - population_time))
    print("Total time: %f" % (wordcount_time - start_time))
    print("-----------------------------------------")
    print("Different words: %d" % (len(result)))
    print("-----------------------------------------")
    import pprint
    pprint.pprint(result)
    print("-----------------------------------------")


if __name__ == "__main__":
    options = parse_arguments()
    main(**vars(options))
