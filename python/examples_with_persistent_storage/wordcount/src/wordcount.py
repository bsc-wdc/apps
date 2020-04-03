import os
from pycompss.api.task import task
from pycompss.api.parameter import *
from collections import defaultdict
from classes.block import Words


@task(returns=Words, file=FILE_IN, priority=True)
def populate_psco(file_path):
    """
    Perform a wordcount of a file.
    :param file_path: Absolute path of the file to process.
    :return: dictionary with the appearance of each word.
    """
    fp = open(file_path)
    data = fp.read()
    fp.close()
    psco = Words(text=data)
    psco.make_persistent()
    return psco


@task(returns=dict, priority=True)
def wordcount(block):
    """
    Wordcount over a psco block object.
    :param block: Block with text to perform word counting.
    :return: dictionary with the words and the number of appearances.
    """
    data = block.get_text().split()
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


def wordcount_pscos(pscos):
    """
    A Wordcount from pscos list algorithm.
    Given a set of pscos, the algorithm checks the number of appearances of
    each word.
    :param pscos: List of pscos
    :return: Final word count
    """
    partial_result = []
    for psco in pscos:
        partial_result.append(wordcount(psco))
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
                      description='A COMPSs-Redis Wordcount implementation.')
    parser.add_argument('-d', '--dataset_path', type=str,
                        help='Dataset path')
    return parser.parse_args()


def main(dataset_path):
    """
    This will be executed if called as main script. Look @ wordcount for the
    Wordcount function.
    This code is used for experimental purposes.
    :param dataset_path: Dataset path
    :return: None
    """
    from pycompss.api.api import compss_wait_on
    import time

    start_time = time.time()

    # We populate the storage with the file contents.
    pscos = []
    for fileName in os.listdir(dataset_path):
        pscos.append(populate_psco(os.path.join(dataset_path, fileName)))

    population_time = time.time()

    # Run Wordcount
    result = wordcount_pscos(pscos)

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
