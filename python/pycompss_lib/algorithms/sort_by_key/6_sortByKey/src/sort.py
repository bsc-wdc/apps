#!/usr/bin/python
# -*- coding: utf-8 -*-
from pycompss.api.task import task


@task(returns=list)
def sort_partition_from_file(path):
    """ Sorts data, which is assumed to consists of (key, value) tuples list.
    :param path: file absolute path where the list of tuples to be sorted is located.
    :return: sorted list of tuples.
    """
    import pickle
    f = open(path, 'r')
    data = pickle.load(f)
    f.close()
    res = sorted(data, key=lambda tuple: tuple[0], reverse=False)
    return res


@task(returns=list)
def sort_partition(data):
    """ Sorts data, which is assumed to consists of (key, value) pairs list.
    :param data: list of tuples to be sorted by key.
    :return: sorted list of tuples.
    """
    res = sorted(data, key=lambda tuple: tuple[0], reverse=False)
    return res


@task(returns=list, priority=True)
def reduce_task(a, b):
    """ Reduce list a and list b.
        They must be already sorted.
    :param a: list.
    :param b: list.
    :return: result of merging a and b lists.
    """
    res = []
    i = 0
    j = 0
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            res.append(a[i])
            i += 1
        else:
            res.append(b[j])
            j += 1
    # Append the remaining tuples
    if i < len(a):
        res += a[i:]
    elif j < len(b):
        res += b[j:]
    return res


def merge_reduce(f, data):
    """ Apply function cumulatively to the items of data,
        from left to right in binary tree structure, so as to
        reduce the data to a single value.
    :param f: function to apply to reduce data.
    :param data: List of items to be reduced.
    :return: result of reduce the data to a single value.
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


@task(returns=list)
def generate_fragment(num_keys, unique_keys, key_length, unique_values, value_length, random_seed, hash_function):
    """ Generate a fragment.
        Each fragment is list of pairs (K, V) generated randomly.
    :param num_keys: number of keys per fragment.
    :param unique_keys: number of unique keys.
    :param key_length: length of each key.
    :param unique_values: number of unique values.
    :param value_length: length of each value.
    :param random_seed: Random seed.
    :param hash_function: Boolean - use hash function.
    :return: fragment (list of tuples).
    """
    ints = generate_int_data(num_keys, unique_keys, unique_values, random_seed)
    data = [(padded_string(k_v[0], key_length, hash_function), padded_string(k_v[1], value_length, hash_function)) for k_v in ints]
    return data


def generate_int_data(num_keys, unique_keys, unique_values, random_seed):
    """ Generate a list of (int, int) tuples with random values.
    :param num_keys: number of keys per fragment.
    :param unique_keys: number of unique keys.
    :param unique_values: number of unique values.
    :param random_seed: Random seed.
    :return: fragment (list of tuples).
    """
    import random
    random.seed(random_seed)
    data = [(random.randint(0, unique_keys), random.randint(0, unique_values)) for _ in range(num_keys)]
    return data


def padded_string(i, length, hash_function):
    """ Converts a int to String with determined length.
    :param i: input integer.
    :param length: length (number of characters).
    :param hash_function: Boolean - use hash function.
    :return: i as String.
    """
    fmtString = "{:0>" + str(length) + "d}"
    if hash_function:
        out = hash(i)
        if len(str(out)) < length:
            return fmtString.format(out)
        else:
            return out
    else:
        return fmtString.format(i)


def main():
    import sys
    import os
    import time

    num_keys = int(sys.argv[1])
    unique_keys = int(sys.argv[2])
    key_length = int(sys.argv[3])     # number of characters
    unique_values = int(sys.argv[4])
    value_length = int(sys.argv[5])   # number of characters
    num_fragments = int(sys.argv[6])
    keys_per_fragment = num_keys / num_fragments
    random_seed = int(sys.argv[7])
    from_files = sys.argv[8]
    path = "Not used - Autogenerating dataset."
    if from_files == "true":
        # Ignore all parameters but 'path'
        # Each file represents a fragment
        # Each file has to contain a pickable dictionary {K,V} with the desired lengths and values.
        path = sys.argv[9]

    print("Sort by Key [(K,V)]:")
    print("Num keys: %d" % num_keys)
    print("Unique keys: %d" % unique_keys)
    print("Key length: %d" % key_length)
    print("Unique values: %d" % unique_values)
    print("Value length: %d" % value_length)
    print("Num fragments: %d" % num_fragments)
    print("Keys per fragment: %d" % keys_per_fragment)
    print("Random seed: %d" % random_seed)
    print("From files: %r" % from_files)
    print("Path: %s" % path)

    start_time = time.time()
    from pycompss.api.api import compss_wait_on

    partial_sorted = []
    result = []
    if from_files == "true":
        # Get Dataset from files (read within sort_partition_from_file task)
        files = []
        for file in os.listdir(path):
            files.append(path+'/'+file)
        partial_sorted = list(map(sort_partition_from_file, files))
        result = merge_reduce(reduce_task, partial_sorted)
    else:
        # Autogenerate dataset
        for i in range(num_fragments):
            fragment = generate_fragment(keys_per_fragment, unique_keys, key_length, unique_values, value_length, random_seed, True)
            partial_sorted.append(sort_partition(fragment))
            random_seed += i
        result = merge_reduce(reduce_task, partial_sorted)

    result = compss_wait_on(result)

    print("Elapsed Time(s)")
    print(time.time() - start_time)
    print("Sorted by Key elements: %d" % len(result))
    # print result


if __name__ == "__main__":
    main()
