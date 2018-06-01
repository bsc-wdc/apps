#!/usr/bin/python
# -*- coding: utf-8 -*-


def generate_int_data(num_records, unique_keys, unique_values, num_partitions, random_seed):
    records_per_partition = int((num_records / float(num_partitions)))

    def generate_partition(index):
        import random
        effective_seed = int(random_seed) ** index  # .toString.hashCode
        random.seed(effective_seed)
        data = [(random.randint(0, unique_keys), random.randint(0, unique_values))
                for i in range(records_per_partition)]
        return data

    return [generate_partition(i) for i in range(num_partitions)]


def generate_string_data(num_records, unique_keys, key_length, unique_values, value_length, num_partitions, random_seed, storage_location, hash_function):
    import pickle
    ints = generate_int_data(num_records, unique_keys, unique_values, num_partitions, random_seed)

    data = []
    for i in range(num_partitions):
        data.append([(padded_string(k_v[0], key_length, hash_function), padded_string(k_v[1], value_length, hash_function)) for k_v in ints[i]])

    ff = open(storage_location, 'w')
    pickle.dump(data, ff)


def padded_string(i, length, hash_function):
    fmt_string = "{:0>" + str(length) + "d}"
    if hash_function:
        out = hash(i)
        if len(str(out)) < length:
            return fmt_string.format(out)
        else:
            return out
    else:
        return fmt_string.format(i)


def chunks(l, n, balanced=False):
    if not balanced or not len(l) % n:
        for i in range(0, len(l), n):
            yield l[i:i + n]
    else:
        rest = len(l) % n
        start = 0
        while rest:
            yield l[start: start + n + 1]
            rest -= 1
            start += n + 1
        for i in range(start, len(l), n):
            yield l[i:i + n]


def partition_by(self, num_partitions, hash):
    if not hash:
        return chunks(self, len(self) / num_partitions, True)
    else:
        partitions = [[] for n in range(num_partitions)]
        for s in self:
            partitions[hashed_partitioner(s, num_partitions)].append(s)
        return partitions


def hashed_partitioner(k, num_partitions, key_func=lambda x: x):
    return hash(key_func(k)) % num_partitions


def keyfunc(x):
    return x


def main():
    # Tests
    # ints = generate_int_data(10, 10, 10, 2, 5)
    # print ints
    # out_path = "/tmp/out.dataset"
    # generate_string_data(10, 10, 5, 10, 5, 2, 8, out_path, False)

    import sys

    num_records = int(sys.argv[1])
    unique_keys = int(sys.argv[2])
    key_length = int(sys.argv[3])
    unique_values = int(sys.argv[4])
    value_length = int(sys.argv[5])
    num_partitions = int(sys.argv[6])
    random_seed = int(sys.argv[7])
    storage_location = sys.argv[8]
    hash_function = bool(sys.argv[9])

    generate_string_data(num_records,
                         unique_keys,
                         key_length,
                         unique_values,
                         value_length,
                         num_partitions,
                         random_seed,
                         storage_location,
                         hash_function)


if __name__ == "__main__":
    main()
