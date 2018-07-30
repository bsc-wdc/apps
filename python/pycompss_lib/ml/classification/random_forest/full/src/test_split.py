from __future__ import division

import numpy as np
from sys import float_info


# Maximizing the Gini gain is equivalent to minimizing this proxy function
def gini_criteria_proxy(l_weight, l_length, r_weight, r_length, not_repeated):
    return -(l_weight / l_length + r_weight / r_length) * not_repeated


def test_split(sample, y_s, feature, n_classes):
    size = y_s.shape[0]
    if size == 0:
        return float_info.max, np.float64(np.inf)

    f = feature[sample]
    sort_indices = np.argsort(f)
    y_sorted = y_s[sort_indices]
    f_sorted = f[sort_indices]

    not_repeated = np.empty(size, dtype=np.bool_)  # type: np.ndarray
    not_repeated[0: size - 1] = (f_sorted[1:] != f_sorted[:-1])
    not_repeated[size - 1] = True

    l_frequencies = np.zeros((n_classes, size), dtype=np.int64)  # type: np.ndarray
    l_frequencies[y_sorted, np.arange(size)] = 1

    r_frequencies = np.zeros((n_classes, size), dtype=np.int64)
    r_frequencies[:, 1:] = l_frequencies[:, :0:-1]

    l_weight = np.sum(np.square(np.cumsum(l_frequencies, axis=-1)), axis=0)
    r_weight = np.sum(np.square(np.cumsum(r_frequencies, axis=-1)), axis=0)[::-1]

    l_length = np.arange(1, size + 1, dtype=np.int32)
    r_length = np.arange(size - 1, -1, -1, dtype=np.int32)  # type: np.ndarray
    r_length[size - 1] = 1  # Avoiding division by zero, the right score will be 0 anyways

    scores = gini_criteria_proxy(l_weight, l_length, r_weight, r_length, not_repeated)

    min_index = size - np.argmin(scores[::-1]) - 1

    if min_index + 1 == size:
        b_value = np.float64(np.inf)
    else:
        b_value = (f_sorted[min_index] + f_sorted[min_index + 1]) / 2
    return scores[min_index], b_value
