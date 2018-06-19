import numpy as np
from pycompss.api.task import task


def fft(a):
    x = a.flatten()
    n = x.size

    # precompute twiddle factors
    w = np.zeros((n, n), dtype=complex)

    for i in range(n):
        for j in range(n):
            w[i, j] = np.exp(-2 * np.pi * 1j * j / (i + 1))

    lin = []

    for xk in x:
        lin.append(np.array(xk, ndmin=1, dtype=complex))

    while len(lin) > 1:
        lout = []
        ln = len(lin)

        for k in range(ln / 2):
            lout.append(reduce(lin[k], lin[k + ln / 2], w))

        lin = lout

    return lin[0]


@task(returns=1)
def reduce(e, o, w):
    x = np.concatenate((e, o))
    n = len(x)

    for k in range(n / 2):
        ek = x[k]
        ok = x[k + n / 2]
        wk = w[n - 1, k]

        x[k] = ek + wk * ok
        x[k + n / 2] = ek - wk * ok

    return x



