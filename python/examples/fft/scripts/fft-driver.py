from pycompss_lib.math.fft import fft
import numpy as np


def main():
    arr = np.random.rand(256)
    print(fft(arr))


if __name__ == "__main__":
    main()
