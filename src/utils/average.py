import numpy as np

def ma(x, size = 1):
    w = np.ones(size) / size
    return np.convolve(x, w, mode = 'same')