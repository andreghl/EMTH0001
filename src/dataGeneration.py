"""
{
VRP: matrix (Vx5),
A : matrix ()
coalitions: vector (K - N),
characteristic function: vector (K - N),
nucleolus: matrix (KxN),
shapley: matrix: (KxN)
}
"""

import h5py
import numpy as np

from tqdm import tqdm
from utils import generate

runs = 10000
N = 9
D = 3
K = 2 ** D

with h5py.File("data/instances.h5", "w") as f:
        
    Dm = f.create_dataset("Dm", shape = (runs, N + D, 5), dtype = np.float32)
    assign = f.create_dataset("assign", shape = (runs, N + D, K), dtype = np.int32)
    coalitions = f.create_dataset("coalitions", shape = (runs, K, D), dtype = np.float32)
    v = f.create_dataset("v", shape = (runs, K), dtype = np.float32)
    Sh = f.create_dataset("shapley", (runs, D), dtype = np.float32)
    n = f.create_dataset("nucleolus", (runs, D), dtype = np.float32)

    for i in tqdm(range(runs)):
        Dm[i], assign[i], coalitions[i], v[i], Sh[i], n[i] = generate(N, D)