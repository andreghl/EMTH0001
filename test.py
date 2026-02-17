from itertools import combinations
import numpy as np
import random

import networkx as nx

depots = [(-0.2, 0.173), (0.2, 0.173), (0, -0.173)]
radius = [0.3, 0.4, 0.6]
N = 12 - len(depots)


null = (0.0, 0.0)
rad = random.choice(radius)

x = np.random.uniform(-rad, rad, N)
y = np.random.uniform(-rad, rad, N)
L = [np.array([x, y]) for x, y in zip(x, y)]

for c in combinations(L, 2):
    a, b = c
    distance = np.linalg.norm(a -b)
    print(f"{(a, b)}: {distance}")

n = N + len(depots)
G = nx.Graph()
G.add_edges_from(L)
