import numpy as np
import matplotlib.pyplot as plt

import random

depots = [(-0.2, 0.173), (0.2, 0.173), (0, -0.173)]
radius = [0.3, 0.4, 0.6]
N = 12 - len(depots)


null = (0.0, 0.0)
rad = random.choice(radius)

x = np.random.uniform(-rad, rad, N)
y = np.random.uniform(-rad, rad, N)

plt.scatter(x, y, c = "grey")
plt.scatter(*zip(*depots), c = "red")
plt.title(label = f"Instance with radius = {rad}")
plt.show()
