import numpy as np
import matplotlib.pyplot as plt
import random
import itertools

from sklearn.cluster import KMeans

def instance(size : tuple, plot : bool = False):
    instance = None
    depots = [(-0.2, 0.173), (0.2, 0.173), (0, -0.173)]
    radius = [0.3, 0.4, 0.6]
    N, D = size

    d = random.sample(depots, k = D)
    r = random.choice(radius)
    x, y = np.random.uniform(-r, r, (2, N)).astype(float)
    c = [(x, y) for x, y in zip(x, y)]

    instance = (size, d, r, c)

    if plot:
        plt.scatter(*zip(*c), c = "grey")
        plt.scatter(*zip(*d), c = "red")
        plt.title(label = f"Instance with radius = {r}")
        plt.show()

    return instance

def partition(instance):
    _, depots, _, customers = instance
    D = len(depots)

    kmeans = KMeans(n_clusters = D, init = depots, n_init = 1)
    depots = np.array(depots).astype(int)
    customers = np.array(customers).astype(float)
    kmeans.fit(customers)
    assign = kmeans.labels_
    dist, index = graph(instance)
    for i in range(len(assign)):
        W = sum(dist[index[tuple(customers[i])], d] for d in range(D))
        weights = [(1 - dist[index[tuple(customers[i])], d]) / W for d in range(D)]
        assign[i] = random.choices([0, 1, 2], weights = weights)[0]

    partition = {d: [] for d in range(D)}
    for i, d in enumerate(assign):
        partition[d].append(customers[i])

    return partition

def _partition(instance):
    _, depots, _, customers = instance
    
    # FIXME: partition is too perfect, risk no colab
    kmeans = KMeans(n_clusters = len(depots), init = depots, n_init = 1)
    depots = np.array(depots).astype(int)
    customers = np.array(customers).astype(float)
    kmeans.fit(customers)
    assign = kmeans.labels_
    partition = {d: [] for d in range(len(depots))}
    for i, d in enumerate(assign):
        partition[d].append(customers[i])

    return partition

def distance(x : np.ndarray, y : np.ndarray):
    return np.linalg.norm(x - y)

def graph(instance):
    size, d, r, c = instance
    N, D = size
    loc = np.array(d + c).astype(float)
    M = len(loc)
    graph = np.zeros((M, M))
    index = {tuple(loc[m]): m for m in range(M)}

    for i in range(M):
        for j in range(M):
            x = np.array(loc[i])
            y = np.array(loc[j])
            graph[i, j] = distance(x, y)
    
    return graph, index

def clarkeWright(instance):
    size, depots, r, c = instance
    N, D = size
    customers = partition(instance)
    dist, index = graph(instance)
    routes = {i: [] for i in range(D)}

    for d in range(D):
        
        savings = []
        cust = customers[d]
        depot = np.array(depots[d]).astype(float)
        r = {k: [x] for k, x in zip(range(len(cust)), cust)}

        for ci, cj in itertools.combinations(cust, 2):
                d = index[tuple(depot)]
                i = index[tuple(ci)]
                j = index[tuple(cj)]

                if i != j:
                    s = dist[d, i] + dist[d, j] - dist[i, j]
                    savings.append((ci, cj, s))
                    savings.append((cj, ci, s))

        savings.sort(key = lambda x: x[-1], reverse = True)

        for ci, cj, s in savings:
            i = j = None
            for k, v in r.items():
                if np.array_equal(v[-1], ci):
                    i = k
                if np.array_equal(v[0], cj):
                    j = k

        
            if i is not None and j is not None and i != j:
                new = r[i] + r[j]
                r[i] = new
                del r[j]

        routes[d] = np.array([depot] + list(*r.values()) + [depot])

    return routes

def costs(routes):
    """
    costs([routes]) if routes contains only one route.
    
    routes : list of routes
    """
    N = len(routes)
    i = 0
    costs = np.zeros(N)
    while i < N:
        
        r = routes[i]
        n = len(r)
        c = 0
        for k in range(1, n):
            c += distance(np.array(r[k - 1]), np.array(r[k]))

        costs[i] = c
        i += 1

    return costs

def initial(instance, routes):
    size, d, r, c = instance
    plt.scatter(*zip(*c), c = "grey")
    plt.scatter(*zip(*d), c = "red")
    plt.title(label = f"Instance with radius = {r}")
    colors = ["blue", "red", "green"]
    for i in range(len(routes)):
        plt.plot(routes[i][:, 0], routes[i][:, 1], color = colors[i])
    plt.show()

instance = instance((12, 3))
routes = clarkeWright(instance)
initial(instance, routes)