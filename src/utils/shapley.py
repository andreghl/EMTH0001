import math
from collections import defaultdict

def Shapley(N : list, v : dict) -> dict:
    Shapley = defaultdict(int)
    n = len(N)

    for i in N:
        for S in v:
            if i not in S:
                s = len(S)

                w = math.factorial(s) * math.factorial(n - s - 1)
                w = w / math.factorial(n)

                Shapley[i] += w * (v[S | {i}] - v[S])
    
    return dict(Shapley)