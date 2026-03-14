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

""" Example

N = [1, 2, 3]
v = {
    frozenset(): 0,
    frozenset([1]): 0,
    frozenset([2]): 0,
    frozenset([3]): 0,
    frozenset([1, 2]): 18,
    frozenset([1, 3]): 14,
    frozenset([2, 3]): 4,
    frozenset([1, 2, 3]): 30
}

print(Shapley(N, v))
> {1: 14.0, 2: 9.0, 3: 7.0}
"""