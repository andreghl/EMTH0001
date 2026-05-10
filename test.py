from src.utils import Shapley, nucleolus

N = [1, 2, 3]

v = {frozenset(): 0,
     frozenset([1]): 0,
     frozenset([2]): 0,
     frozenset([3]): 0,
     frozenset([1, 2]): 18,
     frozenset([1, 3]): 14,
     frozenset([2, 3]): 4,
     frozenset([1, 2, 3]): 30}

print(Shapley(N, v))
print(nucleolus(N, v))