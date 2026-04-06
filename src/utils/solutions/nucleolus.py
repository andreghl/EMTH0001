import itertools
import numpy as np
from math import isclose
from scipy.optimize import linprog

# The code is taken from
# https://github.com/jbansil/Game-Theory---Nucleolus-Solver/blob/main/Game_Theory_Code.ipynb

def Nucleolus(N : list, v : dict, X : list):
    n = len(N)

    return 1

""" Usage Example

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
"""