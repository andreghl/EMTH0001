import itertools
import numpy as np
from math import isclose
from scipy.optimize import linprog

# The code is taken from
# https://github.com/jbansil/Game-Theory---Nucleolus-Solver/blob/main/Game_Theory_Code.ipynb

def powerset(N):
    """Generate all non‑empty proper subsets."""
    for r in range(1, len(N)):
        for combo in itertools.combinations(N, r):
            yield frozenset(combo)

def least_core_lp_dual(v, N, tight_eqs):
    """Solve least‑core LP, return allocation, eps, coalition list, duals."""
    n = len(N)
    # Variable bounds: x_i >= v({i}); eps free
    bounds = [(v[frozenset([p])], None) for p in N] + [(None, None)]
    # Build A_ub for non‑tight coalitions
    A_ub = []
    b_ub = []
    S_list = []
    for S in powerset(N):
        if S in tight_eqs:
            continue
        row = [ -1 if p in S else 0 for p in N] + [-1]
        A_ub.append(row)
        b_ub.append(-v[S])
        S_list.append(S)
    # Build A_eq for efficiency and tight coalitions
    A_eq = []
    b_eq = []
    # Efficiency: sum x_i = v(N)
    A_eq.append([1]*n + [0])
    b_eq.append(v[frozenset(N)])
    # Tight coalition equalities
    for S, delta in tight_eqs.items():
        row = [1 if p in S else 0 for p in N] + [0]
        A_eq.append(row)
        b_eq.append(v[S] - delta)
    # Objective: minimize eps
    c = [0]*n + [1]
    # Solve
    res = linprog(c,
                  A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                  bounds=bounds,
                  method='highs')
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    x = res.x
    eps = x[n]
    return {N[i]: x[i] for i in range(n)}, eps, S_list, res.ineqlin.marginals

def nucleolus(N, v, tol = 1e-8, verbose = False):
    tight_eqs = {}
    prev_alloc = None
    stage = 0
    while True:
        stage += 1
        alloc, eps, S_list, duals = least_core_lp_dual(v, N, tight_eqs)
        if verbose:
            print(f"Stage {stage}: eps = {eps:.6f}, alloc = {alloc}")
        # Stop when allocation stabilizes
        if prev_alloc is not None and all(isclose(alloc[p], prev_alloc[p], abs_tol=tol) for p in N):
            return alloc
        prev_alloc = alloc.copy()
        # Identify newly binding coalitions
        new_sets = [S_list[i] for i, m in enumerate(duals) if abs(m) > tol]
        for S in new_sets:
            tight_eqs[S] = eps

"""# Usage Example

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

print(nucleolus(N, v))
"""