import numpy as np
from numba import jit, prange

@jit(nopython=True, parallel=True)
def random_search(num_genes: int, iterations: int):
    """
    Random algorithm for TSM problem.
    """
    solutions = np.zeros((iterations, num_genes))
    for i in prange(iterations):
        solutions[i] = np.random.permutation(num_genes)
        
    return solutions
