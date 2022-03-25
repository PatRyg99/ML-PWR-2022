import numpy as np
from numba import jit, prange

def generate_distance_matrix(dim: int):
    distance_matrix = np.random.beta(0.2, 0.2, size=(dim, dim))
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = np.tril(distance_matrix) + np.tril(distance_matrix, -1).T

    return distance_matrix


@jit(nopython=True, parallel=True)
def init_population(dim: int, pop_size: int):
    
    permutations = np.zeros((pop_size, dim))
    for i in prange(pop_size): 
        permutations[i] = np.random.permutation(dim)

    return permutations