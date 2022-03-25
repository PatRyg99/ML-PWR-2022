import numpy as np
from numba import jit, prange

@jit(nopython=True, parallel=True)
def fitness_function(population, distance_matrix):

    distances = np.zeros(population.shape[0])

    for i in prange(population.shape[0]):
        for j in prange(population.shape[1] - 1):
            s1 = int(population[i, j])
            s2 = int(population[i, j + 1])
            
            distances[i] += distance_matrix[s1, s2]
        
    return distances