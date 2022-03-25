import numpy as np
from numba import jit, prange

@jit(nopython=True, parallel=True)
def mutation_swap(population: np.ndarray, prob: float = 0.5):

    for i in prange(population.shape[0]):
        idx = np.random.choice(population.shape[1], 2, replace=False)

        population[i, int(idx[0])], population[i, int(idx[1])] = (
            population[i, int(idx[1])], 
            population[i, int(idx[0])]
        )

    return population

@jit(nopython=True, parallel=True)
def mutation_inverse(population: np.ndarray, prob: float):

    for i in prange(population.shape[0]):
        p = np.random.rand(1)

        if p >= prob:
            indices = np.random.choice(population.shape[1], 2, replace=False)
            indices.sort()

            start = indices[0]
            end = indices[1]
            
            for j in prange((end - start + 1) // 2):
                population[i, start + j], population[i, end - j] = (
                    population[i, end - j], 
                    population[i, start + j]
                )

    return population