import numpy as np
from numba import jit, prange

@jit(nopython=True, parallel=True)
def selection_tournament(pop: np.ndarray, pop_fitness: np.ndarray, tour: int, parents: int = 2):

    pop_parents = np.zeros((pop.shape[0], parents, pop.shape[1]))

    for i in prange(pop_fitness.shape[0]):
        for j in prange(parents):
            subset_idx = np.random.choice(pop_fitness.shape[0], tour, replace=False)

            pop_subset = pop[subset_idx]
            pop_fitness_subset = pop_fitness[subset_idx]

            parent_idx = np.argmin(pop_fitness_subset)
            pop_parents[i, j] = pop_subset[int(parent_idx)]

    return pop_parents
