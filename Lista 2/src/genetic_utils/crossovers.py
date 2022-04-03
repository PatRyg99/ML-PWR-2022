import numpy as np
from numba import jit, prange

@jit(nopython=True, parallel=True)
def crossover_ordered(pop_parents: np.ndarray, prob: float):
    pop_children = np.zeros((pop_parents.shape[0], pop_parents.shape[-1]))

    for i in prange(pop_parents.shape[0]):
        p = np.random.rand(1)

        if p <= prob:
            indices = np.random.choice(pop_parents.shape[2], 2, replace=False)
            indices.sort()
            start, end = indices[0], indices[1]

            p1 = pop_parents[i, 0]
            p2 = pop_parents[i, 1]

            p1_genes = p1[start:end]
            p2_genes = np.zeros(p2.shape[-1] - (end - start))

            index = 0
            for j in range(p2.shape[-1]):
                present = False

                for k in range(p1_genes.shape[-1]):
                    if p2[j] == p1_genes[k]:
                        present = True
                        break
                
                if not present:
                    p2_genes[index] = p2[j]
                    index += 1

            child = np.zeros(p1.shape[-1])
            child[:start] = p2_genes[:start]
            child[start:end] = p1_genes
            child[end:] = p2_genes[start:]
            pop_children[i] = child

        else:
            pop_children[i] = pop_parents[i, 0]

    return pop_children