import numpy as np
from numba import jit, prange

@jit(nopython=True, parallel=True)
def greedy_search(input_distance_matrix: np.ndarray):
    """
    Greedy algorithm for TSM problem.
    """
    solutions = np.zeros((input_distance_matrix.shape[0], input_distance_matrix.shape[0]))
    solutions_fitness = np.zeros(input_distance_matrix.shape[0])

    for i in prange(input_distance_matrix.shape[0]):
        distance_matrix = np.copy(input_distance_matrix)

        solution = np.zeros(distance_matrix.shape[0])
        solution[0] = i

        solution_fitness = 0

        distances_mask = np.zeros(distance_matrix.shape[0])

        for j in range(distance_matrix.shape[0] - 1):
            distances_mask[int(solution[j])] = np.inf
            current_distances = distance_matrix[int(solution[j])]

            current_distances += distances_mask

            next_station = np.argmin(current_distances)
            minimal_distance = np.min(current_distances)

            solution[j + 1] = next_station
            solution_fitness += minimal_distance

        solutions[i] = solution
        solutions_fitness[i] = solution_fitness

    return solutions, solutions_fitness
