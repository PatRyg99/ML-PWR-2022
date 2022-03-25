import numpy as np
import pandas as pd

from src.heuristics.heuristic import Heuristic
from src.genetic_utils.initialization import generate_distance_matrix
from src.genetic_utils.fitness_function import fitness_function

class RandomSearch(Heuristic):
    def __init__(self, num_genes: int, iterations: int):
        super().__init__(num_genes, iterations)
        
        self.history = []

    def on_end(self):
        self.history = pd.DataFrame(self.history, columns=["solution", "fitness"])

    def run_iteration(self, i: int):
        self.solution = np.random.permutation(self.distance_matrix.shape[0])
        self.solution_fitness = fitness_function(self.solution[None, :], self.distance_matrix)[0]

        self.log_history()

    def log_history(self):
        self.history.append((
            self.solution,
            self.solution_fitness
        ))