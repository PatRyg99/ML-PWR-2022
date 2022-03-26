import numpy as np
import pandas as pd

from src.heuristics.heuristic import Heuristic
from src.genetic_utils.random_search import random_search
from src.genetic_utils.fitness_function import fitness_function

class RandomSearch(Heuristic):
    def __init__(self, num_genes: int, iterations: int):
        super().__init__(num_genes, 1)
        self.iterations = iterations
        self.history = []

    def on_end(self):
        self.history = [(solution, fitness) for solution, fitness in zip(self.solutions, self.solutions_fitness)]
        self.history = pd.DataFrame(self.history, columns=["solution", "fitness"])

    def run_iteration(self, i: int):
        self.solutions = random_search(self.num_genes, self.iterations)
        self.solutions_fitness = fitness_function(self.solutions, self.distance_matrix)
