import numpy as np
import pandas as pd

from src.heuristics.heuristic import Heuristic
from src.genetic_utils.initialization import generate_distance_matrix
from src.genetic_utils.greedy_search import greedy_search

class GreedySearch(Heuristic):
    def __init__(self, num_genes: int):
        super().__init__(num_genes, 1)
        
        self.history = []

    def on_end(self):
        self.history = [(solution, fitness) for solution, fitness in zip(self.solutions, self.solutions_fitness)]
        self.history = pd.DataFrame(self.history, columns=["solution", "fitness"])

    def run_iteration(self, i: int):
        self.solutions, self.solutions_fitness = greedy_search(self.distance_matrix)