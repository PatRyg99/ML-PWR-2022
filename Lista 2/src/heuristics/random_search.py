import numpy as np
import pandas as pd

from src.problems.problem import Problem
from src.heuristics.heuristic import Heuristic
from src.genetic_utils.random_search import random_search
from src.genetic_utils.fitness_function import fitness_function

class RandomSearch(Heuristic):
    def __init__(self, problem: Problem, iterations: int):
        super().__init__(problem, 1)
        self.iterations = iterations  

    def on_end(self, verbose: bool):
        self.history = [(solution, fitness) for solution, fitness in zip(self.solutions, self.solutions_fitness)]
        self.history = pd.DataFrame(self.history, columns=["solution", "fitness"])

        if verbose:
            best_fitness = np.min(self.history["fitness"])
            print(f"Best fitness: {best_fitness}")

    def run_iteration(self, i: int):
        self.solutions = random_search(self.problem.dimension, self.iterations)
        self.solutions_fitness = fitness_function(self.solutions, self.problem.distance_matrix)

    def best_solution(self):
        return np.min(self.history["fitness"])
