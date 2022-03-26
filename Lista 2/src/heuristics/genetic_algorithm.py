import numpy as np
import pandas as pd

from src.heuristics.heuristic import Heuristic
from src.genetic_utils.initialization import init_population
from src.genetic_utils.fitness_function import fitness_function
from src.genetic_utils.mutations import mutation_inverse
from src.genetic_utils.crossovers import crossover_ordered
from src.genetic_utils.selection import selection_tournament


class GeneticAlgorithm(Heuristic):
    def __init__(self, pop_size: int, num_genes: int, tour: int, generations: int):
        super().__init__(num_genes, generations)

        self.pop_size = pop_size
        self.tour = tour

        self.mutation_prob = 0.1
        self.crossover_prob = 0.8

        self.history = []

    def on_start(self, distance_matrix: np.ndarray):
        super().on_start(distance_matrix)
        self.population = init_population(self.num_genes, self.pop_size)
        self.population_fitness = fitness_function(self.population, self.distance_matrix)

        self.log_history()

    def on_end(self):
        self.history = pd.DataFrame(self.history, columns=["best", "mean", "worst"])

    def run_iteration(self, i: int):
        pop_parents = selection_tournament(self.population, self.population_fitness, self.tour)

        pop_children = crossover_ordered(pop_parents, prob=self.crossover_prob)
        self.population = mutation_inverse(pop_children, prob=self.mutation_prob)
        self.population_fitness = fitness_function(self.population, self.distance_matrix)

        self.log_history()

    def log_history(self):
        self.history.append((
            np.min(self.population_fitness),
            np.mean(self.population_fitness),
            np.max(self.population_fitness)
        ))