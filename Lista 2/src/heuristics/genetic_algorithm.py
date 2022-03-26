import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.problems.problem import Problem
from src.heuristics.heuristic import Heuristic
from src.genetic_utils.initialization import init_population
from src.genetic_utils.fitness_function import fitness_function
from src.genetic_utils.mutations import mutation_inverse
from src.genetic_utils.crossovers import crossover_ordered
from src.genetic_utils.selection import selection_tournament


class GeneticAlgorithm(Heuristic):
    def __init__(
        self, 
        problem: Problem,
        generations: int, 
        pop_size: int, 
        tour: int, 
        mutation_prob: float = 0.1, 
        crossover_prob: float = 0.8
    ):
        super().__init__(problem, generations)

        self.pop_size = pop_size
        self.tour = tour

        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob

    def on_start(self, verbose: bool):
        super().on_start(verbose)
        self.population = init_population(self.problem.dimension, self.pop_size)
        self.population_fitness = fitness_function(self.population, self.problem.distance_matrix)

        self.log_history()

    def on_end(self, verbose: bool):
        self.history = pd.DataFrame(self.history, columns=["best", "mean", "worst", "best_solution"])

        if verbose:
            best_fitness = np.min(self.history["best"])
            print(f"Best fitness: {best_fitness}")

            self.plot_history()

    def run_iteration(self, i: int):
        pop_parents = selection_tournament(self.population, self.population_fitness, self.tour)

        pop_children = crossover_ordered(pop_parents, prob=self.crossover_prob)
        self.population = mutation_inverse(pop_children, prob=self.mutation_prob)
        self.population_fitness = fitness_function(self.population, self.problem.distance_matrix)

        self.log_history()

    def log_history(self):
        self.history.append((
            np.min(self.population_fitness),
            np.mean(self.population_fitness),
            np.max(self.population_fitness),
            self.population[np.argmin(self.population_fitness)],
        ))

    def plot_history(self):
        xs = np.arange(len(self.history))
        
        plt.plot(xs, self.history["best"], label="Best fitness")
        plt.plot(xs, self.history["mean"], label="Mean fitness")
        plt.plot(xs, self.history["worst"], label="Worst fitness")

        plt.title("Fitness progression plot")
        plt.legend()