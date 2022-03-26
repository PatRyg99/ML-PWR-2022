from src.problems.problem import Problem
from src.heuristics.random_search import RandomSearch
from src.genetic_utils.greedy_search import greedy_search

class GreedySearch(RandomSearch):
    def __init__(self, problem: Problem):
        super().__init__(problem, 0)

    def run_iteration(self, i: int):
        self.solutions, self.solutions_fitness = greedy_search(self.problem.distance_matrix)