import numpy as np
from tqdm import tqdm
from pqdm.processes import pqdm

from src.problems.problem import Problem
class Heuristic:
    def __init__(self, problem: Problem, generations: int):
        self.problem = problem
        self.generations = generations

    def on_start(self, verbose: bool):
        """
        On start hook.
        Runs before first iteration.
        """
        self.history = []

        if verbose:
            print(f"Running problem: {self.problem.name}")

    def on_end(self, verbose: bool):
        """
        On end hook.
        Runs after last iteration
        """
        pass

    def run_repeat(self, repeats: int):
        """
        Runs n indepent repeats of the heuristic
        """
        for i in range(repeats):
            self.run(verbose=False)
            print(f"Repeat [{i+1}/{repeats}]: {self.best_solution()}")

    def run(self, verbose: bool = True):
        """
        Runs heuristic for given number of generations.
        """
        self.on_start(verbose)

        with tqdm(total = self.generations, disable = not verbose, leave=False) as pbar:
            for i in range(self.generations):
                pbar.set_description(f"Generation {i+1}")
                self.run_iteration(i)      
                pbar.update()

        self.on_end(verbose)

    def run_iteration(self, i: int):
        """
        Runs one heuristic iteration.
        """
        raise NotImplementedError

    def best_solution(self):
        """
        Returns best solution stored in history.
        """
        raise NotImplementedError