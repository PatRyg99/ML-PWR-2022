import numpy as np
from tqdm import tqdm
from pqdm.processes import pqdm

from src.problems.problem import Problem
class Heuristic:
    def __init__(self, problem: Problem, generations: int):
        self.problem = problem
        self.generations = generations

    def on_start(self, verbose: bool):
        self.history = []
        
        if verbose:
            print(f"Running problem: {self.problem.name}")

    def on_end(self, verbose: bool):
        pass
    
    def run_repeat(self, repeats: int):
        pqdm(range(repeats), self.run)   

    def run(self, verbose: bool = True):
        self.on_start(verbose)

        with tqdm(total = self.generations) as pbar:
            for i in range(self.generations):
                pbar.set_description(f"Generation {i+1}")
                self.run_iteration(i)      
                pbar.update()

        self.on_end(verbose)

    def run_iteration(self, i: int):
        raise NotImplementedError