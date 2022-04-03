import os
from tqdm import tqdm
import pandas as pd

from src.problems.problem import Problem
from src.utils.timer import elapsed_timer
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
    
    def sanity_run(self):
        """
        Sanity check run to perform compilation
        """
        self.run(verbose=False)

    def run_repeat(self, repeats: int, csv_path: str = None):
        """
        Runs n indepent repeats of the heuristic and dump results to csv if specified
        """
        self.sanity_run()
        results = []

        with tqdm(total=repeats, desc=self.problem.name) as pbar:
            for _ in range(repeats):
                with elapsed_timer() as elapsed:
                    self.run(verbose=False)

                results.append((self.best_solution(), elapsed()))

                pbar.set_postfix({"best": self.best_solution()})
                pbar.update()

        if csv_path:
            os.makedirs(csv_path, exist_ok=True)
            df = pd.DataFrame(results, columns=["Best", "Time"])
            df.to_csv(os.path.join(csv_path, f"{self.__class__.__name__}.csv"))

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