import os
from typing import List
from tqdm import tqdm
import pandas as pd

from src.problems.problem import Problem
from src.heuristics.genetic_algorithm import GeneticAlgorithm

def benchmark_params(problem: Problem, generations: int, pop_sizes: List[int], tours: List[int], repeats: int, csv_path: str):
    data = []

    # Generate data
    with tqdm(total=repeats) as rbar:
        for _ in range(repeats):
            for tour in tours:
                for pop_size in pop_sizes:
                    ga = GeneticAlgorithm(problem, generations, pop_size, tour)
                    ga.run(verbose=False)
                    
                    history_df = ga.history
                    history_df["tour"] = tour
                    history_df["pop_size"] = pop_size
                    
                    data.append(history_df)
            rbar.update()

    # Save data
    os.makedirs(csv_path, exist_ok=True)
    out_path = os.path.join(csv_path, f"{problem.name}.csv")

    data_df = pd.concat(data)
    data_df.to_csv(out_path, index_label="iteration")

    print("Saved benchmark data at:", out_path)


def benchmark_probs(
    problem: Problem, 
    generations: int, 
    pop_size: int, 
    tour: int, 
    mutation_probs: List[int], 
    crossover_probs: List[int],
    repeats: int,
    csv_path: str
):
    data = []

    with tqdm(total=repeats) as rbar:
        for _ in range(repeats):
            for mutation_prob in mutation_probs:
                for crossover_prob in crossover_probs:
                    ga = GeneticAlgorithm(
                        problem, 
                        generations, 
                        pop_size, 
                        tour, 
                        mutation_prob=mutation_prob, 
                        crossover_prob=crossover_prob
                    )
                    ga.run(verbose=False)
                    
                    history_df = ga.history
                    history_df["mutation_prob"] = mutation_prob
                    history_df["crossover_prob"] = crossover_prob
                    
                    data.append(history_df)
            rbar.update()


    # Save data
    os.makedirs(csv_path, exist_ok=True)
    out_path = os.path.join(csv_path, f"{problem.name}.csv")

    data_df = pd.concat(data)
    data_df.to_csv(out_path, index_label="iteration")

    print("Saved benchmark data at:", out_path)