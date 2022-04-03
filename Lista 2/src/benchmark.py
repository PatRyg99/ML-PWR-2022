import os
from typing import List
from tqdm import tqdm
import pandas as pd

from src.problems.problem import Problem
from src.heuristics.genetic_algorithm import GeneticAlgorithm

def benchmark_params(problem: Problem, generations: int, pop_sizes: List[int], tour_ratios: List[int], csv_path: str):
    data = []

    # Generate data
    with tqdm(total=len(pop_sizes) * len(tour_ratios)) as pbar:
        for tour_ratio in tour_ratios:
            for pop_size in pop_sizes:
                ga = GeneticAlgorithm(problem, generations, pop_size, int(pop_size * tour_ratio))
                ga.run(verbose=False)
                
                history_df = ga.history
                history_df = history_df.drop(columns=["best_solution"])
                history_df["tour_ratio"] = tour_ratio
                history_df["pop_size"] = pop_size
                
                data.append(history_df)
                pbar.update()

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
    tour_ratio: int, 
    mutation_probs: List[int], 
    crossover_probs: List[int],
    csv_path: str
):
    data = []

    with tqdm(total=len(mutation_probs) * len(crossover_probs)) as pbar:
        for mutation_prob in mutation_probs:
            for crossover_prob in crossover_probs:
                ga = GeneticAlgorithm(
                    problem, 
                    generations, 
                    pop_size, 
                    int(pop_size * tour_ratio), 
                    mutation_prob=mutation_prob, 
                    crossover_prob=crossover_prob
                )
                ga.run(verbose=False)
                
                history_df = ga.history
                history_df = history_df.drop(columns=["best_solution"])
                history_df["mutation_prob"] = mutation_prob
                history_df["crossover_prob"] = crossover_prob
                
                data.append(history_df)
                pbar.update()


    # Save data
    os.makedirs(csv_path, exist_ok=True)
    out_path = os.path.join(csv_path, f"{problem.name}.csv")

    data_df = pd.concat(data)
    data_df.to_csv(out_path, index_label="iteration")

    print("Saved benchmark data at:", out_path)