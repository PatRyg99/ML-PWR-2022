import os
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.problems.problem import Problem

def plot_benchmark_params(data_path: str, optimum: float = None):
    """
    Plotting benchmark for initial GA parameters
    """
    keys = ["best", "mean", "worst"]

    data = pd.read_csv(data_path)
    fig, axes = plt.subplots(len(keys), 4, figsize=(22, 12), sharey='row', sharex=True)
    fig.suptitle("Ablation study of initial parameters choice in genetic algorithm", fontsize=20)

    for i, key in enumerate(keys):
        for j, tour_ratio in enumerate(pd.unique(data["tour_ratio"])):
            tour_df = data[data["tour_ratio"] == tour_ratio]

            for pop_size in pd.unique(data["pop_size"]):
                pop_df = tour_df[tour_df["pop_size"] == pop_size]
                pop_df.plot(x="iteration", y=key, ax=axes[i, j], logy=True, label=f"pop_size={pop_size}", legend=False)
            
            axes[i, j].grid(True, which="both", ls="-")

            if optimum:
                axes[i, j].axhline(y=optimum, color='b', linestyle=':', label="optimum")

            if i == len(keys) - 1:
                axes[i, j].set_xlabel("generations", fontsize=16)

            if i == 0:
                axes[i, j].set_title(f"tour_ratio={tour_ratio}", fontsize=16)
        
            if j == 0:
                axes[i, j].set_ylabel(key, fontsize=16)

    # One legend
    handles, labels = axes[-1, -1].get_legend_handles_labels()
    lgd = fig.legend(
        handles, 
        labels, 
        loc='lower center',  
        bbox_to_anchor = (0,-0.08,1,1),
        bbox_transform = plt.gcf().transFigure, 
        ncol=len(labels),
        fontsize=20
    )
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])


def plot_benchmark_probs(data_path: str, optimum: float = None):
    """
    Plotting benchmark for mutation and crossover probabilities
    """
    keys = ["best", "mean", "worst"]

    data = pd.read_csv(data_path)
    fig, axes = plt.subplots(len(keys), 4, figsize=(22, 12), sharey='row', sharex=True)
    fig.suptitle("Ablation study of mutation and crossovers probabilities in genetic algorithm", fontsize=20)

    for i, key in enumerate(keys):
        for j, mutation_prob in enumerate(pd.unique(data["mutation_prob"])):
            mutation_df = data[data["mutation_prob"] == mutation_prob]

            for crossover_prob in pd.unique(data["crossover_prob"]):
                crossover_df = mutation_df[mutation_df["crossover_prob"] == crossover_prob]
                crossover_df.plot(x="iteration", y=key, ax=axes[i, j], logy=True, label=f"crossover_prob={crossover_prob}", legend=False)

            axes[i, j].grid(True, which="both", ls="-")

            if optimum:
                axes[i, j].axhline(y=optimum, color='b', linestyle=':', label="optimum")

            if i == len(keys) - 1:
                axes[i, j].set_xlabel("generations", fontsize=16)

            if i == 0:
                axes[i, j].set_title(f"mutation_prob={mutation_prob}", fontsize=16)
        
            if j == 0:
                axes[i, j].set_ylabel(key, fontsize=16)

    # One legend
    handles, labels = axes[-1, -1].get_legend_handles_labels()
    lgd = fig.legend(
        handles, 
        labels, 
        loc='lower center',  
        bbox_to_anchor = (0,-0.08,1,1),
        bbox_transform = plt.gcf().transFigure, 
        ncol=len(labels),
        fontsize=20
    )
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])


def plot_heuristic_comparison(heuristics: List[str], problems: List[str], return_data: bool = False):
    fig, axes = plt.subplots(2, 4, figsize=(22, 8))
    fig.suptitle("Comparison between heuristics", fontsize=20)

    # Load data
    problems_data = {}

    for problem in problems:
        problem_df = [
            pd.DataFrame(
                [(
                    "Optimum",
                    Problem.load_from_name(problem, verbose=False).minimal_length,
                    0
                )],
                columns=["Name", "Best", "Time"]
            )
        ]

        for heuristic in heuristics:
            heuristic_df = pd.read_csv(os.path.join(".results", problem, heuristic + ".csv"))
            heuristic_df["Name"] = heuristic
            problem_df.append(heuristic_df)

        problem_df = pd.concat(problem_df)
        problems_data[problem] = problem_df
    
    # Plot data
    for i, (problem_name, problem_df) in enumerate(problems_data.items()):
        sns.boxplot(x="Name", y="Best", data=problem_df, ax=axes[0, i])
        axes[0, i].grid(axis='y')
        axes[0, i].set_xlabel(None)
        axes[0, i].set_title(problem_name, fontsize=16)

        sns.boxplot(x="Name", y="Time", data=problem_df, ax=axes[1, i])
        axes[1, i].set_xlabel(None)
        axes[1, i].grid(axis='y')

        if i != 0:
            axes[0, i].set_ylabel(None)
            axes[1, i].set_ylabel(None)

    plt.tight_layout()
    plt.show()

    if return_data:
        return problems_data

def plot_heuristic_comparison_dataframe(data):
    """
    Prepare pretty print of data
    """
    df = pd.concat(list(data.values()), keys=list(data.keys()))
    df = df.reset_index(level=[0])
    df = df.rename(columns={"level_0": "Problem"})
    df = df.groupby(["Name", "Problem"]).mean()[["Best", "Time"]].unstack()
    df = df.loc[["RandomSearch", "GreedySearch", "GeneticAlgorithm", "Optimum"]]
    df = df.swaplevel(axis='columns')
    df = df.reindex([
        ("FRI26", "Best"), ("FRI26", "Time"), 
        ("BERLIN52", "Best"), ("BERLIN52", "Time"),
        ("KROA100", "Best"), ("KROA100", "Time"),
        ("TSP225", "Best"), ("TSP225", "Time")
        ], 
        axis=1
    )

    return df