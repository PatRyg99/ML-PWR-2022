import matplotlib.pyplot as plt
import pandas as pd

def plot_benchmark_params(data_path: str, optimum: float = None):
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

