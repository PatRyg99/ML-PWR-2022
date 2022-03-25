import numpy as np
from tqdm import tqdm
from pqdm.processes import pqdm
from src.genetic_utils.initialization import generate_distance_matrix
class Heuristic:
    def __init__(self, num_genes: int, iterations: int):
        self.num_genes = num_genes
        self.iterations = iterations
    
    def on_start(self, distance_matrix: np.ndarray):
        if distance_matrix is None:
            self.distance_matrix = generate_distance_matrix(self.num_genes)
        else:
            self.distance_matrix = distance_matrix

    def on_end(self):
        pass
    
    def run_repeat(self, repeats: int):
        pqdm(range(repeats), self.run)   

    def run(self, distance_matrix: np.ndarray = None):
        self.on_start(distance_matrix)

        with tqdm(total = self.iterations) as pbar:
            for i in range(self.iterations):
                pbar.set_description(f"Iteration {i+1}")
                self.run_iteration(i)      
                pbar.update()

        self.on_end()

    def run_iteration(self, i: int):
        raise NotImplementedError