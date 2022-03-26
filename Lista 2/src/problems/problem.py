import os
import yaml
from typing import Union

import numpy as np


class Problem:
    def __init__(self, name: str, distance_matrix: np.ndarray, dimension: int, minimal_length: Union[int, float], verbose: bool = True):
        self.name = name
        self.distance_matrix = distance_matrix
        self.dimension = dimension
        self.minimal_length = minimal_length

        if verbose:
            print("----------------------------------------------")
            print(f"Loaded problem: {self.name}")
            print(f"DIMENSION = {self.dimension}")
            print(f"MINIMAL_LENGTH = {self.minimal_length}")
            print("----------------------------------------------\n")

    
    @classmethod
    def load_from_name(cls, name: str, root: str = ".data", verbose: bool = True):
        data_path = os.path.join(root, name)

        # Load info metadata yaml
        with open(os.path.join(data_path, "info.yaml"), 'r') as file:
            info = yaml.safe_load(file)

        # Load distance matrix
        with open(os.path.join(data_path, f"{name.lower()}_d.txt"), 'r') as file:
            distance_matrix = np.loadtxt(file)

        return cls(name, distance_matrix, info["DIMENSION"], info["MINIMAL_LENGTH"], verbose)

    @classmethod
    def generate(cls, dimension: int, verbose: bool = True):
        distance_matrix = np.random.uniform(size=(dimension, dimension))
        np.fill_diagonal(distance_matrix, 0)
        distance_matrix = np.tril(distance_matrix) + np.tril(distance_matrix, -1).T

        return cls("GENERATED PROBLEM", distance_matrix, dimension, None, verbose)
