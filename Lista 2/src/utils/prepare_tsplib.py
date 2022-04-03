from genericpath import exists
import re
import os
import yaml

from pqdm.processes import pqdm
import numpy as np
from numba import jit, prange

from src.genetic_utils.fitness_function import fitness_function

def preprocess_raw(raw_string: str):
    raw_string = [raw_substring.split("\n") for raw_substring in raw_string][0]
    raw_string = [raw_substring.split(" ") for raw_substring in raw_string]
    return [float(x) for row in raw_string for x in row if x != "" and x != "EOF"]

@jit(nopython=True, parallel=True)
def construct_distance_matrix(locations: np.ndarray, dimension: int):
    distance_matrix = np.zeros((dimension, dimension)) 

    for i in prange(dimension):
        for j in prange(dimension):
            if i != j:
                distance = np.linalg.norm(locations[i] - locations[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

    return distance_matrix

def convert_tsplib(problem_name: str):

    # Process tsp file
    with open(f".data/raw_all/{problem_name}.tsp", "r") as f:
        file_data = f.read()

    # Extract dimension
    dimension = int(re.findall(r'DIMENSION *:(.*)\n', file_data)[0])
    
    # If weight format not provided we deal with locations
    weight_format = re.findall(r'EDGE_WEIGHT_FORMAT *:(.*)\n', file_data)
    if not weight_format:
        coords = re.findall(r'NODE_COORD_SECTION\n((.|\n)*)', file_data)[0]

        coords = preprocess_raw(coords)
        coords = np.array(coords).reshape(dimension, 3)[:, 1:]
        distance_matrix = construct_distance_matrix(coords, dimension)

    else:
        return

    # Process optimal solution file
    with open(f".data/raw_all/{problem_name}.opt.tour", "r") as f:
        solution_data = f.read()

    solution = re.findall(r'TOUR_SECTION\n((.|\n)*)', solution_data)[0]
    solution = preprocess_raw(solution)
    solution = np.array(solution)[:dimension] - 1

    solution_fitness = fitness_function(solution[None, :], distance_matrix)[0]

    # Write files
    problem_dir = f".data/{problem_name.upper()}"
    os.makedirs(problem_dir, exist_ok=True)

    with open(f"{problem_dir}/info.yaml", "w") as info_file:
        info_dict = {"DIMENSION": dimension, "MINIMAL_LENGTH": float(solution_fitness)}
        yaml.dump(info_dict, info_file)

    np.savetxt(f"{problem_dir}/{problem_name.lower()}_d.txt", distance_matrix)


if __name__ == "__main__":
    files = os.listdir(".data/raw_all")
    problems = list(set([file.split(".")[0] for file in files]))

    pqdm(problems, convert_tsplib, n_jobs=12)
