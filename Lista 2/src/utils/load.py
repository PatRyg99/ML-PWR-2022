from typing import List
import os

import pandas as pd

from src.problems.problem import Problem

def load_comparison_data(heuristics: List[str], problems: List[str]):
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

    return problems_data
