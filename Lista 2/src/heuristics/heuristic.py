from tqdm import tqdm
from pqdm.processes import pqdm

class Heuristic:
    def __init__(self, iterations: int):
        self.iterations = iterations
    
    def on_start(self):
        pass

    def on_end(self):
        pass
    
    def run_repeat(self, repeats: int):
        pqdm(range(repeats), self.run)   

    def run(self):
        self.on_start()

        with tqdm(total = self.iterations) as pbar:
            for i in range(self.iterations):
                pbar.set_description(f"Iteration {i+1}")
                self.run_iteration(i)      
                pbar.update()

        self.on_end()

    def run_iteration(self, i: int):
        raise NotImplementedError