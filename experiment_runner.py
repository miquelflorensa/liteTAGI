# experiment_runner.py

import time
from typing import Callable, List, Dict, Any

class ExperimentRunner:
    """
    Manages the execution and results of multiple training sessions.
    """
    def __init__(self, training_function: Callable[[Dict[str, Any]], float]):
        """
        Initializes the runner with a function that runs a single experiment.
        
        Args:
            training_function (Callable): A function that takes a config dict
                                          and returns a performance metric (e.g., accuracy).
        """
        self.training_function = training_function
        self.experiment_configs = []
        self.results = []

    def add_experiment(self, config: Dict[str, Any]):
        """Adds a single experiment configuration to the queue."""
        self.experiment_configs.append(config)

    def run_all(self, verbose: bool = True):
        """
        Runs all queued experiments, stores their results, and prints progress.
        """
        num_experiments = len(self.experiment_configs)
        print(f"--- Starting {num_experiments} experiments ---")
        
        for i, config in enumerate(self.experiment_configs):
            start_time = time.time()
            if verbose:
                print(f"\n[Experiment {i+1}/{num_experiments}] Running with config: {config}")

            # Run the actual training
            accuracy = self.training_function(config)
            
            end_time = time.time()
            duration = end_time - start_time
            
            self.results.append({
                'config': config,
                'accuracy': accuracy,
                'duration_seconds': duration
            })
            
            if verbose:
                print(f"  -> Finished in {duration:.2f}s. Final Accuracy: {accuracy:.4f}")
        
        print("\n--- All experiments complete ---")

    def show_results(self):
        """
        Displays a summary of all experiment results, sorted by accuracy.
        """
        if not self.results:
            print("No results to show. Run experiments first.")
            return

        # Sort results by accuracy in descending order
        sorted_results = sorted(self.results, key=lambda x: x['accuracy'], reverse=True)
        
        print("\n--- Experiment Results Summary (Best to Worst) ---")
        print("-" * 70)
        print(f"{'Rank':<5} | {'Accuracy':<10} | {'σ_M^2':<7} | {'σ_Z^2':<7} | {'Config'} ")
        print("-" * 70)

        for i, result in enumerate(sorted_results):
            acc = f"{result['accuracy']:.4f}"
            sigma_m = result['config']['sigma_m_sq']
            sigma_z = result['config']['sigma_z_sq']
            cfg = (f"epochs={result['config']['epochs']}, "
                   f"batch={result['config']['batch_size']}")

            print(f"{i+1:<5} | {acc:<10} | {sigma_m:<7} | {sigma_z:<7} | {cfg}")
        
        print("-" * 70)