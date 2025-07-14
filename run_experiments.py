# run_experiments.py

from tagi_network import run_training_session
from experiment_runner import ExperimentRunner

def main():
    """
    Defines and executes the hyperparameter search for initialization targets.
    """
    # --- 1. Define the Hyperparameter Search Space ---
    # Define the range of values you want to test for sigma_m_sq and sigma_z_sq
    # The original paper's default is 1.0 for both.
    sigma_m_sq_values = [0.5, 1.0, 1.5, 2.0]
    sigma_z_sq_values = [0.5, 1.0, 1.5, 2.0]

    # --- 2. Setup the Experiment Runner ---
    runner = ExperimentRunner(training_function=run_training_session)

    # --- 3. Generate and Queue Experiment Configurations ---
    print("Generating experiment configurations...")
    for smsq in sigma_m_sq_values:
        for szsq in sigma_z_sq_values:
            config = {
                'sigma_m_sq': smsq,
                'sigma_z_sq': szsq,
                'epochs': 5,          # Keep epochs low for faster experiments
                'batch_size': 32,
                'init_batch_size': 1024 # Batch size for standardization
            }
            runner.add_experiment(config)

    # --- 4. Run All Experiments ---
    runner.run_all()

    # --- 5. Show the Results ---
    runner.show_results()

if __name__ == "__main__":
    main()