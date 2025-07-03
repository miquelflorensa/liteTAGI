import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# --- 1. Data Handling Class ---
class DataManager:
    """Handles data creation, normalization, and splitting."""
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        """
        Initializes the DataManager with normalization parameters.

        Args:
            mean (float): The mean to use for normalization/denormalization.
            std (float): The standard deviation to use for normalization/denormalization.
        """
        self.mean = mean
        self.std = std

    def create_sinusoidal_data(self, n_samples: int = 200, noise_std: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """Generates noisy data from a sinusoidal function."""
        x = np.linspace(-3 * np.pi, 3 * np.pi, n_samples).reshape(-1, 1)
        y = np.sin(x) + np.random.randn(n_samples, 1) * noise_std
        return x, y

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalizes data using the instance's mean and std."""
        # Ensure data is not constant to avoid division by zero during normalization
        if np.std(data) < 1e-9:
            # If standard deviation is effectively zero, just center the data
            return data - np.mean(data)
        return (data - self.mean) / self.std

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """Denormalizes data using the instance's mean and std."""
        return data * self.std + self.mean

    @staticmethod
    def split_data(x: np.ndarray, y: np.ndarray, train_frac: float = 0.8) -> tuple:
        """Randomly splits data into training and testing sets."""
        n_samples = x.shape[0]
        indices = np.random.permutation(n_samples)
        n_train = int(n_samples * train_frac)
        
        train_idx, test_idx = indices[:n_train], indices[n_train:]
        return x[train_idx], y[train_idx], x[test_idx], y[test_idx]

# --- 2. Network Layer Class ---
class TAGILayer:
    """
    Represents a single layer in a Tractable Approximate Gaussian Inference (TAGI) network.
    Manages the parameters (weights, biases) and the forward/backward propagation logic.
    """
    def __init__(self, n_inputs: int, n_outputs: int, is_output_layer: bool = False):
        self.is_output_layer = is_output_layer

        # Initialize parameters. Using He-like initialization for ReLU
        init_var_weights = 2.0 / n_inputs 
        init_std_weights = np.sqrt(init_var_weights)

        self.weight_mean = np.random.randn(n_inputs, n_outputs) * init_std_weights
        self.weight_var = np.full((n_inputs, n_outputs), init_var_weights)
        
        self.bias_mean = np.zeros((1, n_outputs))
        self.bias_var = np.full((1, n_outputs), 0.5) # Small initial variance for biases

        # State variables to be updated during the forward pass (have batch dimension)
        self.pre_activation_mean = None # Stores mean of pre-activations (z) for the current layer
        self.pre_activation_var = None  # Stores variance of pre-activations (z) for the current layer
        self.jacobian = None            # Stores the Jacobian of the activation function w.r.t pre-activations

    def forward(self, prev_activation_mean: np.ndarray, prev_activation_var: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs the forward pass for this layer.
        
        Args:
            prev_activation_mean (np.ndarray): Mean of post-activations from the previous layer. Shape: (batch_size, n_inputs)
            prev_activation_var (np.ndarray): Variance of post-activations from the previous layer. Shape: (batch_size, n_inputs)
            
        Returns:
            A tuple containing the mean and variance of this layer's post-activations. Shape: (batch_size, n_outputs)
        """
        # (1) Propagate means and variances to get pre-activations (z)
        # E[z] = E[A_prev] @ E[W] + E[B]
        self.pre_activation_mean = prev_activation_mean @ self.weight_mean + self.bias_mean
        
        # Var[z] = E[A_prev^2] @ Var[W] + Var[A_prev] @ E[W]^2 + Var[A_prev] @ Var[W] + Var[B]
        # E[A_prev^2] = Var[A_prev] + E[A_prev]^2
        # So, E[A_prev^2] @ Var[W] becomes (prev_activation_var + prev_activation_mean**2) @ self.weight_var
        self.pre_activation_var = (
            (prev_activation_mean**2 @ self.weight_var) + # E[A_prev]^2 @ Var[W]
            (prev_activation_var @ self.weight_mean**2) + # Var[A_prev] @ E[W]^2
            (prev_activation_var @ self.weight_var) +     # Var[A_prev] @ Var[W]
            self.bias_var                                 # Var[B]
        )

        # (2) Apply activation function (ReLU or identity for the output layer)
        if self.is_output_layer:
            # For regression, the output layer typically uses an identity activation.
            activation_mean = self.pre_activation_mean
            activation_var = self.pre_activation_var
            self.jacobian = np.ones_like(self.pre_activation_mean) # Jacobian is 1 for identity
        else:
            # ReLU activation: max(0, x)
            is_positive = self.pre_activation_mean > 0
            activation_mean = self.pre_activation_mean * is_positive
            activation_var = self.pre_activation_var * is_positive # Variance also goes to zero if mean is zero
            self.jacobian = 1.0 * is_positive # Jacobian of ReLU is 1 if x > 0, else 0

        return activation_mean, activation_var

    def backward(self, prev_activation_mean: np.ndarray, prev_activation_var: np.ndarray, 
                 next_delta_mean: np.ndarray, next_delta_var: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs the backward smoothing pass to update layer parameters and propagate deltas.
        
        Args:
            prev_activation_mean (np.ndarray): Mean of post-activations from the previous layer (input to current layer).
                                               Shape: (batch_size, n_inputs)
            prev_activation_var (np.ndarray): Variance of post-activations from the previous layer.
                                              Shape: (batch_size, n_inputs)
            next_delta_mean (np.ndarray): Smoothed mean of the current layer's PRE-ACTIVATION (Z).
                                          Shape: (batch_size, n_outputs)
            next_delta_var (np.ndarray): Smoothed variance of the current layer's PRE-ACTIVATION (Z).
                                         Shape: (batch_size, n_outputs)
            
        Returns:
            A tuple containing the smoothed mean and variance for the previous layer's POST-ACTIVATION (A).
            Shape: (batch_size, n_inputs)
        """
        batch_size = prev_activation_mean.shape[0]
        n_inputs = prev_activation_mean.shape[1]
        n_outputs = self.weight_mean.shape[1]

        # (1) Update weights and biases using Kalman filter-like updates
        # Cov(Z_i, W_jk) = E[ (Z_i - E[Z_i])(W_jk - E[W_jk]) ]
        # Z_i = sum_j (W_ji * A_j) + B_i
        # Cov(Z_i, W_mk) for a specific k is A_m * Var(W_mk)
        cov_z_w = np.expand_dims(prev_activation_mean, axis=2) * np.expand_dims(self.weight_var, axis=0) 
        
        # Cov(Z_i, B_k) = Var(B_k) for a specific k
        cov_z_b = np.tile(np.expand_dims(self.bias_var, axis=0), (batch_size, 1, 1))

        # Kalman gains for weights and biases: K = Cov(parameter, Z) / Var(Z)
        safe_pre_activation_var = np.where(self.pre_activation_var == 0, 1e-9, self.pre_activation_var)
        expanded_safe_pre_activation_var = np.expand_dims(safe_pre_activation_var, axis=1) # Shape: (batch_size, 1, n_outputs)
        
        # Calculate Kalman gain for weights: K_w = Cov(W, Z) / Var(Z)
        gain_w_out_shape = (batch_size, n_inputs, n_outputs)
        gain_w = np.divide(cov_z_w, expanded_safe_pre_activation_var, out=np.zeros(gain_w_out_shape))
        
        # Calculate Kalman gain for biases: K_b = Cov(B, Z) / Var(Z)
        gain_b_out_shape = (batch_size, 1, n_outputs)
        gain_b = np.divide(cov_z_b, expanded_safe_pre_activation_var, out=np.zeros(gain_b_out_shape)) 
        
        # Error terms: (smoothed_Z - predicted_Z)
        err_mean = next_delta_mean - self.pre_activation_mean 
        err_var_term = next_delta_var - self.pre_activation_var 

        # Update weight parameters (mean over batch for shared parameters)
        # E[W_new] = E[W_old] + K_w * (smoothed_Z - predicted_Z)
        self.weight_mean += np.mean(gain_w * np.expand_dims(err_mean, axis=1), axis=0)
        self.weight_var += np.mean(gain_w**2 * np.expand_dims(err_var_term, axis=1), axis=0)
        self.weight_var = np.maximum(self.weight_var, 1e-9) # Ensure variance remains positive
        
        # Update bias parameters (mean over batch for shared parameters)
        # E[B_new] = E[B_old] + K_b * (smoothed_Z - predicted_Z)
        self.bias_mean += np.mean(gain_b * np.expand_dims(err_mean, axis=1), axis=0).reshape(1, n_outputs)
        self.bias_var += np.mean(gain_b**2 * np.expand_dims(err_var_term, axis=1), axis=0).reshape(1, n_outputs)
        self.bias_var = np.maximum(self.bias_var, 1e-9) # Ensure variance remains positive

        # (2) Propagate deltas to the previous layer to get the smoothed POST-ACTIVATION (delta_a)
        # Gain for propagating deltas state-to-state (A_prev to Z_curr)
        # G = Cov(A_prev, Z_curr) / Var(Z_curr) = E[W_mean] * Var(A_prev) / Var(Z_curr)
        gain_numerator = np.expand_dims(self.weight_mean, axis=0) * np.expand_dims(prev_activation_var, axis=2)
        gain_matrix_out_shape = (batch_size, n_inputs, n_outputs)
        gain_matrix = np.divide(gain_numerator, expanded_safe_pre_activation_var, out=np.zeros(gain_matrix_out_shape))

        # Propagate mean delta: E[A_prev|smoothed] = E[A_prev|predicted] + G * (smoothed_Z - predicted_Z)
        correction_mean = np.sum(err_mean[:, np.newaxis, :] * gain_matrix, axis=2)
        delta_mean_prev = prev_activation_mean + correction_mean

        # Propagate variance delta: Var[A_prev|smoothed] = Var[A_prev|predicted] - G^2 * (predicted_Var_Z - smoothed_Var_Z)
        # This involves the information gain from the smoothing operation
        err_var_term_diag = np.zeros((batch_size, n_outputs, n_outputs))
        for k in range(batch_size):
            err_var_term_diag[k, :, :] = np.diag(-err_var_term[k, :])

        # Var_correction_matrix = G @ (smoothed_Var_Z - predicted_Var_Z) @ G^T
        var_correction_matrix = np.matmul(np.matmul(gain_matrix, err_var_term_diag), np.transpose(gain_matrix, (0, 2, 1)))
        
        delta_var_prev = prev_activation_mean - np.diagonal(var_correction_matrix, axis1=1, axis2=2) 
        delta_var_prev = np.maximum(delta_var_prev, 1e-9) # Ensure variance remains positive

        return delta_mean_prev, delta_var_prev

# --- 3. Main Network Class (Modified for initialization) ---
class TAGINetwork:
    """A Bayesian Neural Network using Tractable Approximate Gaussian Inference."""
    def __init__(self, layer_units: List[int]):
        """
        Initializes the network with a sequence of layers.
        
        Args:
            layer_units (List[int]): A list of integers defining the number of units in each layer.
        """
        self.layers = []
        for i in range(len(layer_units) - 1):
            # Determine if the current layer is the final output layer
            is_output = (i == len(layer_units) - 2)
            self.layers.append(TAGILayer(layer_units[i], layer_units[i+1], is_output_layer=is_output))
    
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs a forward pass through all layers to get a prediction (mean and variance).
        Compatible with batch sizes.
        """
        activation_mean = x
        activation_var = np.zeros_like(x) # Initial input variance is assumed to be zero
        
        for layer in self.layers:
            activation_mean, activation_var = layer.forward(activation_mean, activation_var)
            
        return activation_mean, activation_var

    def update(self, x: np.ndarray, y: np.ndarray, obs_noise_var: float):
        """
        Performs a full forward and backward (smoothing) pass to update network parameters.
        
        Args:
            x (np.ndarray): Input batch. Shape: (batch_size, n_inputs)
            y (np.ndarray): Target batch (for regression). Shape: (batch_size, n_outputs)
            obs_noise_var (float): Variance of the observation noise (uncertainty in y).
        """
        # --- Forward Pass: Store all layer activations for use in the backward pass ---
        activation_means = [x]
        activation_vars = [np.zeros_like(x)] # Input variance
        
        current_mean, current_var = x, np.zeros_like(x)
        for layer in self.layers:
            current_mean, current_var = layer.forward(current_mean, current_var)
            activation_means.append(current_mean)
            activation_vars.append(current_var)

        # --- Backward Pass (Smoothing) ---
        # Initialize smoothed state delta at the output layer based on the observation y
        output_mean, output_var = activation_means[-1], activation_vars[-1]
        
        # Calculate the gain for the output layer. This gain helps in updating the pre-activations
        # of the output layer based on the observed target `y`.
        cov_yz = output_var
        total_output_var = output_var + obs_noise_var 

        safe_total_output_var = np.where(total_output_var == 0, 1e-9, total_output_var)
        gain = np.divide(cov_yz, safe_total_output_var)

        # Calculate the smoothed pre-activation (delta_z) for the final layer
        # This is essentially the Kalman update for the hidden state Z given observation Y.
        delta_z_mean = output_mean + gain * (y - output_mean)
        delta_z_var = output_var - gain * cov_yz
        delta_z_var = np.maximum(delta_z_var, 1e-9) # Ensure variance is non-negative

        # Propagate deltas backward through the network, updating parameters layer by layer
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            
            # Get the PREDICTED post-activations of the PREVIOUS layer (which serve as input to current layer)
            prev_a_mean_predicted = activation_means[i]
            prev_a_var_predicted = activation_vars[i]

            # STEP 1: Use the current layer's `backward` method to update its own weights/biases
            # and to compute the smoothed POST-ACTIVATION (delta_a) for the PREVIOUS layer.
            delta_a_mean, delta_a_var = layer.backward(
                prev_a_mean_predicted, 
                prev_a_var_predicted, 
                delta_z_mean, # Smoothed pre-activation mean for the current layer
                delta_z_var   # Smoothed pre-activation variance for the current layer
            )

            # STEP 2: If not the first layer, convert the smoothed POST-ACTIVATION (delta_a)
            # of the previous layer into the smoothed PRE-ACTIVATION (delta_z) of that same previous layer.
            # This `delta_z` will be used in the next iteration for the previous layer's `backward` call.
            if i > 0:
                prev_layer = self.layers[i-1]
                
                # Get predicted pre-activations (z) for the previous layer before any smoothing
                z_mean_prev_predicted = prev_layer.pre_activation_mean
                z_var_prev_predicted = prev_layer.pre_activation_var
                J_prev = prev_layer.jacobian # Jacobian of the previous layer's activation function

                # Calculate the gain to move from smoothed 'a' space to smoothed 'z' space for the previous layer.
                # G = Cov(z_prev, a_prev) / Var(a_prev) = J_prev * Var(z_prev) / Var(a_prev)
                safe_prev_a_var_predicted = np.where(activation_vars[i] == 0, 1e-9, activation_vars[i]) 
                gain_za = J_prev * np.divide(z_var_prev_predicted, safe_prev_a_var_predicted)
                
                # Update the pre-activation (z) of the previous layer using the smoothed 'a' from that layer
                delta_z_mean = z_mean_prev_predicted + gain_za * (delta_a_mean - activation_means[i])
                delta_z_var = z_var_prev_predicted + gain_za**2 * (delta_a_var - activation_vars[i])
                delta_z_var = np.maximum(delta_z_var, 1e-9) # Ensure variance is non-negative
    
    def initialize_parameters_inference(self, x_init_batch: np.ndarray):
        """
        Iterates layer by layer to compute S (sum of pre-activations) and S2 (sum of squares of pre-activations).
        This version only performs forward passes and calculates moments, without updating parameters.
        
        Args:
            x_init_batch (np.ndarray): A small batch of initial input data (normalized)
                                       to propagate through for initialization.
        """
        print("Starting inference-based parameter initialization (compute S and S2 only)...")
        
        # Start with the initial batch as the first layer's input (current_a_mean, current_a_var)
        current_a_mean = x_init_batch
        current_a_var = np.zeros_like(x_init_batch) # Assume input variance is zero

        # Iterate forward through layers
        for i, layer in enumerate(self.layers):
            n_outputs_current_layer = layer.weight_mean.shape[1] # A in the formulation (number of units in the layer)
            batch_size = x_init_batch.shape[0]

            print(f"\n--- Processing Layer {i+1} (n_outputs={n_outputs_current_layer}) ---")

            # --- 1. Forward pass for the current layer to get predicted Z (pre_activation_mean/var) ---
            # This uses the current (unmodified) parameters of the layer
            # and the current_a_mean/var from the *previous* layer.
            # This also populates layer.pre_activation_mean, layer.pre_activation_var, layer.jacobian
            current_a_mean, current_a_var = layer.forward(current_a_mean, current_a_var) 

            # The predicted Z for this layer
            predicted_z_mean = layer.pre_activation_mean
            predicted_z_var = layer.pre_activation_var

            # --- Calculate PREDICTED moments of S and S2 based on current Z ---
            # Predicted moments for S = sum(Z_i)
            current_mu_S_empirical = np.mean(np.sum(predicted_z_mean, axis=1)) # Average sum over batch
            current_sigma_S_sq_empirical = np.mean(np.sum(predicted_z_var, axis=1)) # Average sum of variances over batch
            
            # Predicted moments for S2 = sum(Z_i^2)
            # E[Z_i^2] = Var[Z_i] + E[Z_i]^2
            predicted_mu_Z_sq_i = predicted_z_var + predicted_z_mean**2
            current_mu_S2_empirical = np.mean(np.sum(predicted_mu_Z_sq_i, axis=1))

            # Var[Z_i^2] = 2*Var[Z_i]^2 + 4*Var[Z_i]*E[Z_i]^2
            predicted_sigma_Z_sq_i = 2 * predicted_z_var**2 + 4 * predicted_z_var * predicted_z_mean**2
            current_sigma_S2_sq_empirical = np.mean(np.sum(predicted_sigma_Z_sq_i, axis=1))

            print(f"  --- Empirical Moments from initial forward pass ---")
            print(f"  Sum S: Actual Mu={current_mu_S_empirical:.4f}")
            print(f"  Sum S: Actual Var={current_sigma_S_sq_empirical:.4f}")
            print(f"  Sum S2: Actual Mu={current_mu_S2_empirical:.4f}")
            print(f"  Sum S2: Actual Var={current_sigma_S2_sq_empirical:.4f}")
            # No parameter updates or further smoothing in this version
            
        print("\nInference-based parameter initialization (compute S and S2 only) complete.")

def main():
    """Main function to run the TAGI network training and evaluation on a 1D sinusoidal regression problem."""
    # --- Configuration ---
    np.random.seed(42) # For reproducibility
    
    # For a 1D regression problem: input size is 1, output size is 1
    input_size = 1 
    output_size = 1 

    # Network architecture: input -> 3 hidden layers (32 units each) -> output
    UNITS = [input_size, 32, 32, 32, output_size] 
    
    EPOCHS = 200 # Number of training epochs
    BATCH_SIZE = 32 # Number of samples per batch
    OBS_NOISE_VAR = 0.1 # Assumed variance of the observation noise for the training process

    # --- Data Preparation ---
    print("Generating sinusoidal data...")
    data_manager = DataManager()
    x_full, y_full = data_manager.create_sinusoidal_data(n_samples=500, noise_std=0.1)

    # Normalize inputs and outputs. This is crucial for TAGI's stability and performance.
    x_mean, x_std = x_full.mean(), x_full.std()
    y_mean, y_std = y_full.mean(), y_full.std()
    
    # Create DataManager instances for normalization/denormalization for x and y separately
    x_data_manager = DataManager(mean=x_mean, std=x_std)
    y_data_manager = DataManager(mean=y_mean, std=y_std)

    x_normalized = x_data_manager.normalize(x_full)
    y_normalized = y_data_manager.normalize(y_full)

    # Split data into training and testing sets
    x_train_norm, y_train_norm, x_test_norm, y_test_norm = DataManager.split_data(x_normalized, y_normalized, train_frac=0.8)

    print(f"Normalized x_train shape: {x_train_norm.shape}, y_train shape: {y_train_norm.shape}")
    print(f"Normalized x_test shape: {x_test_norm.shape}, y_test shape: {y_test_norm.shape}")

    # --- Model Initialization ---
    model = TAGINetwork(UNITS)
    # Perform inference-based parameter initialization using a small batch of training data.
    # This sets initial weights and biases to more sensible values.
    model.initialize_parameters_inference(x_train_norm[:BATCH_SIZE])

    # --- Model Training ---
    print("\nStarting training...")
    n_train_samples = x_train_norm.shape[0]
    
    for epoch in range(EPOCHS):
        indices = np.random.permutation(n_train_samples) # Shuffle data indices for stochastic gradient descent-like updates
        
        # Iterate through data in mini-batches
        for i in range(0, n_train_samples, BATCH_SIZE):
            batch_indices = indices[i:i + BATCH_SIZE]
            x_batch = x_train_norm[batch_indices]
            y_batch = y_train_norm[batch_indices]
            # Update network parameters using the current batch
            model.update(x_batch, y_batch, OBS_NOISE_VAR)
        
        # Evaluate performance on the test set after each epoch
        y_pred_mean_norm, _ = model.predict(x_test_norm)
        
        # Denormalize predictions and true values back to original scale for meaningful MSE calculation
        y_pred_denorm = y_data_manager.denormalize(y_pred_mean_norm)
        y_test_denorm = y_data_manager.denormalize(y_test_norm)

        # Calculate Mean Squared Error (MSE)
        mse = np.mean((y_pred_denorm - y_test_denorm)**2)
        
        if (epoch + 1) % 20 == 0: # Print MSE every 20 epochs
            print(f"Epoch {epoch + 1}/{EPOCHS} complete. Test MSE: {mse:.4f}")

    print("\nTraining complete.")
    
    # --- Final Evaluation and Visualization ---
    # Create a dense set of points across the original X range to plot the learned function smoothly
    x_plot = np.linspace(x_full.min(), x_full.max(), 500).reshape(-1, 1)
    # Normalize these points before feeding to the model
    x_plot_norm = x_data_manager.normalize(x_plot)

    # Get predictions (mean and variance) from the trained model
    y_pred_mean_plot_norm, y_pred_var_plot_norm = model.predict(x_plot_norm)

    # Denormalize predictions back to the original scale for plotting
    y_pred_mean_plot = y_data_manager.denormalize(y_pred_mean_plot_norm)
    # Denormalize the predicted standard deviation (sigma_denormalized = sigma_normalized * y_std)
    y_pred_std_plot = np.sqrt(y_pred_var_plot_norm) * y_data_manager.std 

    plt.figure(figsize=(10, 6))
    plt.scatter(x_full, y_full, s=10, alpha=0.6, label='Noisy Data (Training + Test)')
    plt.plot(x_plot, y_pred_mean_plot, color='red', linewidth=2, label='Predicted Mean Function')
    # Plot uncertainty band (e.g., +/- 2 standard deviations)
    plt.fill_between(x_plot.flatten(), 
                     (y_pred_mean_plot - 2 * y_pred_std_plot).flatten(), 
                     (y_pred_mean_plot + 2 * y_pred_std_plot).flatten(), 
                     color='red', alpha=0.2, label='$\pm 2 \sigma$ Prediction Uncertainty')
    plt.title('TAGI Network Regression on Sinusoidal Data (with Inference-based Initialization)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
