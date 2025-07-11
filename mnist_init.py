import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from tensorflow.keras.datasets import mnist # For loading MNIST
from tensorflow.keras.utils import to_categorical # For one-hot encoding labels
from skimage.transform import resize # For resizing images
from scipy.stats import norm

# --- 1. Data Handling Class (No Changes Needed) ---
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

    def create_cubic_data(self, n_samples: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Generates noisy data from a cubic function (not used for MNIST)."""
        x = np.linspace(-5, 5, n_samples).reshape(-1, 1)
        y = x**3 + np.random.randn(n_samples, 1) * 3
        return x, y

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalizes data using the instance's mean and std."""
        return (data - self.mean) / self.std

    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """Denormalizes data using the instance's mean and std."""
        return data * self.std + self.mean

    @staticmethod
    def split_data(x: np.ndarray, y: np.ndarray, train_frac: float = 0.8) -> tuple:
        """Randomly splits data into training and testing sets (not directly used for MNIST)."""
        n_samples = x.shape[0]
        indices = np.random.permutation(n_samples)
        n_train = int(n_samples * train_frac)
        
        train_idx, test_idx = indices[:n_train], indices[n_train:]
        return x[train_idx], y[train_idx], x[test_idx], y[test_idx]

# --- 2. Network Layer Class (No Changes Needed) ---
class TAGILayer:
    """
    Represents a single layer in a Tractable Approximate Gaussian Inference (TAGI) network.
    Manages the parameters (weights, biases) and the forward/backward propagation logic.
    """
    def __init__(self, n_inputs: int, n_outputs: int, is_output_layer: bool = False):
        self.is_output_layer = is_output_layer

        # Initialize parameters with random values scaled by the number of inputs
        init_var = 0.5 / n_inputs
        init_std = np.sqrt(init_var)

        self.weight_mean = np.random.randn(n_inputs, n_outputs) * init_std
        self.weight_var = np.full((n_inputs, n_outputs), init_var)
        
        self.bias_mean = np.zeros((1, n_outputs))
        self.bias_var = np.full((1, n_outputs), 0.5)

        # State variables to be updated during the forward pass (will now have batch dimension)
        self.pre_activation_mean = None # Shape: (batch_size, n_outputs)
        self.pre_activation_var = None  # Shape: (batch_size, n_outputs)
        self.jacobian = None            # Shape: (batch_size, n_outputs)

    def forward(self, prev_activation_mean: np.ndarray, prev_activation_var: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs the forward pass for this layer, compatible with batch sizes.
        
        Args:
            prev_activation_mean (np.ndarray): Mean of activations from the previous layer. Shape: (batch_size, n_inputs)
            prev_activation_var (np.ndarray): Variance of activations from the previous layer. Shape: (batch_size, n_inputs)
            
        Returns:
            A tuple containing the mean and variance of this layer's activations. Shape: (batch_size, n_outputs)
        """
        # (1) Propagate means and variances to get pre-activations (z)
        self.pre_activation_mean = prev_activation_mean @ self.weight_mean + self.bias_mean
        self.pre_activation_var = (
            (prev_activation_mean**2 @ self.weight_var) +
            (prev_activation_var @ self.weight_mean**2) +
            (prev_activation_var @ self.weight_var) + 
            self.bias_var
        )

        # (2) Apply activation function (ReLU or identity for the output layer)
        if self.is_output_layer:
            activation_mean = self.pre_activation_mean
            activation_var = self.pre_activation_var
            self.jacobian = np.ones_like(self.pre_activation_mean)
        else:
            activation_mean, activation_var, self.jacobian = self.mReLU(self.pre_activation_mean, self.pre_activation_var)

        return activation_mean, activation_var

    def backward(self, prev_activation_mean: np.ndarray, prev_activation_var: np.ndarray, 
                 next_delta_mean: np.ndarray, next_delta_var: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs the backward smoothing pass to update layer parameters and propagate deltas,
        compatible with batch sizes.
        
        Args:
            prev_activation_mean (np.ndarray): Mean of activations from the previous layer.
            prev_activation_var (np.ndarray): Variance of activations from the previous layer.
            next_delta_mean (np.ndarray): Smoothed mean of the current layer's PRE-ACTIVATION.
            next_delta_var (np.ndarray): Smoothed variance of the current layer's PRE-ACTIVATION.
            
        Returns:
            A tuple containing the smoothed mean and variance for the previous layer's POST-ACTIVATION.
        """
        batch_size = prev_activation_mean.shape[0]
        n_inputs = prev_activation_mean.shape[1]
        n_outputs = self.weight_mean.shape[1]

        # (1) Update weights and biases using Kalman filter-like updates
        cov_z_w = np.expand_dims(prev_activation_mean, axis=2) * np.expand_dims(self.weight_var, axis=0) 
        cov_z_b = np.tile(np.expand_dims(self.bias_var, axis=0), (batch_size, 1, 1))

        safe_pre_activation_var = np.where(self.pre_activation_var == 0, 1e-9, self.pre_activation_var)
        expanded_safe_pre_activation_var = np.expand_dims(safe_pre_activation_var, axis=1) 
        
        gain_w_out_shape = (batch_size, n_inputs, n_outputs)
        gain_b_out_shape = (batch_size, 1, n_outputs)
        gain_w = np.divide(cov_z_w, expanded_safe_pre_activation_var, out=np.zeros(gain_w_out_shape))
        gain_b = np.divide(cov_z_b, expanded_safe_pre_activation_var, out=np.zeros(gain_b_out_shape)) 
        
        err_mean = next_delta_mean - self.pre_activation_mean
        err_var_term = next_delta_var - self.pre_activation_var

        self.weight_mean += np.mean(gain_w * np.expand_dims(err_mean, axis=1), axis=0)
        self.weight_var += np.mean(gain_w**2 * np.expand_dims(err_var_term, axis=1), axis=0)
        self.weight_var = np.maximum(self.weight_var, 1e-9)
        
        self.bias_mean += np.mean(gain_b * np.expand_dims(err_mean, axis=1), axis=0).reshape(1, n_outputs)
        self.bias_var += np.mean(gain_b**2 * np.expand_dims(err_var_term, axis=1), axis=0).reshape(1, n_outputs)
        self.bias_var = np.maximum(self.bias_var, 1e-9)

        # (2) Propagate deltas to the previous layer
        gain_numerator = np.expand_dims(self.weight_mean, axis=0) * np.expand_dims(prev_activation_var, axis=2)
        gain_matrix_out_shape = (batch_size, n_inputs, n_outputs)
        gain_matrix = np.divide(gain_numerator, expanded_safe_pre_activation_var, out=np.zeros(gain_matrix_out_shape))

        correction_mean = np.sum(err_mean[:, np.newaxis, :] * gain_matrix, axis=2)
        delta_mean_prev = prev_activation_mean + correction_mean

        err_var_term_diag = np.zeros((batch_size, n_outputs, n_outputs))
        for k in range(batch_size):
            err_var_term_diag[k, :, :] = np.diag(-err_var_term[k, :])

        var_correction_matrix = np.matmul(np.matmul(gain_matrix, err_var_term_diag), np.transpose(gain_matrix, (0, 2, 1)))
        
        delta_var_prev = prev_activation_var - np.diagonal(var_correction_matrix, axis1=1, axis2=2)
        delta_var_prev = np.maximum(delta_var_prev, 1e-9)

        return delta_mean_prev, delta_var_prev
    
    def mReLU(self, mean: np.ndarray, var: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # relu_mean = z_mean * Φ(z_mean/σ) + σ * φ(z_mean/σ)
        # relu_var = (z_mean² + σ²) * Φ(z_mean/σ) + 
        #     z_mean * σ * φ(z_mean/σ) - 
        #     relu_mean²
        # safe_var = np.where(var < 0, 1e-6, var)
        # z_mean = mean
        # z_std = np.maximum(np.sqrt(safe_var), 1e-9)
        # cdfn = np.maximum(norm.cdf(z_mean / z_std), 1E-20)  # CDF of standard normal, avoid division by zero
        # pdfn = np.maximum(norm.pdf(z_mean / z_std), 1E-20)  # PDF of standard normal, avoid division by zero
        # relu_mean = np.maximum(z_std * pdfn + z_mean * cdfn, 1E-20)  # Ensure relu_mean is positive
        # relu_var = np.maximum(-relu_mean * relu_mean + 2 * relu_mean * z_mean - z_mean * z_std * pdfn + (safe_var - z_mean * z_mean) * cdfn, 1E-9)  # Ensure relu_var is positive
        # jcb = cdfn
        positive = mean > 0
        relu_mean = mean * positive
        relu_var = var * positive
        jcb = np.where(positive, 1.0, 0.0)  # Jacobian is 1 where mean > 0, else 0
        return relu_mean, relu_var, jcb



# --- 3. Main Network Class (MODIFIED) ---
class TAGINetwork:
    """A Bayesian Neural Network using Tractable Approximate Gaussian Inference."""
    def __init__(self, layer_units: List[int]):
        """
        Initializes the network with a sequence of layers.
        """
        self.layers = []
        for i in range(len(layer_units) - 1):
            is_output = (i == len(layer_units) - 2)
            self.layers.append(TAGILayer(layer_units[i], layer_units[i+1], is_output_layer=is_output))
    
    # --- Weight Standardization Logic (NEW) ---
    @staticmethod
    def _calculate_pre_activation_properties(mu_Z0, var_Z0, mu_W, var_W, mu_B, var_B):
        """Calculates mean/variance of Z1 = Z0 @ W + B.
        mu_Z0: (batch_size, n_inputs)
        var_Z0: (batch_size, n_inputs)
        mu_W: (n_inputs, n_outputs)
        var_W: (n_inputs, n_outputs)
        mu_B: (1, n_outputs) or (n_outputs,)
        var_B: (1, n_outputs) or (n_outputs,)
        Returns: predicted_mu_Z1 (batch_size, n_outputs), predicted_var_Z1 (batch_size, n_outputs)
        """
        # (batch_size, n_inputs) @ (n_inputs, n_outputs) + (1, n_outputs)
        predicted_mu_Z1 = mu_Z0 @ mu_W + mu_B
        
        # Element-wise operations, then sum reduction via matrix multiplication
        # (mu_Z0**2 @ var_W) -> (batch_size, n_inputs) @ (n_inputs, n_outputs) -> (batch_size, n_outputs)
        predicted_var_Z1 = (
            (mu_Z0**2 @ var_W) +
            (var_Z0 @ mu_W**2) +
            (var_Z0 @ var_W) +
            var_B  # Bias variance is (1, n_outputs) or (n_outputs,), broadcasts correctly
        )
        return predicted_mu_Z1, predicted_var_Z1

    @staticmethod
    def _update_for_sum_target(means, variances, target_mean, target_var):
        """Updates Gaussians to match a target sum.
        Inputs: means (batch_size, units), variances (batch_size, units)
        Targets: target_mean (scalar), target_var (scalar) -> these are for the *sum* over units
        """
        # Calculate current sum statistics over the units dimension for each batch item
        mu_S = np.sum(means, axis=1, keepdims=True) # (batch_size, 1)
        sigma2_S = np.sum(variances, axis=1, keepdims=True) # (batch_size, 1)
        
        epsilon = 1e-9
        # Jacobian for the sum: dS/dZ_i = 1. The gain is var(Z_i) / var(S)
        # (batch_size, units) / (batch_size, 1) -> broadcasting happens correctly
        Jz = variances / (sigma2_S + epsilon) 
        
        updated_means = means + Jz * (target_mean - mu_S)
        updated_variances = variances + Jz * (target_var - sigma2_S)
        return updated_means, np.maximum(updated_variances, epsilon)

    @staticmethod
    def _update_for_sum_target_2(means, variances, target_mean_2, target_var_2):
        """Updates Gaussians to match a target sum of squares.
        Inputs: means (batch_size, units), variances (batch_size, units)
        Targets: target_mean_2 (scalar), target_var_2 (scalar) -> these are for the *sum of squares* over units
        """
        # Calculate current sum of squares statistics for each batch item
        mu_Z_sq = means**2 + variances # (batch_size, units)
        var_Z_sq = 2 * variances**2 + 4 * variances * means**2 # (batch_size, units)

        mu_S2 = np.sum(mu_Z_sq, axis=1, keepdims=True) # (batch_size, 1)
        sigma2_S2 = np.sum(var_Z_sq, axis=1, keepdims=True) # (batch_size, 1)
        
        epsilon = 1e-9
        # Jacobian for sum of squares: d(S2)/dZ_i = 2 * Z_i
        # Gain is approximately 2 * E[Z_i] * Var(Z_i) / Var(S2)
        # (batch_size, units) * (batch_size, units) / (batch_size, 1)
        Jz = (2 * means * variances) / (sigma2_S2 + epsilon) 
        
        # (batch_size, units) + (batch_size, units) * ((scalar) - (batch_size, 1))
        updated_means = means + Jz * (target_mean_2 - mu_S2)
        # (batch_size, units) + (batch_size, units)^2 * ((scalar) - (batch_size, 1))
        updated_variances = variances + Jz**2 * (target_var_2 - sigma2_S2) 
        return updated_means, np.maximum(updated_variances, epsilon)

    @staticmethod
    def _update_Z0W_for_Z1_target(mu_B, var_B, target_mu_Z1, target_var_Z1):
        """Infers Z0W properties by subtracting bias B from target Z1.
        Inputs: mu_B (1, n_outputs), var_B (1, n_outputs) or (n_outputs,)
                target_mu_Z1 (batch_size, n_outputs), target_var_Z1 (batch_size, n_outputs)
        Outputs: updated_mu_Z0W (batch_size, n_outputs), updated_var_Z0W (batch_size, n_outputs)
        """
        # Bias is subtracted element-wise, its mean is subtracted and variance is subtracted.
        # This assumes B is independent of Z0W.
        # (batch_size, n_outputs) - (1, n_outputs) -> broadcasting handles this
        updated_var_Z0W = np.maximum(target_var_Z1 - var_B, 1e-9)
        updated_mu_Z0W = target_mu_Z1 - mu_B
        return updated_mu_Z0W, updated_var_Z0W

    @staticmethod
    def _rts_smooth_Z0W_components(initial_mu_Z0, initial_var_Z0, initial_mu_W, initial_var_W,
                                   updated_mu_Z0W_sum_target, updated_var_Z0W_sum_target):
        """Distributes the update for the sum Z0W back to individual product terms P_ij = Z0_i * W_ij.
        This function now explicitly handles the batch dimension for P_ij calculations and updates.

        Inputs:
            initial_mu_Z0 (batch_size, n_inputs)
            initial_var_Z0 (batch_size, n_inputs)
            initial_mu_W (n_inputs, n_outputs)
            initial_var_W (n_inputs, n_outputs)
            updated_mu_Z0W_sum_target (batch_size, n_outputs)
            updated_var_Z0W_sum_target (batch_size, n_outputs)
        Outputs:
            updated_mu_P_ij (batch_size, n_inputs, n_outputs)
            updated_var_P_ij (batch_size, n_inputs, n_outputs)
        """
        epsilon = 1e-9
        
        # Calculate initial moments of P_ij = Z0_i * W_ij
        # These operations need to be broadcast across the batch dimension.
        # initial_mu_Z0 (batch_size, n_inputs) -> (batch_size, n_inputs, 1) for element-wise multiplication with W
        # initial_mu_W (n_inputs, n_outputs) -> (1, n_inputs, n_outputs) to broadcast across batches
        initial_mu_P_ij = initial_mu_Z0[:, :, np.newaxis] * initial_mu_W[np.newaxis, :, :]
        
        initial_var_P_ij = (
            (initial_var_Z0[:, :, np.newaxis] * initial_var_W[np.newaxis, :, :]) +
            (initial_var_Z0[:, :, np.newaxis] * initial_mu_W[np.newaxis, :, :]**2) +
            (initial_var_W[np.newaxis, :, :] * initial_mu_Z0[:, :, np.newaxis]**2)
        )
        
        updated_mu_P_ij = np.copy(initial_mu_P_ij)
        updated_var_P_ij = np.copy(initial_var_P_ij)

        # Sum over inputs (axis=1) to get current sum Z0W for each batch item and output unit
        current_mu_Z0W = np.sum(updated_mu_P_ij, axis=1) # Shape: (batch_size, n_outputs)
        current_var_Z0W = np.sum(updated_var_P_ij, axis=1) # Shape: (batch_size, n_outputs)
        
        error_mu_sum = updated_mu_Z0W_sum_target - current_mu_Z0W
        error_var_sum = updated_var_Z0W_sum_target - current_var_Z0W

        # Gain for updating P_ij from the Z0W sum target.
        # This gain is also batch-dependent.
        # (batch_size, n_inputs, n_outputs) / ((batch_size, 1, n_outputs) + epsilon)
        gain = updated_var_P_ij / (current_var_Z0W[:, np.newaxis, :] + epsilon)

        # Apply updates, broadcasting error terms
        updated_mu_P_ij += gain * error_mu_sum[:, np.newaxis, :]
        updated_var_P_ij += gain * error_var_sum[:, np.newaxis, :] # Gain squared for variance update
        updated_var_P_ij = np.maximum(updated_var_P_ij, epsilon)

        return updated_mu_P_ij, updated_var_P_ij

    
    @staticmethod
    def _update_W_from_Z0_and_Z0W_products(mu_Z0, var_Z0, mu_P_ij, var_P_ij, initial_mu_W, initial_var_W):
        """
        Infers updated weights W from updated product terms P_ij = Z0_i * W_ij.
        This function now explicitly handles the batch dimension for P_ij calculations and updates
        by averaging batch-wise inputs before calculation.

        Inputs:
            mu_Z0 (batch_size, n_inputs) - Mean of activations from the previous layer
            var_Z0 (batch_size, n_inputs) - Variance of activations from the previous layer
            mu_P_ij (batch_size, n_inputs, n_outputs) - Smoothed product terms
            var_P_ij (batch_size, n_inputs, n_outputs) - Smoothed product terms
            initial_mu_W (n_inputs, n_outputs) - Original weight means
            initial_var_W (n_inputs, n_outputs) - Original weight variances
        Outputs:
            updated_mu_W (n_inputs, n_outputs) - Updated weight means
            updated_var_W (n_inputs, n_outputs) - Updated weight variances
        """
        epsilon = 1e-6 # Small value to prevent division by zero

        batch_size, n_inputs = mu_Z0.shape
        n_outputs = mu_P_ij.shape[2]

        # --- Modifications Start Here ---

        # Compute the means of the mini-batches for the inputs
        # These operations are performed BEFORE the main calculations
        avg_mu_Z0 = np.mean(mu_Z0, axis=0) # (n_inputs,)
        avg_var_Z0 = np.mean(var_Z0, axis=0) # (n_inputs,)
        avg_mu_P_ij = np.mean(mu_P_ij, axis=0) # (n_inputs, n_outputs)
        avg_var_P_ij = np.mean(var_P_ij, axis=0) # (n_inputs, n_outputs)


        # Initialize arrays for the updated weights (now directly (n_inputs, n_outputs))
        updated_mu_W = np.zeros((n_inputs, n_outputs))
        updated_var_W = np.zeros((n_inputs, n_outputs))

        # The loops now iterate over n_inputs and n_outputs directly,
        # using the averaged batch values.
        for i in range(n_inputs): # Iterate over inputs (rows of W)
            for j in range(n_outputs): # Iterate over outputs (columns of W)
                
                # Update mean of W_ij using averaged batch values
                denominator_mu = avg_mu_Z0[i]
                if np.abs(denominator_mu) < epsilon:
                    updated_mu_W[i, j] = initial_mu_W[i, j]
                else:
                    updated_mu_W[i, j] = avg_mu_P_ij[i, j] / denominator_mu

                # Update variance of W_ij using averaged batch values
                numerator_var = avg_var_P_ij[i, j] - (avg_var_Z0[i] * updated_mu_W[i, j]**2)
                denominator_var = avg_mu_Z0[i]**2 + avg_var_Z0[i]

                if denominator_var < epsilon:
                    updated_var_W[i, j] = initial_var_W[i, j]
                else:
                    updated_var_W[i, j] = numerator_var / denominator_var
                
                # Ensure variance is positive
                updated_var_W[i, j] = np.maximum(updated_var_W[i, j], epsilon)
        
        # --- Modifications End Here ---

        return updated_mu_W, updated_var_W
    


    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs a forward pass through all layers to get a prediction.
        """
        activation_mean = x
        activation_var = np.zeros_like(x) 
        
        for layer in self.layers:
            activation_mean, activation_var = layer.forward(activation_mean, activation_var)
            
        return activation_mean, activation_var

    def update(self, x: np.ndarray, y: np.ndarray, obs_noise_var: float):
        """
        Performs a full forward and backward (smoothing) pass to update network parameters.
        """
        activation_means = [x]
        activation_vars = [np.zeros_like(x)]
        
        current_mean, current_var = x, np.zeros_like(x)
        for layer in self.layers:
            current_mean, current_var = layer.forward(current_mean, current_var)
            activation_means.append(current_mean)
            activation_vars.append(current_var)

        output_mean, output_var = activation_means[-1], activation_vars[-1]
        cov_yz = output_var
        total_output_var = output_var + obs_noise_var
        safe_total_output_var = np.where(total_output_var == 0, 1e-9, total_output_var)
        gain = np.divide(cov_yz, safe_total_output_var)

        delta_z_mean = output_mean + gain * (y - output_mean)
        delta_z_var = output_var - gain * cov_yz
        delta_z_var = np.maximum(delta_z_var, 1e-9)

        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            prev_a_mean_predicted = activation_means[i]
            prev_a_var_predicted = activation_vars[i]

            delta_a_mean, delta_a_var = layer.backward(
                prev_a_mean_predicted, 
                prev_a_var_predicted, 
                delta_z_mean, 
                delta_z_var
            )

            if i > 0:
                prev_layer = self.layers[i-1]
                z_mean_prev_predicted = prev_layer.pre_activation_mean
                z_var_prev_predicted = prev_layer.pre_activation_var
                J_prev = prev_layer.jacobian
                safe_prev_a_var_predicted = np.where(prev_a_var_predicted == 0, 1e-9, prev_a_var_predicted)
                gain_za = J_prev * np.divide(z_var_prev_predicted, safe_prev_a_var_predicted)
                
                delta_z_mean = z_mean_prev_predicted + gain_za * (delta_a_mean - prev_a_mean_predicted)
                delta_z_var = z_var_prev_predicted + gain_za**2 * (delta_a_var - prev_a_var_predicted)
                delta_z_var = np.maximum(delta_z_var, 1e-9)

    def standardize_weights(self, data_batch: np.ndarray, iterations: int = 1000):
        """
        Iteratively standardizes the weights of each layer in the network, keeping bias constant.
        Uses a provided batch of input data to derive initial mu_a and var_a.
        
        Args:
            data_batch (np.ndarray): A batch of input data. Shape: (batch_size, n_inputs)
            iterations (int): Number of iterations for internal standardization.
        """
        print("--- Starting weight standardization (BIAS IS CONSTANT) ---")
        
        # Start with the statistics of the provided input batch
        # mu_a_prev and var_a_prev must have shape (batch_size, n_inputs)
        mu_a_prev = data_batch
        var_a_prev = np.zeros_like(data_batch) + 1e-6

        for idx, layer in enumerate(self.layers):
            print(f"Standardizing Layer {idx}: {layer.weight_mean.shape[0]} inputs -> {layer.weight_mean.shape[1]} outputs")
            
            # Get the layer's parameters. The bias will be treated as constant during standardization.
            mu_W, var_W = layer.weight_mean, layer.weight_var
            # Ensure bias is treated correctly for broadcasting: (1, n_outputs)
            mu_B, var_B = layer.bias_mean, layer.bias_var 
            n_outputs = mu_W.shape[1]

            # 1. Forward pass to get current pre-activation (Z) stats for the batch
            # mu_Z, var_Z will have shape (batch_size, n_outputs)
            mu_Z, var_Z = self._calculate_pre_activation_properties(mu_a_prev, var_a_prev, mu_W, var_W, mu_B, var_B)
            
            # 2. Define standardization targets for this layer's output Z
            # These targets are for the *sum* and *sum of squares* of activations across units,
            # but they apply to *each item in the batch*.
            target_sum_mean = 0.0
            target_sum_var = float(n_outputs)
            target_sum_mean_sq = 2.0 * n_outputs
            target_sum_var_sq = 6.0 * n_outputs

            # 3. Iteratively update Z to meet S and S2 targets
            # updated_mu_Z, updated_var_Z will retain shape (batch_size, n_outputs)
            updated_mu_Z, updated_var_Z = mu_Z, var_Z
            for _ in range(iterations):
                updated_mu_Z, updated_var_Z = self._update_for_sum_target(updated_mu_Z, updated_var_Z, target_sum_mean, target_sum_var)
                updated_mu_Z, updated_var_Z = self._update_for_sum_target_2(updated_mu_Z, updated_var_Z, target_sum_mean_sq, target_sum_var_sq)

            # Check if the updated means and variances match the targets
            updated_mu_Z_sum = np.sum(updated_mu_Z, axis=1, keepdims=True) # (batch_size, 1)
            updated_var_Z_sum = np.sum(updated_var_Z, axis=1, keepdims=True) # (batch_size, 1)
            updated_mu_Z_sum_sq = np.sum(updated_mu_Z**2 + updated_var_Z, axis=1, keepdims=True) # (batch_size, 1)
            updated_var_Z_sum_sq = np.sum(2 * updated_var_Z**2 + 4 * updated_var_Z * updated_mu_Z**2, axis=1, keepdims=True) # (batch_size, 1)  

            print(f"Layer {idx}")
            print(f"Checking Z just after updates:")

            # Print the final means and variances for debugging
            print(f"Updated Z means (sum) mean: {np.mean(updated_mu_Z_sum)} | target: {target_sum_mean}")
            print(f"Updated Z variances (sum) mean: {np.mean(updated_var_Z_sum)} | target: {target_sum_var}")
            print(f"Updated Z means (sum of squares) mean: {np.mean(updated_mu_Z_sum_sq)} | target: {target_sum_mean_sq}")
            print(f"Updated Z variances (sum of squares) mean: {np.mean(updated_var_Z_sum_sq)} | target: {target_sum_var_sq}")

            
            # 4. Propagate updates back to W (B is constant)
            updated_mu_aW_target, updated_var_aW_target = self._update_Z0W_for_Z1_target(mu_B, var_B, updated_mu_Z, updated_var_Z)

            mu_Aw_B = updated_mu_aW_target + mu_B
            var_Aw_B = updated_var_aW_target + var_B
            mu_Aw_B_sum = np.sum(mu_Aw_B, axis=1, keepdims=True) # (batch_size, 1)
            var_Aw_B_sum = np.sum(var_Aw_B, axis=1, keepdims=True) # (batch_size, 1)
            mu_Aw_B_sum_sq = np.sum(mu_Aw_B**2 + var_Aw_B, axis=1, keepdims=True) # (batch_size, 1)
            var_Aw_B_sum_sq = np.sum(2 * var_Aw_B**2 + 4 * var_Aw_B * mu_Aw_B**2, axis=1, keepdims=True) # (batch_size, 1)
            print(f"After updating Z0W for Z1 targets:")
            print(f"mu_Aw_B means (sum) mean: {np.mean(mu_Aw_B_sum)} | target: {target_sum_mean}")
            print(f"var_Aw_B variances (sum) mean: {np.mean(var_Aw_B_sum)} | target: {target_sum_var}")
            print(f"mu_Aw_B means (sum of squares) mean: {np.mean(mu_Aw_B_sum_sq)} | target: {target_sum_mean_sq}")
            print(f"var_Aw_B variances (sum of squares) mean: {np.mean(var_Aw_B_sum_sq)} | target: {target_sum_var_sq}")

            # mu_P_ij, var_P_ij will have shape (batch_size, n_inputs, n_outputs)
            mu_P_ij, var_P_ij = self._rts_smooth_Z0W_components(
                mu_a_prev, var_a_prev, mu_W, var_W, updated_mu_aW_target, updated_var_aW_target
            )

            # Check Z with updated P_ij
            # P_ij represents the product terms A * W
            mu_Z_P_ij = np.sum(mu_P_ij, axis=1) + mu_B # (batch_size, n_outputs)
            var_Z_P_ij = np.sum(var_P_ij, axis=1) + var_B # (batch_size, n_outputs)

            # Check if the updated means and variances match the targets
            mu_Z_P_ij_sum = np.sum(mu_Z_P_ij, axis=1, keepdims=True) # (batch_size, 1)
            var_Z_P_ij_sum = np.sum(var_Z_P_ij, axis=1, keepdims=True) # (batch_size, 1)
            mu_Z_P_ij_sum_sq = np.sum(mu_Z_P_ij**2 + var_Z_P_ij, axis=1, keepdims=True) # (batch_size, 1)
            var_Z_P_ij_sum_sq = np.sum(2 * var_Z_P_ij**2 + 4 * var_Z_P_ij * mu_Z_P_ij**2, axis=1, keepdims=True) # (batch_size, 1)

            print(f"Layer {idx}")
            print(f"Checking Z with updated P_ij:")
            print(f"Updated Z means (sum) mean: {np.mean(mu_Z_P_ij_sum)} | target: {target_sum_mean}")
            print(f"Updated Z variances (sum) mean: {np.mean(var_Z_P_ij_sum)} | target: {target_sum_var}")
            print(f"Updated Z means (sum of squares) mean: {np.mean(mu_Z_P_ij_sum_sq)} | target: {target_sum_mean_sq}")
            print(f"Updated Z variances (sum of squares) mean: {np.mean(var_Z_P_ij_sum_sq)} | target: {target_sum_var_sq}")

            # This step is crucial: it averages the weight updates across the batch
            # updated_mu_W, updated_var_W will have shape (n_inputs, n_outputs)
            updated_mu_W, updated_var_W = self._update_W_from_Z0_and_Z0W_products(
                mu_a_prev, var_a_prev, mu_P_ij, var_P_ij, mu_W, var_W # Pass original W for gain calculation
            )

            # 6. Set the new, standardized weights for the layer. Bias remains unchanged.
            layer.weight_mean, layer.weight_var = updated_mu_W, updated_var_W

            # 7. Prepare the input for the *next* layer by applying the activation function
            # The resulting Z is based on the updated weights AND the original bias
            final_mu_Z, final_var_Z = self._calculate_pre_activation_properties(
                mu_a_prev, var_a_prev, updated_mu_W, updated_var_W, mu_B, var_B
            )

            final_sum_mean = np.sum(final_mu_Z, axis=1, keepdims=True) # (batch_size, 1)
            final_sum_var = np.sum(final_var_Z, axis=1, keepdims=True) # (batch_size, 1)
            final_sum_mean_sq = np.sum(final_mu_Z**2 + final_var_Z, axis=1, keepdims=True) # (batch_size, 1)
            final_sum_var_sq = np.sum(2 * final_var_Z**2 + 4 * final_var_Z * final_mu_Z**2, axis=1, keepdims=True) # (batch_size, 1)
            print(f"Layer {idx}")
            print(f"Final Z after standardization:")
            print(f"Final Z means (sum) mean: {np.mean(final_sum_mean)} | target: {target_sum_mean}")
            print(f"Final Z variances (sum) mean: {np.mean(final_sum_var)} | target: {target_sum_var}")
            print(f"Final Z means (sum of squares) mean: {np.mean(final_sum_mean_sq)} | target: {target_sum_mean_sq}")
            print(f"Final Z variances (sum of squares) mean: {np.mean(final_sum_var_sq)} | target: {target_sum_var_sq}")
            
            if not layer.is_output_layer:
                mu_a_prev, var_a_prev, _ = layer.mReLU(final_mu_Z, final_var_Z)
            else: # Output layer is linear
                mu_a_prev = layer.weight_mean 
                var_a_prev = layer.weight_var
        print("--- Weight standardization complete. ---")

    def verify_standardization(self, data_batch: np.ndarray = None):
        """
        Verifies that the standardized weights produce the desired pre-activation stats.
        This should be called immediately after standardize_weights().
        
        Args:
            data_batch (np.ndarray): The batch of input data used for standardization.
                                    If None, it defaults to a batch of zeros matching the input layer.
        """
        print("\n--- Verifying Weight Standardization ---")
        
        # Determine mu_a_prev and var_a_prev based on whether a batch was provided
        if data_batch is not None:
            mu_a_prev = data_batch
            var_a_prev = np.zeros_like(data_batch) + 1e-6
        else:
            # Fallback for verification if no specific batch is given (should match default in standardize_weights)
            # This needs to be consistent with how standardize_weights initializes mu_a_prev.
            # If standardize_weights takes `input_activation` that defaults to zeros, this should too.
            # Assuming it takes `data_batch` now. If data_batch is None here, we need a sensible default.
            # Let's create a batch of zeros matching the first layer's input size and batch size of 1.
            n_inputs_first_layer = self.layers[0].weight_mean.shape[0]
            mu_a_prev = np.zeros((1, n_inputs_first_layer))
            var_a_prev = np.zeros_like(mu_a_prev)

        for idx, layer in enumerate(self.layers):
            print(f"\nVerifying Layer {idx}: {layer.weight_mean.shape[0]} inputs -> {layer.weight_mean.shape[1]} outputs")
            mu_W, var_W = layer.weight_mean, layer.weight_var
            # Ensure bias is treated correctly for broadcasting: (1, n_outputs)
            mu_B, var_B = layer.bias_mean, layer.bias_var 
            n_outputs = mu_W.shape[1]

            # 1. Calculate the actual pre-activation (Z) stats with the standardized weights
            # actual_mu_Z, actual_var_Z will have shape (batch_size, n_outputs)
            actual_mu_Z, actual_var_Z = self._calculate_pre_activation_properties(
                mu_a_prev, var_a_prev, mu_W, var_W, mu_B, var_B
            )

            # 2. Define the original targets for comparison (per-unit targets)
            # These are scalar targets for the sum/sum_sq across *units* for *each* batch item.
            target_sum_mean = 0.0
            target_sum_var = float(n_outputs)
            target_sum_mean_sq = 2.0 * n_outputs
            target_sum_var_sq = 6.0 * n_outputs

            # 3. Calculate the actual stats of the sums (S and S2) for each batch item
            # Then average these actual sums over the batch for reporting.
            
            # Stats for S = sum(Z_i)
            actual_sum_mean_per_batch = np.sum(actual_mu_Z, axis=1) # (batch_size,)
            actual_sum_var_per_batch = np.sum(actual_var_Z, axis=1) # (batch_size,)

            # Stats for S2 = sum(Z_i^2)
            actual_sum_mean_sq_per_batch = np.sum(actual_mu_Z**2 + actual_var_Z, axis=1) # (batch_size,)
            actual_sum_var_sq_per_batch = np.sum(2 * actual_var_Z**2 + 4 * actual_var_Z * actual_mu_Z**2, axis=1) # (batch_size,)

            # Report the average over the batch
            print(f"  Mean of Sum (S) (avg over batch):       Actual={np.mean(actual_sum_mean_per_batch): .4f} (Target: {target_sum_mean:.4f})")
            print(f"  Variance of Sum (S) (avg over batch):   Actual={np.mean(actual_sum_var_per_batch):.4f} (Target: {target_sum_var:.4f})")
            print(f"  Mean of Sum^2 (S2) (avg over batch):    Actual={np.mean(actual_sum_mean_sq_per_batch):.4f} (Target: {target_sum_mean_sq:.4f})")
            print(f"  Variance of Sum^2 (S2) (avg over batch): Actual={np.mean(actual_sum_var_sq_per_batch):.4f} (Target: {target_sum_var_sq:.4f})")

            # 5. Propagate the activation to prepare for the next layer
            if not layer.is_output_layer:
                mu_a_prev, var_a_prev, _ = layer.mReLU(actual_mu_Z, actual_var_Z)
            else:
                mu_a_prev = actual_mu_Z
                var_a_prev = actual_var_Z
        print("\n--- Verification Complete ---")


def main():
    """Main function to run the TAGI network training and evaluation on MNIST."""
    np.random.seed(42)
    NEW_IMAGE_RESOLUTION = 14 
    input_size = NEW_IMAGE_RESOLUTION * NEW_IMAGE_RESOLUTION * 1 
    num_classes = 10
    UNITS = [input_size, 64, 64, 64, 64, 64, 64, 64, num_classes] 
    EPOCHS = 10 
    BATCH_SIZE = 5000 # Changed to 64 for demonstration of larger batch
    OBS_NOISE_VAR = 0.01 

    print("Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    print(f"Resizing images to {NEW_IMAGE_RESOLUTION}x{NEW_IMAGE_RESOLUTION}...")
    x_train_resized = np.array([resize(img, (NEW_IMAGE_RESOLUTION, NEW_IMAGE_RESOLUTION), anti_aliasing=True) for img in x_train], dtype=np.float32)
    x_test_resized = np.array([resize(img, (NEW_IMAGE_RESOLUTION, NEW_IMAGE_RESOLUTION), anti_aliasing=True) for img in x_test], dtype=np.float32)

    x_train_flattened = x_train_resized.reshape(x_train_resized.shape[0], -1)
    x_test_flattened = x_test_resized.reshape(x_test_resized.shape[0], -1)

    y_train_one_hot = to_categorical(y_train, num_classes)
    y_test_one_hot = to_categorical(y_test, num_classes)

    print(f"Flattened x_train shape: {x_train_flattened.shape}, y_train_one_hot shape: {y_train_one_hot.shape}")

    # --- Model Initialization and Standardization (MODIFIED) ---
    model = TAGINetwork(UNITS)
    
    # NEW: Select a representative batch for standardization
    # For a big batch, you could select a random subset of the training data.
    # Here, we'll just take the first BATCH_SIZE samples for simplicity.
    # In a real scenario, you might want to randomly sample this batch each time standardization runs,
    # or use a fixed, representative validation set.
    initialization_batch = x_train_flattened[:BATCH_SIZE]

    print(f"\nUsing a batch of size {BATCH_SIZE} from training data for weight standardization.")
    # model.standardize_weights(data_batch=initialization_batch)

    # Verify standardization with the same batch
    model.verify_standardization(data_batch=initialization_batch)

    BATCH_SIZE = 32

    print("\nStarting training...")
    n_train_samples = x_train_flattened.shape[0]
    
    for epoch in range(EPOCHS):
        indices = np.random.permutation(n_train_samples)
        for i in range(0, n_train_samples, BATCH_SIZE):
            batch_indices = indices[i:i + BATCH_SIZE]
            x_batch = x_train_flattened[batch_indices]
            y_batch = y_train_one_hot[batch_indices]
            model.update(x_batch, y_batch, OBS_NOISE_VAR)
        
        y_pred_mean, _ = model.predict(x_test_flattened)
        y_pred_classes = np.argmax(y_pred_mean, axis=1)
        accuracy = np.mean(y_pred_classes == y_test.flatten())
        
        print(f"Epoch {epoch + 1}/{EPOCHS} complete. Test Accuracy: {accuracy:.4f}")

    print("\nTraining complete.")
    
    y_pred_mean_test, _ = model.predict(x_test_flattened)
    y_pred_classes_final = np.argmax(y_pred_mean_test, axis=1)
    final_accuracy = np.mean(y_pred_classes_final == y_test.flatten())
    print(f"Final Test Accuracy: {final_accuracy:.4f}")

    class_names = [str(i) for i in range(10)]
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        img = x_test_resized[i] 
        plt.imshow(img, cmap='gray')
        true_label = y_test[i]
        pred_label = y_pred_classes_final[i]
        color = 'green' if true_label == pred_label else 'red'
        plt.xlabel(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}", color=color, fontsize=8)
    plt.suptitle(f"MNIST Predictions at {NEW_IMAGE_RESOLUTION}x{NEW_IMAGE_RESOLUTION} (Green: Correct, Red: Incorrect)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    main()