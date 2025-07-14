# tagi_network.py

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from skimage.transform import resize
from scipy.stats import norm

# --- DataManager and TAGILayer Classes (No Changes) ---
# (The DataManager and TAGILayer classes remain exactly the same as in your provided file)
class DataManager:
    """Handles data creation, normalization, and splitting."""
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std
    def create_cubic_data(self, n_samples: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        x = np.linspace(-5, 5, n_samples).reshape(-1, 1)
        y = x**3 + np.random.randn(n_samples, 1) * 3
        return x, y
    def normalize(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std
    def denormalize(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean
    @staticmethod
    def split_data(x: np.ndarray, y: np.ndarray, train_frac: float = 0.8) -> tuple:
        n_samples = x.shape[0]
        indices = np.random.permutation(n_samples)
        n_train = int(n_samples * train_frac)
        train_idx, test_idx = indices[:n_train], indices[n_train:]
        return x[train_idx], y[train_idx], x[test_idx], y[test_idx]

class TAGILayer:
    """Represents a single layer in a Tractable Approximate Gaussian Inference (TAGI) network."""
    def __init__(self, n_inputs: int, n_outputs: int, is_output_layer: bool = False):
        self.is_output_layer = is_output_layer
        init_var = 0.5 / n_inputs
        init_std = np.sqrt(init_var)
        self.weight_mean = np.random.randn(n_inputs, n_outputs) * init_std
        self.weight_var = np.full((n_inputs, n_outputs), init_var)
        self.bias_mean = np.zeros((1, n_outputs))
        self.bias_var = np.full((1, n_outputs), 0.5)
        self.pre_activation_mean = None
        self.pre_activation_var = None
        self.jacobian = None
    def forward(self, prev_activation_mean: np.ndarray, prev_activation_var: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.pre_activation_mean = prev_activation_mean @ self.weight_mean + self.bias_mean
        self.pre_activation_var = ((prev_activation_mean**2 @ self.weight_var) + (prev_activation_var @ self.weight_mean**2) + (prev_activation_var @ self.weight_var) + self.bias_var)
        if self.is_output_layer:
            activation_mean, activation_var, self.jacobian = self.pre_activation_mean, self.pre_activation_var, np.ones_like(self.pre_activation_mean)
        else:
            activation_mean, activation_var, self.jacobian = self.mReLU(self.pre_activation_mean, self.pre_activation_var)
        return activation_mean, activation_var
    def backward(self, prev_activation_mean: np.ndarray, prev_activation_var: np.ndarray, next_delta_mean: np.ndarray, next_delta_var: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch_size, n_inputs, n_outputs = prev_activation_mean.shape[0], self.weight_mean.shape[0], self.weight_mean.shape[1]
        cov_z_w = np.expand_dims(prev_activation_mean, axis=2) * np.expand_dims(self.weight_var, axis=0)
        cov_z_b = np.tile(np.expand_dims(self.bias_var, axis=0), (batch_size, 1, 1))
        safe_pre_activation_var = np.where(self.pre_activation_var == 0, 1e-9, self.pre_activation_var)
        expanded_safe_pre_activation_var = np.expand_dims(safe_pre_activation_var, axis=1)
        gain_w = np.divide(cov_z_w, expanded_safe_pre_activation_var, out=np.zeros((batch_size, n_inputs, n_outputs)))
        gain_b = np.divide(cov_z_b, expanded_safe_pre_activation_var, out=np.zeros((batch_size, 1, n_outputs)))
        err_mean = next_delta_mean - self.pre_activation_mean
        err_var_term = next_delta_var - self.pre_activation_var
        self.weight_mean += np.mean(gain_w * np.expand_dims(err_mean, axis=1), axis=0)
        self.weight_var += np.mean(gain_w**2 * np.expand_dims(err_var_term, axis=1), axis=0)
        self.weight_var = np.maximum(self.weight_var, 1e-9)
        self.bias_mean += np.mean(gain_b * np.expand_dims(err_mean, axis=1), axis=0).reshape(1, n_outputs)
        self.bias_var += np.mean(gain_b**2 * np.expand_dims(err_var_term, axis=1), axis=0).reshape(1, n_outputs)
        self.bias_var = np.maximum(self.bias_var, 1e-9)
        gain_numerator = np.expand_dims(self.weight_mean, axis=0) * np.expand_dims(prev_activation_var, axis=2)
        gain_matrix = np.divide(gain_numerator, expanded_safe_pre_activation_var, out=np.zeros((batch_size, n_inputs, n_outputs)))
        correction_mean = np.sum(err_mean[:, np.newaxis, :] * gain_matrix, axis=2)
        delta_mean_prev = prev_activation_mean + correction_mean
        err_var_term_diag = np.zeros((batch_size, n_outputs, n_outputs))
        for k in range(batch_size): err_var_term_diag[k, :, :] = np.diag(-err_var_term[k, :])
        var_correction_matrix = np.matmul(np.matmul(gain_matrix, err_var_term_diag), np.transpose(gain_matrix, (0, 2, 1)))
        delta_var_prev = prev_activation_var - np.diagonal(var_correction_matrix, axis1=1, axis2=2)
        return delta_mean_prev, np.maximum(delta_var_prev, 1e-9)
    def mReLU(self, mean: np.ndarray, var: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        safe_var = np.where(var < 0, 1e-6, var)
        z_mean, z_std = mean, np.maximum(np.sqrt(safe_var), 1e-9)
        cdfn = np.maximum(norm.cdf(z_mean / z_std), 1E-20)
        pdfn = np.maximum(norm.pdf(z_mean / z_std), 1E-20)
        relu_mean = np.maximum(z_std * pdfn + z_mean * cdfn, 1E-20)
        relu_var = np.maximum(-relu_mean * relu_mean + 2 * relu_mean * z_mean - z_mean * z_std * pdfn + (safe_var - z_mean * z_mean) * cdfn, 1E-9)
        return relu_mean, relu_var, cdfn

# --- TAGINetwork Class (MODIFIED) ---
class TAGINetwork:
    """A Bayesian Neural Network using Tractable Approximate Gaussian Inference."""
    def __init__(self, layer_units: List[int]):
        self.layers = []
        for i in range(len(layer_units) - 1):
            is_output = (i == len(layer_units) - 2)
            self.layers.append(TAGILayer(layer_units[i], layer_units[i+1], is_output_layer=is_output))

    # --- All static helper methods (_calculate_pre_activation_properties, etc.) remain unchanged ---
    @staticmethod
    def _calculate_pre_activation_properties(mu_Z0, var_Z0, mu_W, var_W, mu_B, var_B):
        predicted_mu_Z1 = mu_Z0 @ mu_W + mu_B
        predicted_var_Z1 = ((mu_Z0**2 @ var_W) + (var_Z0 @ mu_W**2) + (var_Z0 @ var_W) + var_B)
        return predicted_mu_Z1, predicted_var_Z1
    @staticmethod
    def _update_for_sum_target(means, variances, target_mean, target_var):
        mu_S, sigma2_S = np.sum(means, axis=1, keepdims=True), np.sum(variances, axis=1, keepdims=True)
        epsilon = 1e-9
        Jz = variances / (sigma2_S + epsilon)
        updated_means = means + Jz * (target_mean - mu_S)
        updated_variances = variances + Jz * (target_var - sigma2_S)
        return updated_means, np.maximum(updated_variances, epsilon)
    @staticmethod
    def _update_for_sum_target_2(means, variances, target_mean_2, target_var_2):
        mu_Z_sq, var_Z_sq = means**2 + variances, 2 * variances**2 + 4 * variances * means**2
        mu_S2, sigma2_S2 = np.sum(mu_Z_sq, axis=1, keepdims=True), np.sum(var_Z_sq, axis=1, keepdims=True)
        epsilon = 1e-9
        Jz = (2 * means * variances) / (sigma2_S2 + epsilon)
        updated_means = means + Jz * (target_mean_2 - mu_S2)
        updated_variances = variances + Jz**2 * (target_var_2 - sigma2_S2)
        return updated_means, np.maximum(updated_variances, epsilon)
    @staticmethod
    def _update_Z0W_for_Z1_target(mu_B, var_B, target_mu_Z1, target_var_Z1):
        updated_var_Z0W = np.maximum(target_var_Z1 - var_B, 1e-9)
        updated_mu_Z0W = target_mu_Z1 - mu_B
        return updated_mu_Z0W, updated_var_Z0W
    @staticmethod
    def _rts_smooth_Z0W_components(initial_mu_Z0, initial_var_Z0, initial_mu_W, initial_var_W, updated_mu_Z0W_sum_target, updated_var_Z0W_sum_target):
        epsilon = 1e-9
        initial_mu_P_ij = initial_mu_Z0[:, :, np.newaxis] * initial_mu_W[np.newaxis, :, :]
        initial_var_P_ij = ((initial_var_Z0[:, :, np.newaxis] * initial_var_W[np.newaxis, :, :]) + (initial_var_Z0[:, :, np.newaxis] * initial_mu_W[np.newaxis, :, :]**2) + (initial_var_W[np.newaxis, :, :] * initial_mu_Z0[:, :, np.newaxis]**2))
        updated_mu_P_ij, updated_var_P_ij = np.copy(initial_mu_P_ij), np.copy(initial_var_P_ij)
        current_mu_Z0W, current_var_Z0W = np.sum(updated_mu_P_ij, axis=1), np.sum(updated_var_P_ij, axis=1)
        error_mu_sum, error_var_sum = updated_mu_Z0W_sum_target - current_mu_Z0W, updated_var_Z0W_sum_target - current_var_Z0W
        gain = updated_var_P_ij / (current_var_Z0W[:, np.newaxis, :] + epsilon)
        updated_mu_P_ij += gain * error_mu_sum[:, np.newaxis, :]
        updated_var_P_ij += gain * error_var_sum[:, np.newaxis, :]
        return updated_mu_P_ij, np.maximum(updated_var_P_ij, epsilon)
    @staticmethod
    def _update_W_from_Z0_and_Z0W_products(mu_Z0, var_Z0, mu_P_ij, var_P_ij, initial_mu_W, initial_var_W):
        epsilon = 1e-6
        avg_mu_Z0, avg_var_Z0 = np.mean(mu_Z0, axis=0), np.mean(var_Z0, axis=0)
        avg_mu_P_ij, avg_var_P_ij = np.mean(mu_P_ij, axis=0), np.mean(var_P_ij, axis=0)
        updated_mu_W, updated_var_W = np.zeros_like(initial_mu_W), np.zeros_like(initial_var_W)
        for i in range(mu_Z0.shape[1]):
            for j in range(mu_P_ij.shape[2]):
                if np.abs(avg_mu_Z0[i]) < epsilon: updated_mu_W[i, j] = initial_mu_W[i, j]
                else: updated_mu_W[i, j] = avg_mu_P_ij[i, j] / avg_mu_Z0[i]
                numerator_var = avg_var_P_ij[i, j] - (avg_var_Z0[i] * updated_mu_W[i, j]**2)
                denominator_var = avg_mu_Z0[i]**2 + avg_var_Z0[i]
                if denominator_var < epsilon: updated_var_W[i, j] = initial_var_W[i, j]
                else: updated_var_W[i, j] = numerator_var / denominator_var
                updated_var_W[i, j] = np.maximum(updated_var_W[i, j], epsilon)
        return updated_mu_W, updated_var_W

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # (The predict function remains exactly the same)
        activation_mean, activation_var = x, np.zeros_like(x)
        for layer in self.layers:
            activation_mean, activation_var = layer.forward(activation_mean, activation_var)
        return activation_mean, activation_var

    def update(self, x: np.ndarray, y: np.ndarray, obs_noise_var: float):
        # (The update function remains exactly the same)
        activation_means, activation_vars = [x], [np.zeros_like(x)]
        current_mean, current_var = x, np.zeros_like(x)
        for layer in self.layers:
            current_mean, current_var = layer.forward(current_mean, current_var)
            activation_means.append(current_mean)
            activation_vars.append(current_var)
        output_mean, output_var = activation_means[-1], activation_vars[-1]
        safe_total_output_var = np.where(output_var + obs_noise_var == 0, 1e-9, output_var + obs_noise_var)
        gain = np.divide(output_var, safe_total_output_var)
        delta_z_mean = output_mean + gain * (y - output_mean)
        delta_z_var = np.maximum(output_var - gain * output_var, 1e-9)
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            prev_a_mean_predicted, prev_a_var_predicted = activation_means[i], activation_vars[i]
            delta_a_mean, delta_a_var = layer.backward(prev_a_mean_predicted, prev_a_var_predicted, delta_z_mean, delta_z_var)
            if i > 0:
                prev_layer = self.layers[i-1]
                z_mean_prev_predicted, z_var_prev_predicted, J_prev = prev_layer.pre_activation_mean, prev_layer.pre_activation_var, prev_layer.jacobian
                safe_prev_a_var_predicted = np.where(prev_a_var_predicted == 0, 1e-9, prev_a_var_predicted)
                gain_za = J_prev * np.divide(z_var_prev_predicted, safe_prev_a_var_predicted)
                delta_z_mean = z_mean_prev_predicted + gain_za * (delta_a_mean - prev_a_mean_predicted)
                delta_z_var = np.maximum(z_var_prev_predicted + gain_za**2 * (delta_a_var - prev_a_var_predicted), 1e-9)

    def standardize_weights(self, data_batch: np.ndarray, sigma_m_sq: float, sigma_z_sq: float, iterations: int = 100):
        """
        Iteratively standardizes weights based on target moment parameters.
        Args:
            data_batch (np.ndarray): A batch of input data.
            sigma_m_sq (float): The hyperparameter sigma_M^2 from the formula.
            sigma_z_sq (float): The hyperparameter sigma_Z^2 from the formula.
            iterations (int): Number of iterations for internal standardization.
        """
        # print("--- Starting weight standardization (BIAS IS CONSTANT) ---")
        mu_a_prev, var_a_prev = data_batch, np.zeros_like(data_batch) + 1e-6

        for idx, layer in enumerate(self.layers):
            mu_W, var_W = layer.weight_mean, layer.weight_var
            mu_B, var_B = layer.bias_mean, layer.bias_var
            n_outputs = mu_W.shape[1]

            mu_Z, var_Z = self._calculate_pre_activation_properties(mu_a_prev, var_a_prev, mu_W, var_W, mu_B, var_B)

            # --- MODIFICATION: Calculate targets from hyperparameters ---
            # A = n_outputs
            target_sum_mean = 0.0
            target_sum_var = n_outputs * sigma_z_sq
            target_sum_mean_sq = n_outputs * (sigma_m_sq + sigma_z_sq)
            target_sum_var_sq = n_outputs * (2 * (sigma_z_sq**2) + 4 * sigma_m_sq * sigma_z_sq)
            # --- END MODIFICATION ---

            updated_mu_Z, updated_var_Z = mu_Z, var_Z
            for _ in range(iterations):
                updated_mu_Z, updated_var_Z = self._update_for_sum_target(updated_mu_Z, updated_var_Z, target_sum_mean, target_sum_var)
                updated_mu_Z, updated_var_Z = self._update_for_sum_target_2(updated_mu_Z, updated_var_Z, target_sum_mean_sq, target_sum_var_sq)
            
            updated_mu_aW_target, updated_var_aW_target = self._update_Z0W_for_Z1_target(mu_B, var_B, updated_mu_Z, updated_var_Z)
            
            mu_P_ij, var_P_ij = self._rts_smooth_Z0W_components(mu_a_prev, var_a_prev, mu_W, var_W, updated_mu_aW_target, updated_var_aW_target)
            
            updated_mu_W, updated_var_W = self._update_W_from_Z0_and_Z0W_products(mu_a_prev, var_a_prev, mu_P_ij, var_P_ij, mu_W, var_W)
            
            layer.weight_mean, layer.weight_var = updated_mu_W, updated_var_W
            
            final_mu_Z, final_var_Z = self._calculate_pre_activation_properties(mu_a_prev, var_a_prev, updated_mu_W, updated_var_W, mu_B, var_B)
            
            if not layer.is_output_layer:
                mu_a_prev, var_a_prev, _ = layer.mReLU(final_mu_Z, final_var_Z)
            else:
                mu_a_prev, var_a_prev = final_mu_Z, final_var_Z

def run_training_session(config: dict) -> float:
    """
    Runs a complete training and evaluation session based on a configuration dictionary.
    Args:
        config (dict): A dictionary containing all hyperparameters for the run.
    Returns:
        float: The final test accuracy.
    """
    np.random.seed(42)
    # --- Unpack configuration ---
    sigma_m_sq = config['sigma_m_sq']
    sigma_z_sq = config['sigma_z_sq']
    epochs = config['epochs']
    batch_size = config['batch_size']
    init_batch_size = config['init_batch_size']
    
    # --- Data Loading and Preparation ---
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0

    NEW_IMAGE_RESOLUTION = 14
    x_train_resized = np.array([resize(img, (NEW_IMAGE_RESOLUTION, NEW_IMAGE_RESOLUTION), anti_aliasing=True) for img in x_train], dtype=np.float32)
    x_test_resized = np.array([resize(img, (NEW_IMAGE_RESOLUTION, NEW_IMAGE_RESOLUTION), anti_aliasing=True) for img in x_test], dtype=np.float32)

    x_train_flattened = x_train_resized.reshape(x_train_resized.shape[0], -1)
    x_test_flattened = x_test_resized.reshape(x_test_resized.shape[0], -1)

    num_classes = 10
    y_train_one_hot = to_categorical(y_train, num_classes)

    # --- Model Initialization and Standardization ---
    UNITS = [x_train_flattened.shape[1], 64, 64, num_classes]
    model = TAGINetwork(UNITS)
    
    initialization_batch = x_train_flattened[:init_batch_size]
    model.standardize_weights(data_batch=initialization_batch, sigma_m_sq=sigma_m_sq, sigma_z_sq=sigma_z_sq)

    # --- Training Loop ---
    n_train_samples = x_train_flattened.shape[0]
    OBS_NOISE_VAR = 0.01
    
    for epoch in range(epochs):
        indices = np.random.permutation(n_train_samples)
        for i in range(0, n_train_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            x_batch, y_batch = x_train_flattened[batch_indices], y_train_one_hot[batch_indices]
            model.update(x_batch, y_batch, OBS_NOISE_VAR)
    
    # --- Final Evaluation ---
    y_pred_mean, _ = model.predict(x_test_flattened)
    y_pred_classes = np.argmax(y_pred_mean, axis=1)
    accuracy = np.mean(y_pred_classes == y_test.flatten())
    
    return accuracy