import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from tensorflow.keras.datasets import cifar10 # For loading CIFAR-10
from tensorflow.keras.utils import to_categorical # For one-hot encoding labels
from skimage.transform import resize # For resizing images

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

    def create_cubic_data(self, n_samples: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Generates noisy data from a cubic function (no longer used for CIFAR-10)."""
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
        """Randomly splits data into training and testing sets (can be adapted for CIFAR)."""
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
        # For matrix multiplication with batch dimension, @ operator handles it correctly if weights are 2D.
        self.pre_activation_mean = prev_activation_mean @ self.weight_mean + self.bias_mean
        
        # Variance propagation for batches:
        # Each term needs to broadcast correctly across the batch dimension.
        self.pre_activation_var = (
            (prev_activation_mean**2 @ self.weight_var) +
            (prev_activation_var @ self.weight_mean**2) +
            (prev_activation_var @ self.weight_var) + 
            self.bias_var
        )

        # (2) Apply activation function (ReLU or identity for the output layer)
        if self.is_output_layer:
            # For classification, we treat the output as logits directly.
            # No non-linearity here, Softmax will be applied externally if needed.
            activation_mean = self.pre_activation_mean
            activation_var = self.pre_activation_var
            self.jacobian = np.ones_like(self.pre_activation_mean)
        else:
            # ReLU activation
            is_positive = self.pre_activation_mean > 0
            activation_mean = self.pre_activation_mean * is_positive
            activation_var = self.pre_activation_var * is_positive
            self.jacobian = 1.0 * is_positive # Jacobians also have batch dimension

        return activation_mean, activation_var

    def backward(self, prev_activation_mean: np.ndarray, prev_activation_var: np.ndarray, 
                 next_delta_mean: np.ndarray, next_delta_var: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs the backward smoothing pass to update layer parameters and propagate deltas,
        compatible with batch sizes.
        
        Args:
            prev_activation_mean (np.ndarray): Mean of activations from the previous layer (input to current layer). Shape: (batch_size, n_inputs)
            prev_activation_var (np.ndarray): Variance of activations from the previous layer. Shape: (batch_size, n_inputs)
            next_delta_mean (np.ndarray): Smoothed mean of the current layer's PRE-ACTIVATION. Shape: (batch_size, n_outputs)
            next_delta_var (np.ndarray): Smoothed variance of the current layer's PRE-ACTIVATION. Shape: (batch_size, n_outputs)
            
        Returns:
            A tuple containing the smoothed mean and variance for the previous layer's POST-ACTIVATION.
            Shape: (batch_size, n_inputs)
        """
        batch_size = prev_activation_mean.shape[0]
        n_inputs = prev_activation_mean.shape[1]
        n_outputs = self.weight_mean.shape[1]

        # (1) Update weights and biases using Kalman filter-like updates
        # Covariances needed for Kalman gain calculation. Need to broadcast correctly.
        # cov_z_w shape: (batch_size, n_inputs, n_outputs)
        # prev_activation_mean: (batch_size, n_inputs)
        # self.weight_var: (n_inputs, n_outputs)
        cov_z_w = np.expand_dims(prev_activation_mean, axis=2) * np.expand_dims(self.weight_var, axis=0) 
        
        # cov_z_b should be (batch_size, 1, n_outputs) for broadcasting with err_mean
        cov_z_b = np.tile(np.expand_dims(self.bias_var, axis=0), (batch_size, 1, 1))

        # Kalman gains for weights and biases (ensure no division by zero)
        # pre_activation_var needs to be expanded to (batch_size, 1, n_outputs) for broadcasting
        safe_pre_activation_var = np.where(self.pre_activation_var == 0, 1e-9, self.pre_activation_var)
        expanded_safe_pre_activation_var = np.expand_dims(safe_pre_activation_var, axis=1) # (batch_size, 1, n_outputs)
        
        # Ensure 'out' array has the correct broadcast shape for numpy.divide
        gain_w_out_shape = (batch_size, n_inputs, n_outputs)
        gain_b_out_shape = (batch_size, 1, n_outputs)

        gain_w = np.divide(cov_z_w, expanded_safe_pre_activation_var, out=np.zeros(gain_w_out_shape))
        gain_b = np.divide(cov_z_b, expanded_safe_pre_activation_var, out=np.zeros(gain_b_out_shape)) 
        
        err_mean = next_delta_mean - self.pre_activation_mean # Shape: (batch_size, n_outputs)
        err_var_term = self.pre_activation_var - next_delta_var # Shape: (batch_size, n_outputs)

        # Update weight parameters (mean over batch for shared parameters)
        # err_mean: (batch_size, n_outputs) needs to be (batch_size, 1, n_outputs) for element-wise mult
        self.weight_mean += np.mean(gain_w * np.expand_dims(err_mean, axis=1), axis=0)
        self.weight_var -= np.mean(gain_w**2 * np.expand_dims(err_var_term, axis=1), axis=0)
        self.weight_var = np.maximum(self.weight_var, 1e-9) # Prevent variance from becoming negative
        
        # Update bias parameters (mean over batch for shared parameters)
        self.bias_mean += np.mean(gain_b * np.expand_dims(err_mean, axis=1), axis=0).reshape(1, n_outputs)
        self.bias_var -= np.mean(gain_b**2 * np.expand_dims(err_var_term, axis=1), axis=0).reshape(1, n_outputs)
        self.bias_var = np.maximum(self.bias_var, 1e-9) # Prevent variance from becoming negative

        # (2) Propagate deltas to the previous layer to get the smoothed POST-ACTIVATION (delta_a)
        # Calculate Gain matrix for propagating deltas state-to-state
        # Gain[j,k] = Cov(a_prev[j], z_curr[k]) / Var(z_curr[k])
        # This gain is per sample in the batch
        # self.weight_mean: (n_inputs, n_outputs)
        # prev_activation_var: (batch_size, n_inputs)
        # pre_activation_var: (batch_size, n_outputs)
        
        # Expand dims for broadcasting
        # gain_numerator shape: (batch_size, n_inputs, n_outputs)
        gain_numerator = np.expand_dims(self.weight_mean, axis=0) * np.expand_dims(prev_activation_var, axis=2)
        gain_matrix_out_shape = (batch_size, n_inputs, n_outputs)
        gain_matrix = np.divide(gain_numerator, expanded_safe_pre_activation_var, out=np.zeros(gain_matrix_out_shape))

        # Propagate mean delta.
        # err_mean: (batch_size, n_outputs)
        # gain_matrix: (batch_size, n_inputs, n_outputs)
        # Result should be (batch_size, n_inputs)
        correction_mean = np.sum(err_mean[:, np.newaxis, :] * gain_matrix, axis=2)
        delta_mean_prev = prev_activation_mean + correction_mean

        # Propagate variance delta
        # err_var_term: (batch_size, n_outputs)
        # var_correction_matrix should be (batch_size, n_inputs, n_inputs)
        # Create a batch of diagonal matrices from err_var_term
        err_var_term_diag = np.zeros((batch_size, n_outputs, n_outputs))
        for k in range(batch_size):
            err_var_term_diag[k, :, :] = np.diag(err_var_term[k, :])

        # Ensure correct matrix multiplication order for batch dimensions
        # (batch_size, n_inputs, n_outputs) @ (batch_size, n_outputs, n_outputs) @ (batch_size, n_outputs, n_inputs)
        var_correction_matrix = np.matmul(np.matmul(gain_matrix, err_var_term_diag), np.transpose(gain_matrix, (0, 2, 1)))
        
        # Extract diagonal for variance
        delta_var_prev = prev_activation_var - np.diagonal(var_correction_matrix, axis1=1, axis2=2)
        delta_var_prev = np.maximum(delta_var_prev, 1e-9)

        return delta_mean_prev, delta_var_prev


# --- 3. Main Network Class ---
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
            is_output = (i == len(layer_units) - 2)
            self.layers.append(TAGILayer(layer_units[i], layer_units[i+1], is_output_layer=is_output))
    
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs a forward pass through all layers to get a prediction.
        Compatible with batch sizes.
        """
        activation_mean = x
        activation_var = np.zeros_like(x) # Initial input variance is zero
        
        for layer in self.layers:
            activation_mean, activation_var = layer.forward(activation_mean, activation_var)
            
        return activation_mean, activation_var

    def update(self, x: np.ndarray, y: np.ndarray, obs_noise_var: float):
        """
        Performs a full forward and backward (smoothing) pass to update network parameters,
        compatible with batch sizes.
        
        Args:
            x (np.ndarray): Input batch. Shape: (batch_size, n_inputs)
            y (np.ndarray): Target batch (one-hot encoded for CIFAR-10). Shape: (batch_size, n_outputs)
            obs_noise_var (float): Variance of the observation noise.
        """
        # --- Forward Pass: Store activations for use in the backward pass ---
        # Note: Initial input variance is assumed to be zero
        activation_means = [x]
        activation_vars = [np.zeros_like(x)]
        
        current_mean, current_var = x, np.zeros_like(x)
        for layer in self.layers:
            current_mean, current_var = layer.forward(current_mean, current_var)
            activation_means.append(current_mean)
            activation_vars.append(current_var)

        # --- Backward Pass (Smoothing) ---
        # Initialize smoothed state delta at the output layer based on the observation y
        output_mean, output_var = activation_means[-1], activation_vars[-1]
        
        cov_yz = output_var
        total_output_var = output_var + obs_noise_var # For classification, obs_noise_var acts as a temperature/scaling factor

        # Ensure no division by zero when calculating gain for output layer
        safe_total_output_var = np.where(total_output_var == 0, 1e-9, total_output_var)
        gain = np.divide(cov_yz, safe_total_output_var)

        # Start with the smoothed PRE-ACTIVATION (z) of the final layer
        # For classification, (y - output_mean) represents the error in logits/scores
        delta_z_mean = output_mean + gain * (y - output_mean)
        delta_z_var = output_var - gain * cov_yz
        delta_z_var = np.maximum(delta_z_var, 1e-9)

        # Propagate deltas backward through the network
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            
            # Get the PREDICTED post-activations of the PREVIOUS layer (which are inputs to this layer)
            prev_a_mean_predicted = activation_means[i]
            prev_a_var_predicted = activation_vars[i]

            # STEP 1: Use current layer to update its own weights and find the smoothed
            # POST-ACTIVATION (delta_a) of the PREVIOUS layer.
            delta_a_mean, delta_a_var = layer.backward(
                prev_a_mean_predicted, 
                prev_a_var_predicted, 
                delta_z_mean, 
                delta_z_var
            )

            # STEP 2: Convert the smoothed POST-ACTIVATION (delta_a) of the previous layer
            # into the smoothed PRE-ACTIVATION (delta_z) of that same previous layer.
            # This is required for the next iteration of the loop.
            if i > 0:
                prev_layer = self.layers[i-1]
                
                # These are now batch-dependent
                z_mean_prev_predicted = prev_layer.pre_activation_mean
                z_var_prev_predicted = prev_layer.pre_activation_var
                J_prev = prev_layer.jacobian

                # Gain to go from 'a' space to 'z' space: G = Cov(z,a)/Var(a) = J*Var(z)/Var(a)
                safe_prev_a_var_predicted = np.where(prev_a_var_predicted == 0, 1e-9, prev_a_var_predicted)
                gain_za = J_prev * np.divide(z_var_prev_predicted, safe_prev_a_var_predicted)
                
                # Update z
                delta_z_mean = z_mean_prev_predicted + gain_za * (delta_a_mean - prev_a_mean_predicted)
                delta_z_var = z_var_prev_predicted - gain_za**2 * (prev_a_var_predicted - delta_a_var)
                delta_z_var = np.maximum(delta_z_var, 1e-9)


# --- 4. Main Execution ---
def main():
    """Main function to run the TAGI network training and evaluation on CIFAR-10."""
    # --- Configuration ---
    np.random.seed(42)
    
    # NEW: Define the target resolution for CIFAR-10 images
    NEW_IMAGE_RESOLUTION = 16 # e.g., 8, 16, or 24
    
    # Calculate input layer size based on new resolution
    input_size = NEW_IMAGE_RESOLUTION * NEW_IMAGE_RESOLUTION * 3 
    num_classes = 10 # CIFAR-10 has 10 output classes

    # NEW: Smaller network architecture
    UNITS = [input_size, 64, 32, num_classes] 
    
    EPOCHS = 4 # Increased epochs for better training
    BATCH_SIZE = 64 # Define batch size
    OBS_NOISE_VAR = 0.1 

    # --- Data Preparation ---
    print("Loading CIFAR-10 data...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    print(f"Original x_train shape: {x_train.shape}, Original x_test shape: {x_test.shape}")

    # Normalize pixel values to [0, 1] before resizing
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # NEW: Resize images
    print(f"Resizing images to {NEW_IMAGE_RESOLUTION}x{NEW_IMAGE_RESOLUTION}...")
    x_train_resized = np.zeros((x_train.shape[0], NEW_IMAGE_RESOLUTION, NEW_IMAGE_RESOLUTION, 3), dtype=np.float32)
    for i in range(x_train.shape[0]):
        x_train_resized[i] = resize(x_train[i], (NEW_IMAGE_RESOLUTION, NEW_IMAGE_RESOLUTION, 3), anti_aliasing=True)

    x_test_resized = np.zeros((x_test.shape[0], NEW_IMAGE_RESOLUTION, NEW_IMAGE_RESOLUTION, 3), dtype=np.float32)
    for i in range(x_test.shape[0]):
        x_test_resized[i] = resize(x_test[i], (NEW_IMAGE_RESOLUTION, NEW_IMAGE_RESOLUTION, 3), anti_aliasing=True)

    # Flatten images: (N, new_res, new_res, 3) -> (N, input_size)
    x_train_flattened = x_train_resized.reshape(x_train_resized.shape[0], -1)
    x_test_flattened = x_test_resized.reshape(x_test_resized.shape[0], -1)

    # One-hot encode target labels: (N, 1) -> (N, 10)
    y_train_one_hot = to_categorical(y_train, num_classes)
    y_test_one_hot = to_categorical(y_test, num_classes)

    print(f"Flattened x_train shape: {x_train_flattened.shape}, y_train_one_hot shape: {y_train_one_hot.shape}")
    print(f"Flattened x_test shape: {x_test_flattened.shape}, y_test_one_hot shape: {y_test_one_hot.shape}")

    # --- Model Training ---
    model = TAGINetwork(UNITS)

    print("Starting training...")
    n_train_samples = x_train_flattened.shape[0]
    
    for epoch in range(EPOCHS):
        indices = np.random.permutation(n_train_samples)
        
        # Iterate through data in batches
        for i in range(0, n_train_samples, BATCH_SIZE):
            batch_indices = indices[i:i + BATCH_SIZE]
            x_batch = x_train_flattened[batch_indices]
            y_batch = y_train_one_hot[batch_indices]
            model.update(x_batch, y_batch, OBS_NOISE_VAR)
        
        # Evaluate performance on test set after each epoch
        y_pred_mean, _ = model.predict(x_test_flattened)
        
        # For classification, we take the argmax of the predicted means
        y_pred_classes = np.argmax(y_pred_mean, axis=1)
        
        # Compare with original integer labels for test set
        accuracy = np.mean(y_pred_classes == y_test.flatten())
        
        print(f"Epoch {epoch + 1}/{EPOCHS} complete. Test Accuracy: {accuracy:.4f}")

    print("\nTraining complete.")
    
    # --- Final Evaluation ---
    y_pred_mean_test, y_pred_var_test = model.predict(x_test_flattened)
    y_pred_classes_final = np.argmax(y_pred_mean_test, axis=1)
    final_accuracy = np.mean(y_pred_classes_final == y_test.flatten())
    print(f"Final Test Accuracy: {final_accuracy:.4f}")

    # --- Visualization (Optional & Simplified for Classification) ---
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(10, 10))
    for i in range(25): # Display 25 images
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        
        # Reshape the flattened image back to NEW_IMAGE_RESOLUTION x NEW_IMAGE_RESOLUTION x 3 for plotting
        # Note: We use x_test_resized directly here, which is already resized and normalized.
        img = x_test_resized[i] 
        plt.imshow(img)
        
        true_label = y_test[i][0]
        pred_label = y_pred_classes_final[i]
        
        color = 'green' if true_label == pred_label else 'red'
        plt.xlabel(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}", color=color, fontsize=8)
    plt.suptitle(f"CIFAR-10 Predictions at {NEW_IMAGE_RESOLUTION}x{NEW_IMAGE_RESOLUTION} (Green: Correct, Red: Incorrect)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    main()