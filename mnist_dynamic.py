import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Tuple
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from skimage.transform import resize
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="skimage.transform")

# --- 1. Data Handling Class (No Changes Needed) ---
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

# --- 2. Network Layer Class (No Changes Needed) ---
class TAGILayer:
    """
    Represents a single layer in a Tractable Approximate Gaussian Inference (TAGI) network.
    Manages the parameters (weights, biases) and the forward/backward propagation logic.
    """
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
        
        self.pre_activation_var = (
            (prev_activation_mean**2 @ self.weight_var) +
            (prev_activation_var @ self.weight_mean**2) +
            (prev_activation_var @ self.weight_var) + 
            self.bias_var
        )

        if self.is_output_layer:
            activation_mean = self.pre_activation_mean
            activation_var = self.pre_activation_var
            self.jacobian = np.ones_like(self.pre_activation_mean)
        else:
            is_positive = self.pre_activation_mean > 0
            activation_mean = self.pre_activation_mean * is_positive
            activation_var = self.pre_activation_var * is_positive
            self.jacobian = 1.0 * is_positive

        return activation_mean, activation_var

    def backward(self, prev_activation_mean: np.ndarray, prev_activation_var: np.ndarray, 
                 next_delta_mean: np.ndarray, next_delta_var: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = prev_activation_mean.shape[0]
        n_inputs = prev_activation_mean.shape[1]
        n_outputs = self.weight_mean.shape[1]

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


# --- 3. Main Network Class (No Changes Needed) ---
class TAGINetwork:
    """A Bayesian Neural Network using Tractable Approximate Gaussian Inference."""
    def __init__(self, layer_units: List[int]):
        self.layers = []
        for i in range(len(layer_units) - 1):
            is_output = (i == len(layer_units) - 2)
            self.layers.append(TAGILayer(layer_units[i], layer_units[i+1], is_output_layer=is_output))
    
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        activation_mean = x
        activation_var = np.zeros_like(x)
        
        for layer in self.layers:
            activation_mean, activation_var = layer.forward(activation_mean, activation_var)
            
        return activation_mean, activation_var

    def update(self, x: np.ndarray, y: np.ndarray, obs_noise_var: float):
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

    def get_layer_info(self, layer_idx: int):
        if not (0 <= layer_idx < len(self.layers)):
            raise ValueError(f"Layer index {layer_idx} out of bounds.")
        
        layer = self.layers[layer_idx]
        return {
            "weight_mean": layer.weight_mean,
            "weight_var": layer.weight_var,
            "bias_mean": layer.bias_mean,
            "bias_var": layer.bias_var,
            "pre_activation_mean": layer.pre_activation_mean,
            "pre_activation_var": layer.pre_activation_var
        }

# --- Global Variables for Live Plotting ---
param_fig, state_fig, progress_fig = None, None, None
param_axes_list = []
state_axes_list = []
epoch_text = None
accuracy_text = None

def init_visualization_plots_live(model: TAGINetwork, num_layers_to_plot: int):
    """Initializes the Matplotlib figures and axes for live plotting."""
    plt.ion() # Turn on interactive mode
    
    global param_fig, state_fig, progress_fig, param_axes_list, state_axes_list, epoch_text, accuracy_text

    actual_num_layers_to_plot = min(num_layers_to_plot, len(model.layers))

    # --- Figure for Parameters ---
    # MODIFIED: Reduced figsize
    param_fig, axes_param_rows = plt.subplots(actual_num_layers_to_plot, 4, 
                                               figsize=(14, 3 * actual_num_layers_to_plot)) # Smaller height
    param_fig.suptitle("Parameter Distributions (Live Update)", fontsize=14) # Smaller title font
    
    param_axes_list = []
    for i in range(actual_num_layers_to_plot):
        if actual_num_layers_to_plot > 1:
            param_axes_list.extend(axes_param_rows[i])
        else:
            param_axes_list.extend(axes_param_rows)

        # Smaller font sizes for titles and labels within subplots
        param_axes_list[i*4 + 0].set_title(f'L{i} W Mean', fontsize=9); param_axes_list[i*4 + 0].set_xlabel('Value', fontsize=8); param_axes_list[i*4 + 0].set_ylabel('Count', fontsize=8)
        param_axes_list[i*4 + 1].set_title(f'L{i} W Var', fontsize=9); param_axes_list[i*4 + 1].set_xlabel('Value', fontsize=8); param_axes_list[i*4 + 1].set_ylabel('Count (log)', fontsize=8)
        param_axes_list[i*4 + 2].set_title(f'L{i} B Mean', fontsize=9); param_axes_list[i*4 + 2].set_xlabel('Value', fontsize=8); param_axes_list[i*4 + 2].set_ylabel('Count', fontsize=8)
        param_axes_list[i*4 + 3].set_title(f'L{i} B Var', fontsize=9); param_axes_list[i*4 + 3].set_xlabel('Value', fontsize=8); param_axes_list[i*4 + 3].set_ylabel('Count (log)', fontsize=8)

    # Added padding reduction with w_pad and h_pad
    param_fig.tight_layout(rect=[0, 0.03, 1, 0.95], w_pad=0.5, h_pad=1.0) 
    
    # --- Figure for Hidden States ---
    # MODIFIED: Reduced figsize
    state_fig, axes_state_rows = plt.subplots(actual_num_layers_to_plot, 2, 
                                              figsize=(9, 3 * actual_num_layers_to_plot)) # Smaller height
    state_fig.suptitle("Hidden State (Pre-activation) Distributions (Live Update)", fontsize=14) # Smaller title font

    state_axes_list = []
    for i in range(actual_num_layers_to_plot):
        if actual_num_layers_to_plot > 1:
            state_axes_list.extend(axes_state_rows[i])
        else:
            state_axes_list.extend(axes_state_rows)

        state_axes_list[i*2 + 0].set_title(f'L{i} Pre-Act Mean', fontsize=9); state_axes_list[i*2 + 0].set_xlabel('Value', fontsize=8); state_axes_list[i*2 + 0].set_ylabel('Count', fontsize=8)
        state_axes_list[i*2 + 1].set_title(f'L{i} Pre-Act Var', fontsize=9); state_axes_list[i*2 + 1].set_xlabel('Value', fontsize=8); state_axes_list[i*2 + 1].set_ylabel('Count (log)', fontsize=8)

    state_fig.tight_layout(rect=[0, 0.03, 1, 0.95], w_pad=0.5, h_pad=1.0)

    # --- Figure for Accuracy/Loss ---
    # MODIFIED: Reduced figsize
    progress_fig, progress_ax = plt.subplots(figsize=(6, 3)) 
    progress_ax.set_title("Training Progress (Live Update)", fontsize=12) # Smaller title font
    progress_ax.set_xlim(0, 1)
    progress_ax.set_ylim(0, 1)
    progress_ax.set_xticks([])
    progress_ax.set_yticks([])
    
    # Smaller font sizes for text
    epoch_text = progress_ax.text(0.1, 0.8, 'Epoch: 0', transform=progress_ax.transAxes, fontsize=10)
    accuracy_text = progress_ax.text(0.1, 0.6, 'Accuracy: 0.000', transform=progress_ax.transAxes, fontsize=10)
    
    progress_fig.tight_layout()

    param_fig.canvas.draw()
    state_fig.canvas.draw()
    progress_fig.canvas.draw()

    plt.pause(0.01)


def update_parameter_plots(frame, model: TAGINetwork, num_layers_to_plot: int):
    """Update function for parameter (weight/bias) histograms."""
    updated_artists = []
    actual_num_layers_to_plot = min(num_layers_to_plot, len(model.layers))

    for i in range(actual_num_layers_to_plot):
        layer_info = model.get_layer_info(i)

        # Weight Mean
        ax_wm = param_axes_list[i*4 + 0]
        ax_wm.cla()
        ax_wm.hist(layer_info["weight_mean"].flatten(), bins=50, color='skyblue', edgecolor='black')
        ax_wm.set_title(f'L{i} W Mean (Avg: {np.mean(np.abs(layer_info["weight_mean"])):.2e})', fontsize=9) # Ensure font size is set on redraw
        ax_wm.set_xlabel('Value', fontsize=8); ax_wm.set_ylabel('Count', fontsize=8)

        # Weight Variance
        ax_wv = param_axes_list[i*4 + 1]
        ax_wv.cla()
        ax_wv.hist(layer_info["weight_var"].flatten(), bins=50, color='lightcoral', edgecolor='black', log=True)
        ax_wv.set_title(f'L{i} W Var (Avg: {np.mean(layer_info["weight_var"]):.2e})', fontsize=9)
        ax_wv.set_xlabel('Value', fontsize=8); ax_wv.set_ylabel('Count (log)', fontsize=8)

        # Bias Mean
        ax_bm = param_axes_list[i*4 + 2]
        ax_bm.cla()
        ax_bm.hist(layer_info["bias_mean"].flatten(), bins=30, color='lightgreen', edgecolor='black')
        ax_bm.set_title(f'L{i} B Mean', fontsize=9)
        ax_bm.set_xlabel('Value', fontsize=8); ax_bm.set_ylabel('Count', fontsize=8)

        # Bias Variance
        ax_bv = param_axes_list[i*4 + 3]
        ax_bv.cla()
        ax_bv.hist(layer_info["bias_var"].flatten(), bins=30, color='plum', edgecolor='black', log=True)
        ax_bv.set_title(f'L{i} B Var', fontsize=9)
        ax_bv.set_xlabel('Value', fontsize=8); ax_bv.set_ylabel('Count (log)', fontsize=8)
        
    return []

def update_state_plots(frame, model: TAGINetwork, sample_x: np.ndarray, num_layers_to_plot: int):
    """Update function for hidden state (pre-activation) histograms."""
    updated_artists = []
    actual_num_layers_to_plot = min(num_layers_to_plot, len(model.layers))

    model.predict(sample_x)

    for i in range(actual_num_layers_to_plot):
        layer_info = model.get_layer_info(i)

        if layer_info["pre_activation_mean"] is not None:
            # Pre-activation Mean
            ax_pam = state_axes_list[i*2 + 0]
            ax_pam.cla()
            ax_pam.hist(layer_info["pre_activation_mean"].flatten(), bins=50, color='gold', edgecolor='black')
            ax_pam.set_title(f'L{i} Pre-Act Mean (Avg: {np.mean(np.abs(layer_info["pre_activation_mean"])):.2e})', fontsize=9)
            ax_pam.set_xlabel('Value', fontsize=8); ax_pam.set_ylabel('Count', fontsize=8)

            # Pre-activation Variance
            ax_pav = state_axes_list[i*2 + 1]
            ax_pav.cla()
            ax_pav.hist(layer_info["pre_activation_var"].flatten(), bins=50, color='darkorange', edgecolor='black', log=True)
            ax_pav.set_title(f'L{i} Pre-Act Var (Avg: {np.mean(layer_info["pre_activation_var"]):.2e})', fontsize=9)
            ax_pav.set_xlabel('Value', fontsize=8); ax_pav.set_ylabel('Count (log)', fontsize=8)
        else:
            pass
            
    return []

def update_progress_plot(frame, current_epoch: int, current_accuracy: float):
    """Update function for epoch and accuracy text."""
    global epoch_text, accuracy_text
    epoch_text.set_text(f'Epoch: {current_epoch}')
    accuracy_text.set_text(f'Accuracy: {current_accuracy:.4f}')
    return epoch_text, accuracy_text


def main():
    np.random.seed(42)
    
    NEW_IMAGE_RESOLUTION = 14 
    input_size = NEW_IMAGE_RESOLUTION * NEW_IMAGE_RESOLUTION * 1 
    num_classes = 10 

    UNITS = [input_size, 64, 64, 64, num_classes] 
    
    EPOCHS = 10 
    BATCH_SIZE = 32 
    OBS_NOISE_VAR = 0.01 
    
    UPDATE_INTERVAL_BATCHES = 200 # Update plots every 200 batches

    print("Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    print(f"Resizing images to {NEW_IMAGE_RESOLUTION}x{NEW_IMAGE_RESOLUTION}...")
    x_train_resized = np.zeros((x_train.shape[0], NEW_IMAGE_RESOLUTION, NEW_IMAGE_RESOLUTION), dtype=np.float32)
    for i in range(x_train.shape[0]):
        x_train_resized[i] = resize(x_train[i], (NEW_IMAGE_RESOLUTION, NEW_IMAGE_RESOLUTION), anti_aliasing=True)

    x_test_resized = np.zeros((x_test.shape[0], NEW_IMAGE_RESOLUTION, NEW_IMAGE_RESOLUTION), dtype=np.float32)
    for i in range(x_test.shape[0]):
        x_test_resized[i] = resize(x_test[i], (NEW_IMAGE_RESOLUTION, NEW_IMAGE_RESOLUTION), anti_aliasing=True)

    x_train_flattened = x_train_resized.reshape(x_train_resized.shape[0], -1)
    x_test_flattened = x_test_resized.reshape(x_test_resized.shape[0], -1)

    y_train_one_hot = to_categorical(y_train, num_classes)
    y_test_one_hot = to_categorical(y_test, num_classes)

    model = TAGINetwork(UNITS)

    print("Initializing live visualization plots...")
    # You can also reduce the number of layers to plot here if you have many layers
    # For example: num_layers_to_plot=2 will only show the first two hidden layers.
    init_visualization_plots_live(model, num_layers_to_plot=len(UNITS) - 1) 

    visualization_sample_x = x_test_flattened[:BATCH_SIZE]
    
    n_train_samples = x_train_flattened.shape[0]
    
    current_epoch_display = 0
    current_accuracy_display = 0.0

    print("Starting training...")
    for epoch in range(EPOCHS):
        indices = np.random.permutation(n_train_samples)
        
        for i in range(0, n_train_samples, BATCH_SIZE):
            batch_indices = indices[i:i + BATCH_SIZE]
            x_batch = x_train_flattened[batch_indices]
            y_batch = y_train_one_hot[batch_indices]
            model.update(x_batch, y_batch, OBS_NOISE_VAR)
            
            if (i // BATCH_SIZE) % UPDATE_INTERVAL_BATCHES == 0:
                y_pred_mean_current_batch, _ = model.predict(x_test_flattened[:1000])
                y_pred_classes_current_batch = np.argmax(y_pred_mean_current_batch, axis=1)
                accuracy_current_batch = np.mean(y_pred_classes_current_batch == y_test[:1000].flatten())
                
                current_epoch_display = epoch + 1
                current_accuracy_display = accuracy_current_batch

                update_parameter_plots(None, model, len(UNITS) - 1)
                update_state_plots(None, model, visualization_sample_x, len(UNITS) - 1)
                update_progress_plot(None, current_epoch_display, current_accuracy_display)
                
                param_fig.canvas.flush_events()
                state_fig.canvas.flush_events()
                progress_fig.canvas.flush_events()
                plt.pause(0.001)

        y_pred_mean, _ = model.predict(x_test_flattened)
        y_pred_classes = np.argmax(y_pred_mean, axis=1)
        accuracy = np.mean(y_pred_classes == y_test.flatten())
        
        print(f"Epoch {epoch + 1}/{EPOCHS} complete. Test Accuracy: {accuracy:.4f}")

        current_epoch_display = epoch + 1
        current_accuracy_display = accuracy
        update_parameter_plots(None, model, len(UNITS) - 1)
        update_state_plots(None, model, visualization_sample_x, len(UNITS) - 1)
        update_progress_plot(None, current_epoch_display, current_accuracy_display)
        param_fig.canvas.flush_events()
        state_fig.canvas.flush_events()
        progress_fig.canvas.flush_events()
        plt.pause(0.01)

    print("\nTraining complete.")
    
    y_pred_mean_test, y_pred_var_test = model.predict(x_test_flattened)
    y_pred_classes_final = np.argmax(y_pred_mean_test, axis=1)
    final_accuracy = np.mean(y_pred_classes_final == y_test.flatten())
    print(f"Final Test Accuracy: {final_accuracy:.4f}")

    plt.ioff()
    
    class_names = [str(i) for i in range(10)]

    plt.figure(figsize=(8, 8)) # Adjusted for the final plot too
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
        plt.xlabel(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}", color=color, fontsize=7) # Smaller font
    plt.suptitle(f"MNIST Predictions at {NEW_IMAGE_RESOLUTION}x{NEW_IMAGE_RESOLUTION} (Green: Correct, Red: Incorrect)", fontsize=14) # Smaller font
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    plt.show(block=True)

if __name__ == "__main__":
    main()