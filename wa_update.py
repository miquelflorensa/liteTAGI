import numpy as np

# Original functions for single-step updates (unchanged)
def update_independent_gaussians_for_sum_target_2(
    initial_means,
    initial_variances,
    target_sum_mean_2,
    target_sum_variance_2
):
    mu_Z = initial_means**2 + initial_variances
    var_Z = 2 * initial_variances**2 + 4 * initial_variances * initial_means**2
    
    mu_S2 = np.sum(mu_Z)
    sigma2_S2 = np.sum(var_Z)

    epsilon = 1e-9 # Small value to prevent division by zero

    # Jz as defined in the original user's code for S2
    Jz = 2 * initial_means * initial_variances / (sigma2_S2 + epsilon)

    updated_means = initial_means + Jz * (target_sum_mean_2 - mu_S2)
    updated_variances = initial_variances + Jz**2 * (target_sum_variance_2 - sigma2_S2)
    
    # Ensure variances remain positive
    updated_variances = np.maximum(updated_variances, epsilon)
    
    return updated_means, updated_variances

def update_independent_gaussians_for_sum_target(
    initial_means,
    initial_variances,
    target_sum_mean,
    target_sum_variance
):
    mu_S = np.sum(initial_means)
    sigma2_S = np.sum(initial_variances)

    epsilon = 1e-9 # Small value to prevent division by zero
    Jz = initial_variances / (sigma2_S + epsilon)

    updated_means = initial_means + Jz * (target_sum_mean - mu_S)
    updated_variances = initial_variances + Jz * (target_sum_variance - sigma2_S)

    # Ensure variances remain positive
    updated_variances = np.maximum(updated_variances, epsilon)

    return updated_means, updated_variances

# Helper function to compute Z1 properties (unchanged)
def calculate_Z1_properties(mu_Z0, var_Z0, mu_W, var_W, mu_B, var_B):
    """
    Calculates the mean and variance of Z1 = Z0 @ W + B (matrix multiplication)
    assuming Z0, W, B are independent Gaussian random variables and
    elements within Z0, W, B are also independent.
    """
    A_size = mu_Z0.shape[0]
    
    predicted_mu_Z1 = np.zeros(A_size)
    for j in range(A_size):
        predicted_mu_Z1[j] = np.sum(mu_Z0 * mu_W[:, j]) + mu_B[j]

    predicted_var_Z1 = np.zeros(A_size)
    for j in range(A_size):
        term_variances = var_Z0 * var_W[:, j] + var_Z0 * mu_W[:, j]**2 + var_W[:, j] * mu_Z0**2
        predicted_var_Z1[j] = np.sum(term_variances) + var_B[j]

    return predicted_mu_Z1, predicted_var_Z1

# Helper function to compute Z0W properties (unchanged)
def calculate_Z0W_properties(mu_Z0, var_Z0, mu_W, var_W):
    """
    Calculates the mean and variance of the Z0W term (matrix multiplication output)
    assuming Z0, W are independent Gaussian random variables and
    elements within Z0, W are also independent.
    """
    A_size = mu_Z0.shape[0]
    initial_mu_Z0W = np.zeros(A_size)
    initial_var_Z0W = np.zeros(A_size)

    for j in range(A_size): # For each output dimension of Z0W
        initial_mu_Z0W[j] = np.sum(mu_Z0 * mu_W[:, j])
        term_variances = var_Z0 * var_W[:, j] + \
                         var_Z0 * mu_W[:, j]**2 + \
                         var_W[:, j] * mu_Z0**2
        initial_var_Z0W[j] = np.sum(term_variances)
    return initial_mu_Z0W, initial_var_Z0W


# Function to update Z0W given Z1 and B (now a direct subtraction as you suggested)
def update_Z0W_for_Z1_target(
    current_mu_Z0W, # Not strictly used with direct subtraction, but kept for signature consistency
    current_var_Z0W, # Not strictly used, but kept for signature consistency
    mu_B,
    var_B,
    target_mu_Z1,
    target_var_Z1
):
    """
    Directly infers mu_Z0W and var_Z0W by subtracting mu_B and var_B
    from the target Z1 properties. This assumes B is a fixed quantity.
    """
    # Ensure variances remain positive
    updated_var_Z0W = np.maximum(target_var_Z1 - var_B, 1e-9)
    return target_mu_Z1 - mu_B, updated_var_Z0W

def rts_smooth_Z0W_components(
    initial_mu_Z0, initial_var_Z0,
    initial_mu_W, initial_var_W,
    updated_mu_Z0W_sum_target, updated_var_Z0W_sum_target
):
    """
    Infers the updated mean and variance for each individual Z0_i * W_ij product term (P_ij)
    given the updated mean and variance of their sum (Z0W_j).
    This simulates a backward pass to distribute the updated sum's information.

    Args:
        initial_mu_Z0, initial_var_Z0: Initial means and variances of Z0.
        initial_mu_W, initial_var_W: Initial means and variances of W.
        updated_mu_Z0W_sum_target, updated_var_Z0W_sum_target: The target mean and variance
                                                                for the sum Z0 @ W for each output dimension.

    Returns:
        updated_mu_P_ij (A_size, A_size), updated_var_P_ij (A_size, A_size):
            Updated means and variances for each individual product Z0_i * W_ij.
    """
    A_size = initial_mu_Z0.shape[0]
    epsilon = 1e-9

    # Calculate initial properties of each individual product P_ij = Z0_i * W_ij
    # E[XY] = E[X]E[Y]
    # Var(XY) = Var(X)Var(Y) + Var(X)E[Y]^2 + Var(Y)E[X]^2
    initial_mu_P_ij = initial_mu_Z0[:, np.newaxis] * initial_mu_W
    initial_var_P_ij = (initial_var_Z0[:, np.newaxis] * initial_var_W) + \
                       (initial_var_Z0[:, np.newaxis] * initial_mu_W**2) + \
                       (initial_var_W * initial_mu_Z0[:, np.newaxis]**2)

    updated_mu_P_ij = np.copy(initial_mu_P_ij)
    updated_var_P_ij = np.copy(initial_var_P_ij)

    # Iterate through each output dimension 'j' of Z0W
    for j in range(A_size):
        # Current sum properties for this output dimension j
        current_mu_Z0W_j = np.sum(updated_mu_P_ij[:, j])
        current_var_Z0W_j = np.sum(updated_var_P_ij[:, j])

        # Error for this sum
        error_mu_sum = updated_mu_Z0W_sum_target[j] - current_mu_Z0W_j
        error_var_sum = updated_var_Z0W_sum_target[j] - current_var_Z0W_j

        # Calculate gain for each individual P_ij based on its contribution to the sum's variance
        # This is analogous to the Kalman gain for individual components in a sum
        gain_P_ij_for_sum = updated_var_P_ij[:, j] / (current_var_Z0W_j + epsilon)

        # Update each P_ij for the current output dimension 'j'
        updated_mu_P_ij[:, j] += gain_P_ij_for_sum * error_mu_sum
        updated_var_P_ij[:, j] += gain_P_ij_for_sum * error_var_sum # Gain for variance, not squared as per common linear gain update

        # Ensure variances remain positive
        # updated_var_P_ij[:, j] = np.maximum(updated_var_P_ij[:, j], epsilon)

    return updated_mu_P_ij, updated_var_P_ij

def update_W_from_Z0_and_Z0W_products(
    mu_Z0, var_Z0, # Current state of Z0
    mu_P_ij, var_P_ij # Updated means/variances of individual products P_ij = Z0_i * W_ij
):
    
    A_size = mu_Z0.shape[0]
    epsilon = 1e-9 # Small value to prevent division by zero

    updated_mu_W = np.zeros((A_size, A_size))
    updated_var_W = np.zeros((A_size, A_size))

    for i in range(A_size): # Iterate over rows of W (corresponding to Z0_i)
        for j in range(A_size): # Iterate over columns of W (corresponding to output dimension j)
            denominator_mu = mu_Z0[i]
            if np.abs(denominator_mu) < epsilon:
                updated_mu_W[i, j] = 0.0 # Or some prior mean for W_ij
            else:
                updated_mu_W[i, j] = mu_P_ij[i, j] / denominator_mu

            # Calculate updated variance for W_ij
            numerator_var = var_P_ij[i, j] - (var_Z0[i] * updated_mu_W[i, j]**2)
            denominator_var = mu_Z0[i]**2 + var_Z0[i]

            # Ensure denominator is not zero or too small
            if denominator_var < epsilon:
                updated_var_W[i, j] = 1.0 # Or some prior variance for W_ij
            else:
                updated_var_W[i, j] = numerator_var / denominator_var
            
            # Ensure variance is positive
            # updated_var_W[i, j] = np.maximum(updated_var_W[i, j], epsilon)
            
    return updated_mu_W, updated_var_W


if __name__ == "__main__":
    A_size = 64

    # 1. Initialize Z0, W, B
    print(f"--- Initializing variables (A_size={A_size}) ---")
    
    # Z0: (A) vector
    initial_mu_Z0 = np.random.normal(0.0, 1, A_size)
    initial_var_Z0 = np.abs(np.random.normal(0.0, 0.0, A_size))

    #Make half of the Z0 variables 0
    initial_mu_Z0[:A_size // 2] = 0.0
    initial_var_Z0[:A_size // 2] = 0.0


    # W: (A, A) matrix
    initial_mu_W = np.random.normal(0, 0.1, (A_size, A_size))
    initial_var_W = np.abs(np.random.normal(0.01, 0.5, (A_size, A_size))) + 1e-6

    # W: (A, A) matrix
    initial_mu_W = np.random.normal(0, 0.1, (A_size, A_size))
    initial_var_W = np.abs(np.random.normal(0.01, 0.5, (A_size, A_size))) + 1e-6

    # B: (A) vector
    initial_mu_B = np.random.normal(0.1, 0.05, A_size)
    initial_var_B = np.abs(np.random.normal(0.01, 0.5, A_size)) + 1e-6

    # 2. Compute Initial Z1 = Z0 @ W + B
    print("--- Computing initial Z1 from Z0, W, B ---")
    initial_mu_Z1, initial_var_Z1 = calculate_Z1_properties(
        initial_mu_Z0, initial_var_Z0, initial_mu_W, initial_var_W, initial_mu_B, initial_var_B
    )

    # Define targets for Z1 (S~ and S~2)
    target_sum_mean_Z1 = 0.0
    target_sum_variance_Z1 = A_size 

    target_sum_mean_2_Z1 = 2.0 * A_size
    target_sum_variance_2_Z1 = 6.0 * A_size 

    updated_means_Z1_from_S2, updated_variances_Z1_from_S2 = initial_mu_Z1, initial_var_Z1

    for i in range(1000):
        # 3. Perform a SINGLE update for Z1 based on targets for S and S2
        updated_means_Z1, updated_variances_Z1 = update_independent_gaussians_for_sum_target(
            updated_means_Z1_from_S2,
            updated_variances_Z1_from_S2,
            target_sum_mean_Z1,
            target_sum_variance_Z1
        )

        updated_means_Z1_from_S2, updated_variances_Z1_from_S2 = update_independent_gaussians_for_sum_target_2(
            updated_means_Z1,
            updated_variances_Z1,
            target_sum_mean_2_Z1,
            target_sum_variance_2_Z1
        )
    print(f"Verification of Z1 properties after S and S2 updates:")
    current_sum_mean_Z1 = np.sum(updated_means_Z1_from_S2)
    current_sum_variance_Z1 = np.sum(updated_variances_Z1_from_S2)
    print(f"Sum of updated Z1 means (S): {current_sum_mean_Z1:.6f} (Original target S: {target_sum_mean_Z1})")
    print(f"Sum of updated Z1 variances (Var(S)): {current_sum_variance_Z1:.6f} (Original target Var(S): {target_sum_variance_Z1})")

    current_S2_mean_Z1 = np.sum(updated_means_Z1_from_S2**2 + updated_variances_Z1_from_S2)
    current_S2_variance_Z1 = np.sum(2 * updated_variances_Z1_from_S2**2 + 4 * updated_variances_Z1_from_S2 * updated_means_Z1_from_S2**2)
    print(f"E[Sum Z1^2] (S2): {current_S2_mean_Z1:.6f} (Target S2: {target_sum_mean_2_Z1})")
    print(f"Var(Sum Z1^2) (Var(S2)): {current_S2_variance_Z1:.6f} (Target Var(S2): {target_sum_variance_2_Z1})\n")

    # 4. Update (Z0*W) given updated Z1 and initial B
    initial_mu_Z0W, initial_var_Z0W = calculate_Z0W_properties(initial_mu_Z0, initial_var_Z0, initial_mu_W, initial_var_W)
    
    print("--- Directly inferring Z0W properties by subtracting B ---")
    updated_mu_Z0W_target, updated_var_Z0W_target = update_Z0W_for_Z1_target(
        initial_mu_Z0W, initial_var_Z0W, # Not used in this version, but kept for signature
        initial_mu_B, initial_var_B,
        updated_means_Z1_from_S2,
        updated_variances_Z1_from_S2
    )

    print(f"Directly Inferred mu_Z0W target: {updated_mu_Z0W_target.shape}")
    print(f"Directly Inferred var_Z0W target: {updated_var_Z0W_target.shape}\n")

    # Final verification: Check Z1 calculated from directly inferred Z0W and initial B against target Z1
    final_predicted_mu_Z1 = updated_mu_Z0W_target + initial_mu_B
    final_predicted_var_Z1 = updated_var_Z0W_target + initial_var_B

    print("--- Verification of Z1 calculated from directly inferred Z0W and initial B ---")
    print(f"Max abs difference in Z1 Means: {np.max(np.abs(final_predicted_mu_Z1 - updated_means_Z1_from_S2)):.6f}")
    print(f"Max abs difference in Z1 Variances: {np.max(np.abs(final_predicted_var_Z1 - updated_variances_Z1_from_S2)):.6f}\n")
    
    # --- New section: Inferring back to individual Z0W product terms using RTS smoother concept ---
    print("--- Inferring back to individual Z0_i * W_ij product terms ---")
    updated_mu_P_ij, updated_var_P_ij = rts_smooth_Z0W_components(
        initial_mu_Z0, initial_var_Z0,
        initial_mu_W, initial_var_W,
        updated_mu_Z0W_target, updated_var_Z0W_target
    )

    print(f"Updated mu_P_ij (shape): {updated_mu_P_ij.shape}")
    print(f"Updated var_P_ij (shape): {updated_var_P_ij.shape}\n")

    # Verification: Sum the updated individual P_ij to see if they match the Z0W target
    re_summed_mu_Z0W = np.sum(updated_mu_P_ij, axis=0)
    re_summed_var_Z0W = np.sum(updated_var_P_ij, axis=0)

    print("--- Verification of Z0W sums from smoothed P_ij components ---")
    print(f"Target mu_Z0W (from Z1 - B): {updated_mu_Z0W_target[:5]}")
    print(f"Re-summed mu_Z0W (from P_ij): {re_summed_mu_Z0W[:5]}")
    print(f"Max abs difference in mu_Z0W: {np.max(np.abs(re_summed_mu_Z0W - updated_mu_Z0W_target)):.6f}\n")

    print(f"Target var_Z0W (from Z1 - B): {updated_var_Z0W_target[:5]}")
    print(f"Re-summed var_Z0W (from P_ij): {re_summed_var_Z0W[:5]}")
    print(f"Max abs difference in var_Z0W: {np.max(np.abs(re_summed_var_Z0W - updated_var_Z0W_target)):.6f}\n")

    # --- NEW: Update W from the individual P_ij products and current Z0 ---
    print("--- Updating W from P_ij and Z0 using inverse product formulas ---")
    updated_mu_W_final, updated_var_W_final = update_W_from_Z0_and_Z0W_products(
        initial_mu_Z0, initial_var_Z0, 
        updated_mu_P_ij, updated_var_P_ij
    )

    # print(f"Updated mu_W_final: {updated_mu_W_final}")
    # print(f"Updated var_W_final: {updated_var_W_final}\n")

    # Verification: Check if the updated W matches the expected properties
    print("--- Verification of updated W properties ---")
    re_summed_mu_Z0W_from_W = np.dot(initial_mu_Z0, updated_mu_W_final)
    re_summed_var_Z0W_from_W = np.sum(initial_var_Z0[:, np.newaxis] * updated_var_W_final, axis=0) + \
                               np.sum(initial_mu_Z0[:, np.newaxis]**2 * updated_var_W_final, axis=0)
    print(f"Re-summed mu_Z0W from updated W: {re_summed_mu_Z0W_from_W[:5]}")
    print(f"Re-summed var_Z0W from updated W: {re_summed_var_Z0W_from_W[:5]}")
    print(f"Max abs difference in mu_Z0W from W: {np.max(np.abs(re_summed_mu_Z0W_from_W - updated_mu_Z0W_target)):.6f}")
    print(f"Max abs difference in var_Z0W from W: {np.max(np.abs(re_summed_var_Z0W_from_W - updated_var_Z0W_target)):.6f}\n")

    # Re compute Z1 from updated Z0 and W
    print("--- Recomputing Z1 from updated Z0 and W ---")
    final_updated_mu_Z1, final_updated_var_Z1 = calculate_Z1_properties(
        initial_mu_Z0, initial_var_Z0, updated_mu_W_final, updated_var_W_final, initial_mu_B, initial_var_B
    )
    print(f"Max abs difference in final updated Z1 Means: {np.max(np.abs(final_updated_mu_Z1 - updated_means_Z1_from_S2)):.6f}")
    print(f"Max abs difference in final updated Z1 Variances: {np.max(np.abs(final_updated_var_Z1 - updated_variances_Z1_from_S2)):.6f}\n")

    # Check targets for final Z1
    final_sum_mean_Z1 = np.sum(final_updated_mu_Z1)
    final_sum_variance_Z1 = np.sum(final_updated_var_Z1)
    final_S2_mean_Z1 = np.sum(final_updated_mu_Z1**2 + final_updated_var_Z1)
    final_S2_variance_Z1 = np.sum(2 * final_updated_var_Z1**2 + 4 * final_updated_var_Z1 * final_updated_mu_Z1**2)
    print(f"Final Sum of Updated Z1 Means (S): {final_sum_mean_Z1:.6f} (Target: {target_sum_mean_Z1})")
    print(f"Final Sum of Updated Z1 Variances (Var(S)): {final_sum_variance_Z1:.6f} (Target: {target_sum_variance_Z1})")
    print(f"Final Sum of Updated Z1 Means Squared (E[S2]): {final_S2_mean_Z1:.6f} (Target: {target_sum_mean_2_Z1})")
    print(f"Final Variance of Sum of Updated Z1 Squares (Var(S2)): {final_S2_variance_Z1:.6f} (Target: {target_sum_variance_2_Z1})\n")



    # Print initial and final weights
    print("--- Initial and Final Weights ---")
    print(f"Initial mu_W (first 5 elements): {initial_mu_W.flatten()[:5]}")
    print(f"Final mu_W (first 5 elements): {updated_mu_W_final.flatten()[:5]}")
    print(f"Initial var_W (first 5 elements): {initial_var_W.flatten()[:5]}")
    print(f"Final var_W (first 5 elements): {updated_var_W_final.flatten()[:5]}\n")