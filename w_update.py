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

# Helper function to compute Z0W properties (unchanged - still useful for initial Z0W_ij)
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


def update_Z0W_ij_from_Z1_target(
    initial_mu_W,       # Initial mean of W (matrix)
    initial_var_W,      # Initial variance of W (matrix)
    mu_Z0,              # Mean of Z0 (vector)
    var_Z0,             # Variance of Z0 (vector)
    initial_mu_Z1,      # Initial mean of Z1 (vector)
    initial_var_Z1,     # Initial variance of Z1 (vector)
    target_mu_Z1,       # Target mean of Z1 (vector)
    target_var_Z1,      # Target variance of Z1 (vector)
    mu_B,               # Mean of B (vector) - ADDED
    var_B               # Variance of B (vector) - ADDED
):
    A_size = mu_Z0.shape[0]

    # 1. Calculate the initial properties of each individual product term Z0W_ij = Z0_i * W_ij
    initial_mu_Z0W_ij = mu_Z0[:, np.newaxis] * initial_mu_W 
    initial_var_Z0W_ij = var_Z0[:, np.newaxis] * initial_var_W + \
                       var_Z0[:, np.newaxis] * initial_mu_W**2 + \
                       initial_var_W * mu_Z0[:, np.newaxis]**2

    # Initialize matrices for updated Z0W_ij values
    updated_mu_Z0W_ij = np.copy(initial_mu_Z0W_ij)
    updated_var_Z0W_ij = np.copy(initial_var_Z0W_ij)

    # 3. Distribute the Z1 target update to each Z0W_ij component
    # For each output dimension j: Z1_j = sum_i(Z0W_ij) + B_j
    for j in range(A_size):
        # Let S_j = sum_i(Z0W_ij). We have Z1_j = S_j + B_j.
        # We want to update S_j based on target Z1_j.
        # We need to compute the initial mu_S_j and var_S_j
        initial_mu_S_j = np.sum(initial_mu_Z0W_ij[:, j])
        initial_var_S_j = np.sum(initial_var_Z0W_ij[:, j])

        # New target for S_j (mu_S_j_target, var_S_j_target) based on Z1_j target
        # E[Z1_j] = E[S_j] + E[B_j]  => E[S_j]_target = E[Z1_j]_target - E[B_j]
        # Var[Z1_j] = Var[S_j] + Var[B_j] => Var[S_j]_target = Var[Z1_j]_target - Var[B_j]
        
        target_mu_S_j = target_mu_Z1[j] - mu_B[j]
        target_var_S_j = target_var_Z1[j] - var_B[j]
        
        epsilon = 1e-9 # Small value to prevent division by zero
        # Jz for updating individual Z0W_ij terms based on the sum S_j target
        Jz_Z0W_ij_for_S_update = initial_var_Z0W_ij[:, j] / (initial_var_S_j + epsilon)

        updated_mu_Z0W_ij[:, j] = initial_mu_Z0W_ij[:, j] + Jz_Z0W_ij_for_S_update * (target_mu_S_j - initial_mu_S_j)
        updated_var_Z0W_ij[:, j] = initial_var_Z0W_ij[:, j] + Jz_Z0W_ij_for_S_update**2 * (target_var_S_j - initial_var_S_j)

        # Ensure variances remain positive
        updated_var_Z0W_ij[:, j] = np.maximum(updated_var_Z0W_ij[:, j], epsilon)
        
    return updated_mu_Z0W_ij, updated_var_Z0W_ij


# --- update_W_from_Z0W_target remains as the final direct step ---
def update_W_from_Z0W_target(
    initial_mu_W,
    initial_var_W,
    mu_Z0,    
    var_Z0,
    updated_mu_Z0W_ij,
    updated_var_Z0W_ij
):
    epsilon = 1e-9
    A_size = mu_Z0.shape[0] 
    
    updated_mu_W = np.zeros_like(initial_mu_W)
    updated_var_W = np.zeros_like(initial_var_W)

    for i in range(A_size): # For each input dimension (row i of W)
        for j in range(A_size): # For each output dimension (column j of W)
            # We have Z0W_ij = Z0_i * W_ij
            # E[Z0W_ij] = E[Z0_i] * E[W_ij]
            # Var[Z0W_ij] = Var[Z0_i]Var[W_ij] + Var[Z0_i]E[W_ij]^2 + Var[W_ij]E[Z0_i]^2

            mu_Z0_i = mu_Z0[i]
            var_Z0_i = var_Z0[i]
            
            # Target for Z0W_ij
            target_mu_Z0W_ij = updated_mu_Z0W_ij[i, j]
            target_var_Z0W_ij = updated_var_Z0W_ij[i, j]

            # Current W_ij properties (for Jacobian calculation in iterative updates, not strictly needed for direct inference here if closed-form)
            # But the user's current implementation infers directly.

            # Infer mean of W_ij: mu_W_ij = E[Z0W_ij] / E[Z0_i]
            # This is correct if Z0_i is deterministic or E[Z0_i] is non-zero
            updated_mu_W[i, j] = target_mu_Z0W_ij / mu_Z0_i if mu_Z0_i != 0 else initial_mu_W[i,j] # Fallback if mu_Z0_i is zero

            # Infer variance of W_ij from Var[Z0W_ij] equation
            # var_W_ij = (Var[Z0W_ij] - Var[Z0_i]E[W_ij]^2) / (E[Z0_i]^2 + Var[Z0_i])
            denominator_var = mu_Z0_i**2 + var_Z0_i
            
            # Avoid division by zero
            if denominator_var < epsilon:
                updated_var_W[i, j] = initial_var_W[i, j] # Keep original if denominator is near zero
            else:
                numerator_var = target_var_Z0W_ij - var_Z0_i * updated_mu_W[i, j]**2
                updated_var_W[i, j] = numerator_var / denominator_var
            
            # Ensure variance is positive
            updated_var_W[i, j] = np.maximum(updated_var_W[i, j], epsilon)
    
    return updated_mu_W, updated_var_W


if __name__ == "__main__":
    A_size = 1000 # Using a smaller A_size for easier inspection of outputs

    # 1. Initialize Z0, W, B
    print(f"--- Initializing variables (A_size={A_size}) ---")
    
    # Z0: (A) vector
    initial_mu_Z0 = np.random.normal(1.0, 0.5, A_size)
    initial_var_Z0 = np.abs(np.random.normal(1.0, 0.2, A_size)) + 1e-6

    # W: (A, A) matrix
    initial_mu_W = np.random.normal(0, 0.1, (A_size, A_size))
    initial_var_W = np.ones((A_size, A_size)) / A_size / 2

    # B: (A) vector
    initial_mu_B = np.random.normal(0.1, 0.05, A_size)
    initial_var_B = np.ones(A_size)

    print(f"Initial mu_Z0: {initial_mu_Z0}")
    print(f"Initial var_Z0: {initial_var_Z0}")
    print(f"Initial mu_W (first 2x2): \n{initial_mu_W[:2,:2]}")
    print(f"Initial var_W (first 2x2): \n{initial_var_W[:2,:2]}")
    print(f"Initial mu_B: {initial_mu_B}")
    print(f"Initial var_B: {initial_var_B}\n")

    # Define targets for Z1 (S~ and S~2)
    # These targets should be for the sum over elements of Z1
    target_sum_mean_Z1 = 0.0
    target_sum_variance_Z1 = A_size 

    target_sum_mean_2_Z1 = 2.0 * A_size 
    target_sum_variance_2_Z1 = 6.0 * A_size 


    # 2. Compute Initial Z1 = Z0 @ W + B
    print("--- Computing initial Z1 from Z0, W, B ---")
    initial_mu_Z1, initial_var_Z1 = calculate_Z1_properties(
        initial_mu_Z0, initial_var_Z0, initial_mu_W, initial_var_W, initial_mu_B, initial_var_B
    )
    print(f"Initial mu_Z1: {initial_mu_Z1}")
    print(f"Initial var_Z1: {initial_var_Z1}\n")


    updated_means_Z1, updated_variances_Z1 = initial_mu_Z1, initial_var_Z1
    
    for i in range(100):
        # 3. Perform a SINGLE update for Z1 based on targets for S and S2 sequentially
        # print("--- Performing a SINGLE update for Z1 based on targets for S ---")
        updated_means_Z1_s, updated_variances_Z1_s = update_independent_gaussians_for_sum_target(
            updated_means_Z1,
            updated_variances_Z1,
            target_sum_mean_Z1,
            target_sum_variance_Z1
        )

        # print("--- Performing a SINGLE update for Z1 based on targets for S2 ---")
        # Apply S2 update on the results of S update for Z1
        updated_means_Z1, updated_variances_Z1 = update_independent_gaussians_for_sum_target_2(
            updated_means_Z1_s,
            updated_variances_Z1_s,
            target_sum_mean_2_Z1,
            target_sum_variance_2_Z1
        )
        
    print(f"Updated Z1 means: {updated_means_Z1}")
    print(f"Updated Z1 variances: {updated_variances_Z1}\n")

    print("--- Verification of Z1 properties after update:")
    # Calculate sum S properties from updated Z1
    current_sum_mean_Z1 = np.sum(updated_means_Z1)
    current_sum_variance_Z1 = np.sum(updated_variances_Z1)
    print(f"Sum of updated Z1 means (S): {current_sum_mean_Z1:.6f} (Original target S: {target_sum_mean_Z1})")
    print(f"Sum of updated Z1 variances (Var(S)): {current_sum_variance_Z1:.6f} (Original target Var(S): {target_sum_variance_Z1})\n")
    # Calculate sum of squares S2 properties from updated Z1
    current_S2_mean_Z1 = np.sum(updated_means_Z1**2 + updated_variances_Z1)
    current_S2_variance_Z1 = np.sum(2 * updated_variances_Z1**2 + 4 * updated_variances_Z1 * updated_means_Z1**2)
    print(f"Sum of updated Z1 means squared (E[S2]): {current_S2_mean_Z1:.6f} (Original target S2: {target_sum_mean_2_Z1})")
    print(f"Variance of sum of updated Z1 squares (Var(S2)): {current_S2_variance_Z1:.6f} (Original target Var(S2): {target_sum_variance_2_Z1})\n")


    # --- NEW SINGLE STEP: Update individual WZ0 (Z0W_ij) directly from Z1 target ---
    print("--- Updating individual WZ0 (Z0W_ij) directly from Z1 target ---")
    updated_mu_Z0W_ij, updated_var_Z0W_ij = update_Z0W_ij_from_Z1_target(
        initial_mu_W,
        initial_var_W,
        initial_mu_Z0,
        initial_var_Z0,
        initial_mu_Z1, # Initial Z1 means
        initial_var_Z1, # Initial Z1 variances
        updated_means_Z1, # Directly use the updated Z1 as target
        updated_variances_Z1, # Directly use the updated Z1 as target
        initial_mu_B, # Pass B means - ADDED
        initial_var_B # Pass B variances - ADDED
    )

    # Check if the updated Z0W_ij matches the target Z1 properties
    print("--- Verification of updated Z0W_ij properties against Z1 target ---")
    # Calculate Z1 properties from updated Z0W_ij
    predicted_mu_Z1_from_Z0W = np.sum(updated_mu_Z0W_ij, axis=0) + initial_mu_B
    predicted_var_Z1_from_Z0W = np.sum(updated_var_Z0W_ij, axis=0) + initial_var_B

    # Calculate sum S properties from predicted Z1
    current_sum_mean_Z1_from_Z0W = np.sum(predicted_mu_Z1_from_Z0W)
    current_sum_variance_Z1_from_Z0W = np.sum(predicted_var_Z1_from_Z0W)
    print(f"Sum of predicted Z1 means (S) from updated Z0W_ij: {current_sum_mean_Z1_from_Z0W:.6f} (Target S: {target_sum_mean_Z1})")
    print(f"Sum of predicted Z1 variances (Var(S)) from updated Z0W_ij: {current_sum_variance_Z1_from_Z0W:.6f} (Target Var(S): {target_sum_variance_Z1})\n")
    # Calculate sum of squares S2 properties from predicted Z1
    current_S2_mean_Z1_from_Z0W = np.sum(predicted_mu_Z1_from_Z0W**2 + predicted_var_Z1_from_Z0W)
    current_S2_variance_Z1_from_Z0W = np.sum(2 * predicted_var_Z1_from_Z0W**2 + 4 * predicted_var_Z1_from_Z0W * predicted_mu_Z1_from_Z0W**2)
    print(f"Sum of predicted Z1 means squared (E[S2]) from updated Z0W_ij: {current_S2_mean_Z1_from_Z0W:.6f} (Target S2: {target_sum_mean_2_Z1})")
    print(f"Variance of sum of predicted Z1 squares (Var(S2)) from updated Z0W_ij: {current_S2_variance_Z1_from_Z0W:.6f} (Target Var(S2): {target_sum_variance_2_Z1})\n")

    # 5. Update W from updated individual WZ0 (Z0W_ij)
    print("--- Updating W from updated individual Z0W (Z0W_ij) and initial Z0 ---")
    # Now, update_W_from_Z0W_target directly takes the individual product updates
    updated_mu_W, updated_var_W = update_W_from_Z0W_target(
        initial_mu_W, # Used as fallback
        initial_var_W, # Used as fallback
        initial_mu_Z0,  # Z0 is considered fixed for this update
        initial_var_Z0, # Z0 is considered fixed for this update
        updated_mu_Z0W_ij, # Now passing the matrix of individual product updates
        updated_var_Z0W_ij # Now passing the matrix of individual product updates
    )

    # print(f"Updated mu_W (first 2x2): \n{updated_mu_W[:2,:2]}")
    # print(f"Updated var_W (first 2x2): \n{updated_var_W[:2,:2]}\n")

    # Final verification: Calculate Z0W from updated W and initial Z0
    # Then calculate Z1 from this new Z0W and initial B
    final_predicted_mu_Z0W, final_predicted_var_Z0W = calculate_Z0W_properties(
        initial_mu_Z0, initial_var_Z0, updated_mu_W, updated_var_W
    )

    # Calculate final Z1 to see how far it is from the original target Z1
    final_Z1_mu_from_updated_W = final_predicted_mu_Z0W + initial_mu_B
    final_Z1_var_from_updated_W = final_predicted_var_Z0W + initial_var_B

    print("--- Overall Verification: Z1 calculated from updated W & B vs original Z1 target ---")
    print(f"Overall Target Z1 Means: {updated_means_Z1}")
    print(f"Final Z1 Means (from updated W, initial B): {final_Z1_mu_from_updated_W}")
    print(f"Max abs difference in final Z1 Means: {np.max(np.abs(final_Z1_mu_from_updated_W - updated_means_Z1)):.6f}\n")

    print(f"Overall Target Z1 Variances: {updated_variances_Z1}")
    print(f"Final Z1 Variances (from updated W, initial B): {final_Z1_var_from_updated_W}")
    print(f"Max abs difference in final Z1 Variances: {np.max(np.abs(final_Z1_var_from_updated_W - updated_variances_Z1)):.6f}\n")

    # Final verification of Z1 properties
    current_sum_mean_Z1 = np.sum(final_Z1_mu_from_updated_W)
    current_sum_variance_Z1 = np.sum(final_Z1_var_from_updated_W)
    print(f"Sum of updated Z1 means (S): {current_sum_mean_Z1:.6f} (Target S: {target_sum_mean_Z1})")
    print(f"Sum of updated Z1 variances (Var(S)): {current_sum_variance_Z1:.6f} (Target Var(S): {target_sum_variance_Z1})\n")
    # Final verification of Z1^2 properties
    current_S2_mean_Z1 = np.sum(final_Z1_mu_from_updated_W**2 + final_Z1_var_from_updated_W)
    current_S2_variance_Z1 = np.sum(2 * final_Z1_var_from_updated_W**2 + 4 * final_Z1_var_from_updated_W * final_Z1_mu_from_updated_W**2)
    print(f"E[Sum Z1^2] (S2): {current_S2_mean_Z1:.6f} (Target S2: {target_sum_mean_2_Z1})")
    print(f"Var(Sum Z1^2) (Var(S2)): {current_S2_variance_Z1:.6f} (Target Var(S2): {target_sum_variance_2_Z1})\n")