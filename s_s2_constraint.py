import numpy as np

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

    Jz = 2 * initial_means * initial_variances / sigma2_S2

    updated_means = initial_means + Jz * (target_sum_mean_2 - mu_S2)
    updated_variances = initial_variances + Jz**2 * (target_sum_variance_2 - sigma2_S2)

    return updated_means, updated_variances

def update_independent_gaussians_for_sum_target(
    initial_means,
    initial_variances,
    target_sum_mean,
    target_sum_variance
):
    mu_S = np.sum(initial_means)
    sigma2_S = np.sum(initial_variances)

    Jz = initial_variances / sigma2_S

    updated_means = initial_means + Jz * (target_sum_mean - mu_S)
    updated_variances = initial_variances + Jz * (target_sum_variance - sigma2_S)

    return updated_means, updated_variances

def iterative_joint_update_independent_gaussians(
    initial_means,
    initial_variances,
    target_sum_mean,
    target_sum_variance,
    target_sum_mean_2,
    target_sum_variance_2,
    max_iterations=1000,
    tolerance=1e-6
):
    """
    Performs an iterative joint update of means and variances for independent
    Gaussian random variables to simultaneously hit target mean/variance
    for their sum (S) and sum of squares (S2). This is an approximation
    as it applies the updates sequentially.

    Args:
        initial_means (np.array): 1D array of initial means for Z_i.
        initial_variances (np.array): 1D array of initial variances for Z_i.
        target_sum_mean (float): Desired mean for the sum S.
        target_sum_variance (float): Desired variance for the sum S.
        target_sum_mean_2 (float): Desired mean for the sum of squares S2.
        target_sum_variance_2 (float): Desired variance for the sum of squares S2.
        max_iterations (int): Maximum number of iterations for convergence.
        tolerance (float): Convergence tolerance for changes in means and variances.

    Returns:
        tuple: A tuple containing:
            - updated_means (np.array): The updated means for Z_i.
            - updated_variances (np.array): The updated variances for Z_i.
    """
    current_means = np.copy(initial_means)
    current_variances = np.copy(initial_variances)

    print("Starting iterative joint update...")

    for i in range(max_iterations):
        prev_means = np.copy(current_means)
        prev_variances = np.copy(current_variances)

        # Apply update for S
        current_means, current_variances = update_independent_gaussians_for_sum_target(
            current_means, current_variances, target_sum_mean, target_sum_variance
        )

        # Apply update for S2
        current_means, current_variances = update_independent_gaussians_for_sum_target_2(
            current_means, current_variances, target_sum_mean_2, target_sum_variance_2
        )

        # Check for convergence
        mean_diff = np.max(np.abs(current_means - prev_means))
        var_diff = np.max(np.abs(current_variances - prev_variances))

        if mean_diff < tolerance and var_diff < tolerance:
            print(f"Converged after {i+1} iterations.")
            break
    else:
        print(f"Did not converge after {max_iterations} iterations. Result may be an approximation.")

    return current_means, current_variances

if __name__ == "__main__":
    print("--- Iterative Joint Update Example ---")
    A = 10 # Number of Z variables
    initial_means = np.random.normal(0, 1, A)
    initial_variances = np.random.normal(0.1, 5, A) # Small positive variances
    initial_variances = np.abs(initial_variances) # Ensure positive variances

    # Target values for the sum S
    target_sum_mean = 0.0
    target_sum_variance = A 

    # Target values for the sum of squares S2
    target_sum_mean_2 = 2.0 * A
    target_sum_variance_2 = 6.0 * A 

    updated_means, updated_variances = iterative_joint_update_independent_gaussians(
        initial_means,
        initial_variances,
        target_sum_mean,
        target_sum_variance,
        target_sum_mean_2,
        target_sum_variance_2,
        max_iterations=100, # Increased iterations for better convergence
        tolerance=1e-3 # Stricter tolerance
    )

    print(f"\n--- Initial Means and Variances ---")
    print(f"Initial Means: {initial_means}")
    print(f"Initial Variances: {initial_variances}\n")

    print(f"--- Updated Means and Variances ---")
    print(f"Updated Means: {updated_means}")
    print(f"Updated Variances: {updated_variances}\n")

    print(f"\n--- Results After Iterative Joint Update ---")
    print(f"Initial Means Sum (S): {np.sum(initial_means)}")
    print(f"Initial Variances Sum (Var(S)): {np.sum(initial_variances)}")
    print(f"Initial Sum of Squares (E[S2]): {np.sum(initial_means**2 + initial_variances)}")
    print(f"Initial Variance of Sum of Squares (Var(S2)): {np.sum(2 * initial_variances**2 + 4 * initial_variances * initial_means**2)}\n")

    # Verify updated sum S properties
    current_S_mean = np.sum(updated_means)
    current_S_variance = np.sum(updated_variances) 

    # Verify updated sum of squares S2 properties
    current_S2_mean = np.sum(updated_means**2 + updated_variances)
    current_S2_variance = np.sum(2 * updated_variances**2 + 4 * updated_variances * updated_means**2)

    print(f"Sum of Updated Means (S): {current_S_mean:.6f} (Target: {target_sum_mean})")
    print(f"Sum of Updated Variances (Var(S)): {current_S_variance:.6f} (Target: {target_sum_variance})\n")
    print(f"Sum of Updated Means Squared (E[S2]): {current_S2_mean:.6f} (Target: {target_sum_mean_2})")
    print(f"Variance of Sum of Updated Squares (Var(S2)): {current_S2_variance:.6f} (Target: {target_sum_variance_2})\n")
    