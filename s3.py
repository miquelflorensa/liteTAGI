import numpy as np

def update_independent_gaussians_for_sum_target_3(
    initial_means,
    initial_variances,
    target_sum_mean_3,
    target_sum_variance_3
):
    # Calculate the mean of Z_i^3 for each Z_i
    # E[Z_i^3] = mu_Z_i^3 + 3 * mu_Z_i * sigma_Z_i^2
    mu_Z_cubed = initial_means**3 + 3 * initial_means * initial_variances

    # Calculate the variance of Z_i^3 for each Z_i
    # Var(Z_i^3) = 9 * mu_Z_i^4 * sigma_Z_i^2 + 36 * mu_Z_i^2 * sigma_Z_i^4 + 15 * sigma_Z_i^6
    var_Z_cubed = (
        9 * initial_means**4 * initial_variances
        + 36 * initial_means**2 * initial_variances**2
        + 15 * initial_variances**3
    )
    
    # Calculate the current sum mean and sum variance for S3 = sum(Z_i^3)
    # Assuming independence, E[S3] = sum(E[Z_i^3]) and Var(S3) = sum(Var(Z_i^3))
    mu_S3 = np.sum(mu_Z_cubed)
    sigma2_S3 = np.sum(var_Z_cubed)

    # Calculate the Jacobian term Jz for the update
    # Based on the original code's pattern, Jz = cov(Z_i, Z_i^3) / Var(S3)
    # cov(Z_i, Z_i^3) = 3 * sigma_Z_i^2 * (mu_Z_i^2 + sigma_Z_i^2)
    numerator_Jz = 3 * initial_variances * (initial_means**2 + initial_variances)
    
    Jz = numerator_Jz / sigma2_S3

    # Update the means and variances using the EKF-like update rule
    updated_means = initial_means + Jz * (target_sum_mean_3 - mu_S3)
    # The variance update term follows the pattern from the original code (Jz**2)
    updated_variances = initial_variances + Jz**2 * (target_sum_variance_3 - sigma2_S3)

    # Ensure variances remain positive
    updated_variances = np.maximum(updated_variances, 1e-9) 
    # Ensure means are within a reasonable numerical range to prevent issues
    updated_means = np.minimum(np.maximum(updated_means, -1e9), 1e9)  

    return updated_means, updated_variances


if __name__ == "__main__":
    A = 1000 # Number of Z variables

    # Initial parameters for Z_i
    # Using slightly different initial distributions for better numerical stability with cubic terms
    initial_means = np.random.normal(0.5, 0.2, A) # Mean around 0.5, small std
    initial_variances = np.random.uniform(0.1, 0.5, A) # Variances between 0.1 and 0.5

    # Target values for the sum of cubes S3
    # These targets are chosen to be representative for initial means around 0.5 and variances around 0.3
    target_sum_mean_3 = A * 2
    target_sum_variance_3 = A * 6

    print(f"Initial Means: {initial_means[:5]}...") # Display first few
    print(f"Initial Variances: {initial_variances[:5]}...") # Display first few

    num_iterations = 1

    updated_means, updated_variances = initial_means, initial_variances

    for i in range(num_iterations):
        updated_means, updated_variances = update_independent_gaussians_for_sum_target_3(
            updated_means,
            updated_variances,
            target_sum_mean_3,
            target_sum_variance_3
        )
    
    # Recalculate moments after update to show results
    final_mu_Z_cubed = updated_means**3 + 3 * updated_means * updated_variances
    final_var_Z_cubed = (
        9 * updated_means**4 * updated_variances
        + 36 * updated_means**2 * updated_variances**2
        + 15 * updated_variances**3
    )

    print(f"\nUpdated Means: {updated_means[:5]}...")
    print(f"Updated Variances: {updated_variances[:5]}...")
    print(f"\nInitial Sum of Cubes (E[S3]): {np.sum(initial_means**3 + 3 * initial_means * initial_variances)}")
    print(f"Initial Variance of Sum of Cubes (Var(S3)): {np.sum(9 * initial_means**4 * initial_variances + 36 * initial_means**2 * initial_variances**2 + 15 * initial_variances**3)}\n")
    print(f"Target Sum Mean (S3): {target_sum_mean_3}")
    print(f"Target Sum Variance (S3): {target_sum_variance_3}\n")
    print(f"Updated Sum of Cubes (E[S3]): {np.sum(final_mu_Z_cubed)}")
    print(f"Updated Variance of Sum of Cubes (Var(S3)): {np.sum(final_var_Z_cubed)}\n")