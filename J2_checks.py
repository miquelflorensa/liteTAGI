import numpy as np

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
    print(f"Jz: {Jz}")

    updated_means = initial_means + Jz * (target_sum_mean - mu_S)
    updated_variances = initial_variances + Jz * (target_sum_variance - sigma2_S)

    # Ensure variances remain positive
    updated_variances = np.maximum(updated_variances, epsilon)

    return updated_means, updated_variances

def update_gaussians_for_sum_target_full_cov(
    initial_means,
    initial_covariance,
    target_sum_mean=0.0,
    target_sum_variance=1.0
):
    """
    Update jointly Gaussian variables so their sum matches target mean and variance.
    
    Parameters:
        initial_means: Array of initial means [μ1, μ2, ...]
        initial_covariance: Full covariance matrix [[σ1², σ12, ...], [σ21, σ2², ...], ...]
        target_sum_mean: Desired mean for the sum (default: 0)
        target_sum_variance: Desired variance for the sum (default: 1)
        
    Returns:
        updated_means: Adjusted means
        updated_covariance: Adjusted covariance matrix
    """
    n = len(initial_means)
    h = np.ones(n)  # Sum vector
    
    # Calculate current sum statistics
    current_sum_mean = np.sum(initial_means)
    current_sum_variance = h @ initial_covariance @ h
    
    # Compute Kalman gain (J)
    sum_variance_safe = max(current_sum_variance, 1e-9)
    J = initial_covariance @ h / sum_variance_safe
    
    # Update means
    mean_diff = target_sum_mean - current_sum_mean
    updated_means = initial_means + J * mean_diff
    
    # Update covariance
    variance_diff = target_sum_variance - current_sum_variance
    updated_covariance = initial_covariance + np.outer(J, J) * variance_diff
    
    return updated_means, updated_covariance

# Example usage
if __name__ == "__main__":
    # Example Full Covariance
    means = np.array([1.0, 0.5])
    covariance = np.array([[0.5, 1.0],
                          [0.3, 3.5]])

    new_means, new_cov = update_gaussians_for_sum_target_full_cov(
        initial_means=means,
        initial_covariance=covariance,
        target_sum_mean=0.0,
        target_sum_variance=1.0
    )
    
    print("Example with 2 variables:")
    print(f"Initial means: {means}")
    print(f"Initial covariance:\n{covariance}")
    print(f"\nUpdated means: {new_means}")
    print(f"Updated covariance:\n{new_cov}")
    print(f"\nSum mean: {np.sum(means):.6f} (before update)")
    print(f"Sum variance: {np.sum(covariance)} (before update)")
    print(f"Sum mean: {np.sum(new_means):.6f} (target: 0)")
    print(f"Sum variance: {np.sum(new_cov)} (target: 1)")

    # Example Independent Gaussians
    initial_means = np.array([1.0, 0.5])
    initial_variances = np.array([0.5, 1.0])
    target_sum_mean = 0.0
    target_sum_variance = 1.0   

    updated_means, updated_variances = update_independent_gaussians_for_sum_target(
        initial_means,
        initial_variances,
        target_sum_mean,
        target_sum_variance
    )   

    print("\nExample with independent Gaussians:")
    print(f"Initial means: {initial_means}")
    print(f"Initial variances: {initial_variances}")
    print(f"\nUpdated means: {updated_means}")
    print(f"Updated variances: {updated_variances}")
    print(f"\nSum mean: {np.sum(initial_means):.6f} (before update)")
    print(f"Sum variance: {np.sum(initial_variances):.6f} (before update)")
    print(f"Sum mean: {np.sum(updated_means):.6f} (target: 0)")
    print(f"Sum variance: {np.sum(updated_variances):.6f} (target: 1)")
    