import numpy as np
from scipy.stats import norm

def update_independent_gaussians_for_sum_target_2(
    initial_means,
    initial_variances,
    target_sum_mean_2,
    target_sum_variance_2
):
    epsilon = 1e-9  # Small value to ensure numerical stability
    positive = initial_means > 0

    # mu_Z = initial_means * positive + epsilon
    # var_Z = initial_variances * positive + epsilon
    # cov_Z = (1.0 * positive + epsilon) * initial_variances

    # cdfn = np.maximum(1E-20,norm.cdf(mZ/sZ))
    # pdfn = np.maximum(1E-20,norm.pdf(mZ/sZ))
    # mM = np.maximum(1E-20,sZ*pdfn + mZ * cdfn)
    # s2M = np.maximum(1E-6,-mM * mM + 2 * mM * mZ - mZ * sZ * pdfn + (s2Z - mZ * mZ) * cdfn)
    # cov_ZM = s2Z * cdfn
    cdfn = np.maximum(1E-20, norm.cdf(initial_means / np.sqrt(initial_variances + epsilon)))
    pdfn = np.maximum(1E-20, norm.pdf(initial_means / np.sqrt(initial_variances + epsilon)))
    mu_Z = np.maximum(1E-20, np.sqrt(initial_variances + epsilon) * pdfn + initial_means * cdfn)
    var_Z = np.maximum(1E-6, -mu_Z * mu_Z + 2 * mu_Z * initial_means - initial_means * np.sqrt(initial_variances + epsilon) * pdfn + (initial_variances - initial_means * initial_means) * cdfn)
    cov_Z = np.sqrt(initial_variances + epsilon) * cdfn


    mu_S2 = np.sum(mu_Z)
    sigma2_S2 = np.sum(var_Z)

    # Jz = 2 * initial_means * initial_variances / sigma2_S2
    Jz = cov_Z / sigma2_S2

    updated_means = initial_means + Jz * (target_sum_mean_2 - mu_S2)
    updated_variances = initial_variances + Jz * (target_sum_variance_2 - sigma2_S2)

    # Ensure variances remain positive
    updated_variances = np.maximum(updated_variances, 1e-9) 


    return updated_means, updated_variances


if __name__ == "__main__":
    A = 10 # Number of Z variables
    initial_means = np.random.normal(0, 0.1, A)
    initial_variances = np.random.normal(0.01, 0.2, A) # Small positive variances
    initial_variances = np.abs(initial_variances) # Ensure positive variances
    # initial_means = np.array([2.0])
    # initial_variances = np.array([3.0])

    # Target values for the sum S
    target_sum_mean = 0.0
    target_sum_variance = A 

    # Target values for the sum of squares S2
    target_sum_mean_2 = 2.0 * A
    target_sum_variance_2 = 6.0 * A 

    print(f"Initial Means: {initial_means}")
    print(f"Initial Variances: {initial_variances}")

    num_iterations = 100

    updated_means, updated_variances = initial_means, initial_variances

    for i in range(num_iterations):
        updated_means, updated_variances = update_independent_gaussians_for_sum_target_2(
            updated_means,
            updated_variances,
            target_sum_mean_2,
            target_sum_variance_2
        )
    print(f"\nUpdated Means: {updated_means}")
    print(f"Updated Variances: {updated_variances}")
    # print(f"\nInitial Sum of Squares (E[S2]): {np.sum(initial_means**2 + initial_variances)}")
    # print(f"Initial Variance of Sum of Squares (Var(S2)): {np.sum(2 * initial_variances**2 + 4 * initial_variances * initial_means**2)}\n")
    print(f"Target Sum Mean (S2): {target_sum_mean_2}")
    print(f"Target Sum Variance (S2): {target_sum_variance_2}\n")
    # print(f"Updated Sum of Squares (E[S2]): {np.sum(updated_means**2 + updated_variances)}")
    # print(f"Updated Variance of Sum of Squares (Var(S2)): {np.sum(2 * updated_variances**2 + 4 * updated_variances * updated_means**2)}\n")
    # positive = updated_means > 0
    # mu_Z = updated_means * positive
    # var_Z = updated_variances * positive

    cdfn = np.maximum(1E-20, norm.cdf(updated_means / np.sqrt(updated_variances + 1e-9)))
    pdfn = np.maximum(1E-20, norm.pdf(updated_means / np.sqrt(updated_variances + 1e-9)))
    mu_Z = np.maximum(1E-20, np.sqrt(updated_variances + 1e-9) * pdfn + updated_means * cdfn)
    var_Z = np.maximum(1E-6, -mu_Z * mu_Z + 2 * mu_Z * updated_means - updated_means * np.sqrt(updated_variances + 1e-9) * pdfn + (updated_variances - updated_means * updated_means) * cdfn)
    cov_Z = np.sqrt(updated_variances + 1e-9) * cdfn

    mu_S2 = np.sum(mu_Z)
    sigma2_S2 = np.sum(var_Z)

    print(f"Updated Sum of Squares (E[S2]): {mu_S2}")
    print(f"Updated Variance of Sum of Squares (Var(S2)): {sigma2_S2}\n")

