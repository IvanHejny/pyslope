import numpy as np
import matplotlib.pyplot as plt
from src.slope.solvers import*


# Constants
N = 2000  # Number of experiments
n = 1000  # Largest sample size per experiment
n_values = [100, 500, n]  # Different sample sizes (n1, n2, n3, n)
beta = np.array([2.0, 2.0])  # 2-dimensional vector
C = np.array([[1, 0], [0, 1]])  # Covariance matrix
sigma = 1
p = len(beta)

alphas = np.linspace(0, 1, 6)  # 8 points between 0 and 3
rmse_dict = {n_val: np.zeros(len(alphas)) for n_val in n_values}  # Store RMSEs
rmse = np.zeros(len(alphas))
print('rmse_dict:', rmse_dict)
for _ in range(N):
    # Generate full dataset (X, eps) once per experiment
    X_full = np.random.multivariate_normal(np.zeros(p), C, size=n)
    eps_full = np.random.multivariate_normal(np.zeros(n), sigma * np.eye(n))
    y_full = X_full @ beta + eps_full

    for n_val in n_values:
        X = X_full[:n_val]  # Take first n_val samples
        eps = eps_full[:n_val]
        y = y_full[:n_val]

        for idx, alpha in enumerate(alphas):
            lambdas = alpha * np.array([0.9, 0.3])  # Regularization parameter

            beta_hat = pgd_slope(X, y, lambdas / np.sqrt(n_val), fit_intercept=False, gap_tol=1e-3, max_it=1000)['beta']
            u_n_hat = np.sqrt(n_val) * (beta_hat - beta)
            rmse_dict[n_val][idx] += np.linalg.norm(u_n_hat) ** 2

    for idx in range(len(alphas)):
        alpha = alphas[idx]
        W_n = X_full.T @ eps_full / np.sqrt(n_val)
        u_hat = pgd_slope_b_0_FISTA(C, W_n, beta, alpha*lambdas)
        rmse[idx] += np.linalg.norm(u_hat) ** 2


# Compute final RMSE values
for n_val in n_values:
    rmse_dict[n_val] = np.sqrt(rmse_dict[n_val] / N)
rmse = np.sqrt(rmse / N)

# Plot results
plt.figure(figsize=(8, 6))
for n_val in n_values:
    plt.plot(alphas, rmse_dict[n_val], label=f"RMSE (n={n_val})", marker='o')
plt.plot(alphas, rmse, label="RMSE $n=\infty$", marker='o')
plt.xlabel(r"$\alpha$")
plt.ylabel("RMSE")
plt.legend()
plt.title("RMSE vs Alpha for Different Sample Sizes")
plt.grid()
plt.show()