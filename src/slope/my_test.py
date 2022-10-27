import numpy as np

from src.slope.solvers import pgd_slope
from src.slope.utils import prox_slope
X = np.array([[1.0, 0.0], [0.0, 1.0]])
y = np.array([5.0, 4.0])
lambdas = np.array([3.0, 1.0])

print(prox_slope(y,lambdas))# solves 0.5||y-b||^2+ <lambda, b_i> subject to b_1>=b_2>=0
print(pgd_slope(X, y, lambdas, fit_intercept=False, gap_tol=1e-6, max_it=10_000, verbose=False,)) # solves 0.5||y-Xb||^2+ <lambda,|b|_(i)> ??

