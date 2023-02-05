import numpy as np
import math

from src.slope.solvers import pgd_slope, pgd_slope_without_n
from src.slope.utils import prox_slope
from src.slope.solvers import prox_slope_new
X = np.array([[1.0, 0.0], [0.0, 1.0]])
y = np.array([5.0, 4.0])
lambdas = np.array([3.0, 1.0])

print("prox_slope:", prox_slope(y,lambdas))# solves 0.5||y-b||^2+ <lambda, b_i> subject to b_1>=b_2>=0
print("prox_slope_new:", prox_slope_new(y,lambdas)) #correct

print("prox_slope_new:", prox_slope_new(np.array([5, 4]), np.array([3, 1])))# different (incorrect) result
print("prox_slope_new:", prox_slope_new([5.0, 4.0], [3.0, 1.0])) # correct
#print("prox_slope:", prox_slope(np.array([5, 4]), np.array([3, 1]))) #gives error



print("pgd_slope:", pgd_slope(X, y, lambdas, fit_intercept=False, gap_tol=1e-4, max_it=10_00, verbose=False,)) # solves 0.5/n*||y-Xb||^2+ <lambda,|b|_(i)>

print("pgd_slope:", pgd_slope(X, y, lambdas/2, fit_intercept=False, gap_tol=1e-4, max_it=10_00, verbose=False,))



#print("pgd_slope_without_n:", pgd_slope_without_n(X, y, lambdas, fit_intercept=False, gap_tol=1e-6, max_it=10_000, verbose=False,)) # solves 0.5*||y-Xb||^2+ <lambda,|b|_(i)>


#a=math.sqrt(2)
#print("pgd_slope:", pgd_slope(a*X, a*y, lambdas, fit_intercept=False, gap_tol=1e-6, max_it=10_000, verbose=False,)) # solves 0.5||y-Xb||^2+ <lambda,|b|_(i)>