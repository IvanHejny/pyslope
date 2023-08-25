import numpy as np
from src.slope.solvers import*
#import math

#from src.slope.solvers import pgd_slope, pgd_slope_without_n
#from src.slope.utils import prox_slope
#from src.slope.solvers import prox_slope_new
'''
X = np.array([[1.0, 0.0], [0.0, 1.0]])
y = np.array([5.0, 4.0])
lambdas = np.array([3.0, 1.0])

print("prox_slope:", prox_slope(y,lambdas))# solves 0.5||y-b||^2+ <lambda, b_i> subject to b_1>=b_2>=0
print("prox_slope_new:", prox_slope_new(y,lambdas)) #correct

#print("prox_slope_new:", prox_slope_new(np.array([5, 4]), np.array([3, 1])))# different (incorrect) result
#print("prox_slope_new:", prox_slope_new([5.0, 4.0], [3.0, 1.0])) # correct
#print("prox_slope:", prox_slope(np.array([5, 4]), np.array([3, 1]))) #gives error



#print("pgd_slope:", pgd_slope(X, y, lambdas, fit_intercept=False, gap_tol=1e-4, max_it=10_00, verbose=False,)) # solves 0.5/n*||y-Xb||^2+ <lambda,|b|_(i)>
#print("pgd_slope:", pgd_slope(X, y, lambdas/2, fit_intercept=False, gap_tol=1e-4, max_it=10_00, verbose=False,))



#print("pgd_slope_without_n:", pgd_slope_without_n(X, y, lambdas, fit_intercept=False, gap_tol=1e-6, max_it=10_000, verbose=False,)) # solves 0.5*||y-Xb||^2+ <lambda,|b|_(i)>
#a=math.sqrt(2)
#print("pgd_slope:", pgd_slope(a*X, a*y, lambdas, fit_intercept=False, gap_tol=1e-6, max_it=10_000, verbose=False,)) # solves 0.5||y-Xb||^2+ <lambda,|b|_(i)>
'''


C1 = np.array([[1, 0.5], [0.5, 1]])
W1 = np.array([5.0, 4.0])
lambdas1 = np.array([0.9, 0.2])
b0_test1 = np.array([1, 1])
stepsize_t = 0.35 # to guarantee convergence take stepsize < 1/max eigenvalue of C (max eval of C is the Lipschitz constant of grad(1/2 uCu - uW)=(Cu-W))
print("pdg_slope_b_0_ISTA_x:", pgd_slope_b_0_ISTA( C = C1, W = W1, b_0 = b0_test1, lambdas = lambdas1, t = 0.35, n = 50))
print("pdg_slope_b_0_FISTA_x:", pgd_slope_b_0_FISTA( C = C1, W = W1, b_0 = b0_test1, lambdas = lambdas1, t = 0.35, n = 50))


C2 = np.identity(4)
b0_test2 = np.array([1, 1, -1, 1])
W2 = np.array([60.0, 50.0, -5.0, 10.0])
lambdas2 = np.array([65.0, 42.0, 40.0, 40.0])
print("pdg_slope_b_0_ISTA_x:", pgd_slope_b_0_ISTA( C = C2, W = W2, b_0 = b0_test2, lambdas = lambdas2, t = 0.35, n = 50))
print("pdg_slope_b_0_FISTA_x:", pgd_slope_b_0_FISTA( C = C2, W = W2, b_0 = b0_test2, lambdas = lambdas2, t = 0.35, n = 50))


