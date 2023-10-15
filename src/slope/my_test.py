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
#print("pdg_slope_b_0_ISTA_x:", pgd_slope_b_0_ISTA( C = C1, W = W1, b_0 = b0_test1, lambdas = lambdas1, t = 0.35, n = 50))
#print("pdg_slope_b_0_FISTA_x:", pgd_slope_b_0_FISTA( C = C1, W = W1, b_0 = b0_test1, lambdas = lambdas1, t = 0.35, n = 50))








'''
C2 = np.identity(4)
b0_test2 = np.array([1, 1, -1, 1])
W2 = np.array([60.0, 50.0, -5.0, 10.0])
lambdas2 = np.array([65.0, 42.0, 40.0, 40.0])
#print("pdg_slope_b_0_ISTA_x:", pgd_slope_b_0_ISTA( C = C2, W = W2, b_0 = b0_test2, lambdas = lambdas2, t = 0.35, n = 50))
#print("pdg_slope_b_0_FISTA_x:", pgd_slope_b_0_FISTA( C = C2, W = W2, b_0 = b0_test2, lambdas = lambdas2, t = 0.35, n = 50))
'''


# Sampling u_hat from the limiting distribution, with given C = [[1,alpha][alpha,1]] and lambda = [3,2] and b_0 = [1,0].
# Testing the irrepresantability condition. This is satisfied for iff E[Z] on the interior of the line (3,2)-(3,-2), which is iff alpha < 2/3.
# For alpha <2/3 and lambda big, we will recover the true pattern with high probability.
alpha = 2/3 - 0.01 # for b0 = [1,0], lambda = [3,2], alpha = 2/3 is the critical threshold for perfect pattern recovery
C3 = np.array([[1, alpha], [alpha, 1]])
W1 = np.array([5.0, 4.0])
lambdas3 = np.array([0.3, 0.2])
const = 1 # for const = 0 we get OLS estimator, which minimizes asymptotic MSE but has zero pattern recovery success
b0_test3 = np.array([1, 1])
stepsize_t = 0.35 # to guarantee convergence take stepsize < 1/max eigenvalue of C (max eval of C is the Lipschitz constant of grad(1/2 uCu - uW)=(Cu-W))
#print("pdg_slope_b_0_ISTA_x:", pgd_slope_b_0_ISTA( C = C1, W = W1, b_0 = b0_test1, lambdas = lambdas1, t = 0.35, n = 50))

def patternMSE(b_0, C, lambdas, n):
    p = len(b_0)
    b_0 = pattern(b_0)
    correct_recovery = 0
    MSE = 0
    for i in range(n):
        W = np.random.multivariate_normal(np.zeros(p), C)
        sol = pgd_slope_b_0_FISTA(C=C, W=W, b_0=b_0, lambdas = lambdas, t=0.35, n=30)
        norm2 = np.linalg.norm(sol) ** 2
        MSE = MSE + norm2
        #print("pdg_slope_b_0_FISTA_x:", sol, norm2)
        if all(pattern(b_0 + 0.0001 * sol) == b_0):
            correct_recovery = correct_recovery + 1
    #print('pattern recovery proportion + MSE')
    return(correct_recovery / n, MSE / n)
    #print('proportion of correct recoveries is', correct_recovery / n)
    #print('MSE is', MSE / n)

alpha = 2/3
print(patternMSE(b_0 = np.array([1, 0]), C = np.array([[1, alpha], [alpha, 1]]), lambdas = 10*np.array([0.3, 0.3]), n = 100))

print(pattern(np.array([2.2,0])))

import matplotlib.pyplot as plt


# Define the range of x values
x = np.linspace(0, 1, 20)  # Generates 10 points between 0 and 5

# Calculate y values for each function
alpha = 2/3
y1 = np.empty(shape=(0,))
y2 = np.empty(shape=(0,))
y1Lasso = np.empty(shape=(0,))
y2Lasso = np.empty(shape=(0,))
for i in range(len(x)):
    result = patternMSE(b_0 = np.array([1, 0]), C = np.array([[1, alpha], [alpha, 1]]), lambdas = x[i]*np.array([0.8, 1.2]), n = 500)
    resultLasso = patternMSE(b_0 = np.array([1, 0]), C = np.array([[1, alpha], [alpha, 1]]), lambdas = x[i]*np.array([1, 1]), n = 500)
    y1 = np.append(y1, result[0])
    y2 = np.append(y2, result[1])
    y1Lasso = np.append(y1Lasso, resultLasso[0])
    y2Lasso = np.append(y2Lasso, resultLasso[1])
print(y1Lasso)
#print(y2Lasso)

# Plot the functions on the same graph
plt.figure(figsize=(3, 6))
plt.plot(x, y1, label='SLOPE pattern recovery', color='blue')  # Plot probability of pattern recovery by SLOPE in blue
plt.plot(x, y2, label='SLOPE MSE', color='green')  # Plot MSE of SLOPE in green
plt.plot(x, y1Lasso, label='Lasso pattern recovery', color='red')
plt.plot(x, y2Lasso, label='Lasso MSE', color='orange') # Plot MSE of Lasso in red
plt.xlabel('penalty scaling $\sigma$')
plt.ylabel('Performance')
plt.title('Pattern Recovery and MSE')
caption_text = 'Figure 1. Asymptotic performance of SLOPE and Lasso in terms of MSE and Pattern recovery, with $patt(b^0) = [1, 0]$, $C$ a correlation matrix with off diagonal entry $2/3$, and penalty $\lambda = \sigma[0.8, 1.2]$ (SLOPE) and $\lambda = \sigma[1, 1]$ (Lasso).'
#plt.figtext(0.5, 0.01, caption_text, wrap =True, ha='center', va='center', fontsize=10, color='black')
plt.figtext(0.5, +0.02, caption_text, wrap=True, horizontalalignment='center', fontsize=10)
plt.legend()  # Show legend to differentiate between functions
plt.grid(True)
plt.tight_layout()
plt.show()



'''
n = 200

correct_recovery = 0
MSE = 0
for i in range(n):
    W3 = np.random.multivariate_normal(np.zeros(2), C3)
    sol = pgd_slope_b_0_FISTA( C = C3, W = W3, b_0 = b0_test3, lambdas = const*lambdas3, t = 0.35, n = 50)
    norm2 = np.linalg.norm(sol)**2
    MSE = MSE + norm2
    print("pdg_slope_b_0_FISTA_x:", sol, norm2)
    if all(pattern(b0_test3 + 0.0001 * sol) == pattern(b0_test3)):
        correct_recovery = correct_recovery + 1
print('proportion of correct recoveries is', correct_recovery/n)
print('MSE is', MSE/n)
'''


#print([1,2]/2)
'''
from scipy.stats import norm
C4 = np.identity(10)
rho = 0.2
C5 = rho * np.identity(10) + (1-rho) * np.ones((10, 10))
print(C5)
b0_test4 = np.concatenate((np.zeros(5), np.ones(5))).astype(int) # pattern vector
print(b0_test4)
b0_test5 = pattern(np.array([0,0,0,1,1,2,2,-2,3, -3]))
#print(type(b0_test4))
#print(type(b0_test3))
# Specify the quantiles you want (e.g., 10 equidistant quantiles given by the BH sequence)
p = 10
q = 0.15
quantiles = np.linspace(1-(p-1)*q/(2*p), 1-q/(2*p), p)
# Calculate the p BH coefficients for a given level q
sigma = 1
lambdas_BH = norm.ppf(quantiles)
#print(type(lambdas_BH))
#print(type(lambdas3))
W4 = np.random.multivariate_normal(np.zeros(p), C4)
print("pdg_slope_b_0_FISTA_x:", pgd_slope_b_0_FISTA( C = C4, W = W4, b_0 = b0_test4, lambdas = 10*lambdas_BH, t = 0.35, n = 50))
#for i in range(n):
#    W4 = np.random.multivariate_normal(np.zeros(p), C4)
#    sol = pgd_slope_b_0_FISTA( C = C4, W = W4, b_0 = b0_test4, lambdas = lambdas_BH, t = 0.35, n = 50)
#    print("pdg_slope_b_0_FISTA_x:", sol)

n = 100
correct_recovery = 0
for i in range(n):
    W4 = np.random.multivariate_normal(np.zeros(p), C5)
    u_hat = pgd_slope_b_0_FISTA( C = C4, W = W4, b_0 = b0_test4, lambdas = 3*lambdas_BH, t = 0.35, n = 50)
    print("pdg_slope_b_0_FISTA_x:", u_hat, pattern(u_hat))

    patt_b0 = pattern(b0_test4)
    patt_bhat = pattern(b0_test4 + 0.01 * pattern(u_hat))
    if all(patt_bhat_i == patt_b0_i for patt_bhat_i, patt_b0_i in zip(patt_bhat,patt_b0)):
        correct_recovery = correct_recovery + 1
print('proportion of correct recoveries is', correct_recovery/n)
'''
