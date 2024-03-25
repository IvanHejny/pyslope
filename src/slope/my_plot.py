import numpy as np
from src.slope.solvers import*
from admm_glasso import*
#import math
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.interpolate import PchipInterpolator


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

'''
def patternMSE(b_0, C, lambdas, n, Cov = None):
    if Cov is None:
        Cov = C
    p = len(b_0)
    b_0 = pattern(b_0)
    correct_pattern_recovery = 0
    correct_support_recovery = 0
    MSE = 0
    dim_reduction = 0
    for i in range(n):
        W = np.random.multivariate_normal(np.zeros(p), Cov)
        u_hat = pgd_slope_b_0_FISTA(C=C, W=W, b_0=b_0, lambdas=lambdas, t=0.35, n=30)
        norm2 = np.linalg.norm(u_hat) ** 2
        MSE = MSE + norm2
        dim_reduction = dim_reduction + p-len(np.unique(u_hat))
        #print("pdg_slope_b_0_FISTA_x:", u_hat, np.sqrt(norm2))
        if all(pattern(b_0 + 0.0001 * u_hat) == b_0):
            correct_pattern_recovery = correct_pattern_recovery + 1
        if all(np.sign(pattern(b_0 + 0.0001 * u_hat)) == np.sign(b_0)):
            correct_support_recovery = correct_support_recovery + 1
    #print('pattern recovery proportion + MSE')
    return(np.sqrt(MSE / n), correct_pattern_recovery / n , correct_support_recovery /n, dim_reduction/n)
    #print('proportion of correct recoveries is', correct_pattern_recovery / n)
'''


def patternMSE(b_0, C, lambdas, n, Cov=None, genlasso=False, A = None):
    """
    Calculate mean squared error (MSE), probability of pattern and support recovery, and dimension reduction
    for a given pattern vector 'b_0', covariance C, and penalty sequence lambdas, using FISTA algorithm for SLOPE.

    Parameters:
        b_0 (array-like): True pattern vector to recover.
        C (array-like): Covariance matrix used in the optimization.
        lambdas (array-like): Regularization parameter for SLOPE optimization.
        n (int): Number of simulations to run for performance evaluation.
        Cov (array-like, optional): Covariance matrix used for data generation. Defaults to None,
                                    which means it's the same as 'C'.

    Returns:
        tuple: A tuple containing three performance metrics:
            - Root Mean Squared Error (RMSE) of the optimization result.
            - Proportion of correct pattern recoveries.
            - Proportion of correct support recoveries.
            - Average dimension reduction.
    """
    if Cov is None:
        Cov = C

    p = len(b_0)
    b_0 = pattern(b_0)
    correct_pattern_recovery = 0
    correct_support_recovery = 0
    MSE = 0
    dim_reduction = 0
    stepsize_t = 1/max(np.linalg.eigvals(C))  # stepsize <= 1/max eigenvalue of C;
    # (max eval of C is the Lipschitz constant of grad(1/2 uCu - uW)=(Cu-W));
    # gives O(1/n^2) convergence;

    for i in range(n):
        # Generate random data with given covariance matrix
        W = np.random.multivariate_normal(np.zeros(p), Cov)

        # Perform SLOPE optimization using PGD and FISTA algorithm
        if genlasso == True:
            if A is None:
                A = Acustom(a=np.ones(len(b_0)), b=np.ones(len(b_0)-1))
            u_hat = admm_glasso(C=C, A=A, w=W, beta0=b_0, lambdas=1.0)
        else:
            u_hat = pgd_slope_b_0_FISTA(C=C, W=W, b_0=b_0, lambdas=lambdas, t=stepsize_t, n=20 )

        # Calculate MSE
        norm2 = np.linalg.norm(u_hat) ** 2
        MSE += norm2

        # Calculate dimension reduction
        dim_reduction += p - len(np.unique(u_hat))

        # Check pattern and support recoveries
        if all(pattern(b_0 + 0.0001 * np.round(u_hat, 3)) == b_0):
            correct_pattern_recovery += 1
        if all(np.sign(pattern(b_0 + 0.0001 * np.round(u_hat, 3))) == np.sign(b_0)):
            correct_support_recovery += 1

    # Calculate average metrics over simulations
    rmse = np.sqrt(MSE / n)
    pattern_recovery_rate = correct_pattern_recovery / n
    support_recovery_rate = correct_support_recovery / n
    avg_dim_reduction = dim_reduction / n

    return rmse, pattern_recovery_rate, support_recovery_rate, avg_dim_reduction


alpha = 0
#print('patternMSE:', patternMSE(b_0 = np.array([1, 0]), C = np.array([[1, alpha], [alpha, 1]]), lambdas = 5*np.array([0.3, 0.3]), n = 100, glasso = True, glasso_penalty=8))


rho = 0.8
#print(rho * np.identity(10) + (1-rho) * np.ones((10, 10)))
C_compound = (1-rho) * np.identity(4) + rho * np.ones((4, 4))
C_block = np.array([[1, rho, 0, 0],
                    [rho, 1, 0, 0],
                    [0, 0, 1, rho],
                    [0, 0, rho, 1]])
C_block1 = np.array([[1, 0, 0, 0],
                     [0, 1, rho, rho],
                     [0, rho, 1, rho],
                     [0, rho, rho, 1]])

array1 = np.ones(1)
#print(array1)
array2 = (1-rho) * np.identity(6) + rho * np.ones((6, 6))
#print(array2)
block_diag_matrix = np.block([[array1, np.zeros((1,6))],
                              [np.zeros((6,1)), array2]])
#print(block_diag_matrix)
compound_block = (1-rho) * np.identity(3) + rho * np.ones((3, 3))
block_diag_matrix9 = np.block([[compound_block, np.zeros((3,3)), np.zeros((3,3))],
                              [np.zeros((3,3)), compound_block, np.zeros((3,3))],
                               [np.zeros((3,3)), np.zeros((3,3)), compound_block]])

print(block_diag_matrix9)

'''
my_W1 = np.array([-1.621111, 1.16656]) #np.random.multivariate_normal(np.zeros(2), np.array([[1, 2/3], [2/3, 1]]))
my_W2 = np.array([-3.1, 1.7])
total = 0
my_matrix = block_diag_matrix9
print('1/max eigval of C', 1/max(np.linalg.eigvals(my_matrix)))
final = pgd_slope_b_0_ISTA(C=np.array([[1, 1/3], [1/3, 1]]), W=my_W1, b_0=np.array([1, 1]), lambdas=np.array([1.2, 0.8]), t=0.33, n=100)
for i in range(2, 100):
     step_i = pgd_slope_b_0_ISTA(C=np.array([[1, 1/3], [1/3, 1]]), W=my_W1, b_0=np.array([1, 1]), lambdas=np.array([1.2, 0.8]), t=0.33, n=i)
     print(i, step_i, final - step_i)
     # print(pgd_slope_b_0_ISTA(C=np.array([[1]]), W=np.array([2.78]), b_0=np.array([1]), lambdas=np.array([0]), t=1.46, n=i))
print('small_step', pgd_slope_b_0_ISTA(C=np.array([[1, 1/3], [1/3, 1]]), W=my_W1, b_0=np.array([1, 1]), lambdas=np.array([1.2, 0.8]), t=0.3, n=100))
# for i in range(2, 100):
# print(pgd_slope_b_0_ISTA(C=np.array([[1, 2/3], [2/3, 1]]), W=my_W2, b_0=np.array([1, 0]), lambdas=np.array([1.2, 0.8]), t=0.31, n=i))
'''

# print('patternMSE:', patternMSE(b_0 = np.array([0, 1, 1, 1]), C = C_block1, lambdas = np.array([1.3, 1.1, 0.9, 0.7]), n = 100, glasso = False, glasso_penalty=40))
# print(patternMSE(b_0 = np.array([0, 1, 1, 1]), C = C_block1, lambdas = np.array([1, 1, 1, 1 ]), n = 100))
# print(patternMSE(b_0 = np.array([0, 0, 1, 1]), C = np.identity(4), lambdas = 15*np.array([1.6, 1.2, 0.8, 0.6]), n = 500, Cov = np.linalg.inv(C_compound))) # 2 step SLOPE, perfect pattern recovery
# print(patternMSE(b_0 = np.array([0, 0, 1, 1]), C = np.identity(4), lambdas = np.array([1, 1, 1, 1]), n = 500, Cov = np.linalg.inv(C_compound))) # 2 STEP Lasso, perfect support recovery
# print('SLOPE_patMSE:', patternMSE(b_0 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), C = block_diag_matrix9, lambdas = 10*np.array([1.4, 1.3, 1.2, 1.1, 1, 0.9, 0.8, 0.7, 0.6]), n = 100))


# print('glasso_patMSE:', patternMSE(b_0=np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
#                                    C=block_diag_matrix9,
#                                    lambdas=0*np.array([1.4, 1.3, 1.2, 1.1, 1, 0.9, 0.8, 0.7, 0.6]),
#                                    Cov=0.2**2*block_diag_matrix9,
#                                    n=100,
#                                    glasso=True,
#                                    A=40*AFLmon(9, 0.1)))




# print(np.sign(pattern(np.array([0,0,-1.3,1.3, 2.7]))))

# p=4 simulations
'''
# Define the range of x values
x = np.linspace(0, 3, 24)  # Generates 10 points between 0 and 5

# Calculate y values for each function

PattSLOPE = np.empty(shape=(0, ))
MseSLOPE = np.empty(shape=(0, ))
PattLasso = np.empty(shape=(0, ))
MseLasso = np.empty(shape=(0, ))
SupportSLOPE = np.empty(shape=(0, ))
SupportLasso = np.empty(shape=(0, ))
for i in range(len(x)):
    resultSLOPE = patternMSE(b_0 = np.array([0, 1, 1, 1]), C = C_block1, lambdas = x[i]*np.array([1.3, 1.1, 0.9, 0.7]), n = 50)
    resultLasso = patternMSE(b_0 = np.array([0, 1, 1, 1]), C = C_block1, lambdas = x[i]*np.array([1, 1, 1, 1 ]), n = 50)
    MseSLOPE = np.append(MseSLOPE, resultSLOPE[0])
    PattSLOPE = np.append(PattSLOPE, resultSLOPE[1])
    SupportSLOPE = np.append(SupportSLOPE, resultSLOPE[2])
    MseLasso = np.append(MseLasso, resultLasso[0])
    PattLasso = np.append(PattLasso, resultLasso[1])
    SupportLasso = np.append(SupportLasso, resultLasso[2])

#print(PattLasso)
#print(MseLasso)

# Plot the functions on the same graph
plt.figure(figsize=(3, 6))
plt.plot(x, MseSLOPE, label='SLOPE RMSE', color='green')  # Plot RMSE of SLOPE in green
plt.plot(x, PattSLOPE, label='SLOPE pattern recovery', color='blue')  # Plot probability of pattern recovery by SLOPE in blue
plt.plot(x, SupportSLOPE, label='SLOPE support recovery', color='black') # Plot prob of support recovery by SLOPE in black
plt.plot(x, MseLasso, label='Lasso RMSE', color='orange') # Plot RMSE of Lasso in red
plt.plot(x, PattLasso, label='Lasso pattern recovery', color='red') # Plot prob of pattern by Lasso in red
plt.plot(x, SupportLasso, label='Lasso support recovery', color='purple') # Plot prob of support recovery by Lasso in purple

plt.xlabel('penalty scaling $\sigma$')
plt.ylabel('Performance')
plt.title('Pattern Recovery and RMSE')
caption_text = 'Asymptotic performance of SLOPE and Lasso in terms of MSE and Pattern recovery, with $patt(b^0) = [0, 1, 1, 1]$, $C$, and penalty $\lambda = \sigma[1.3, 1.1, 0.9, 0.7]$ (SLOPE) and $\lambda = \sigma[1, 1, 1, 1]$ (Lasso).'
#plt.figtext(0.5, 0.01, caption_text, wrap =True, ha='center', va='center', fontsize=10, color='black')
plt.figtext(0.5, +0.02, caption_text, wrap=True, horizontalalignment='center', fontsize=10)
plt.legend()  # Show legend to differentiate between functions
plt.grid(True)
plt.tight_layout()
plt.show()
#'''


def plot_performance(b_0, C, lambdas, x, n, Cov=None, flasso=False, A_flasso = None, glasso=False, A_glasso = None, smooth = None):
    PattSLOPE = np.empty(shape=(0,))
    MseSLOPE = np.empty(shape=(0,))
    SupportSLOPE = np.empty(shape=(0,))

    PattLasso = np.empty(shape=(0,))
    MseLasso = np.empty(shape=(0,))
    SupportLasso = np.empty(shape=(0,))

    Pattglasso = np.empty(shape=(0,))
    Mseglasso = np.empty(shape=(0,))
    Supportglasso = np.empty(shape=(0,))

    Pattflasso = np.empty(shape=(0,))
    Mseflasso = np.empty(shape=(0,))
    Supportflasso = np.empty(shape=(0,))

    for i in range(len(x)):
        resultSLOPE = patternMSE(b_0=b_0, C=C, Cov=Cov, lambdas=x[i] * lambdas, n=n)
        #resultLasso = patternMSE(b_0=b_0, C=C, Cov=Cov, lambdas=x[i] * np.ones(len(b_0)), n=n)
        resultLasso = patternMSE(b_0=b_0, C=C, Cov=Cov, lambdas=x[i] * lambdas, n=n, genlasso=True, A=x[i]*Acustom(a=np.ones(len(b_0)), b=np.zeros(len(b_0)-1)))
        # using admm for lasso instead of pgd_slope_b_0_FISTA

        MseSLOPE = np.append(MseSLOPE, resultSLOPE[0])
        PattSLOPE = np.append(PattSLOPE, resultSLOPE[1])
        #SupportSLOPE = np.append(SupportSLOPE, resultSLOPE[2])

        MseLasso = np.append(MseLasso, resultLasso[0])
        PattLasso = np.append(PattLasso, resultLasso[1])
        #SupportLasso = np.append(SupportLasso, resultLasso[2])

        if flasso == True:
            if A_flasso is None:
                A_flasso = Acustom(a=np.ones(len(b_0)), b=np.ones(len(b_0) - 1))
            resultflasso = patternMSE(b_0=b_0, C=C, Cov=Cov, lambdas=x[i] * lambdas, n=n, genlasso=True,
                                      A=x[i]*A_flasso)
            Mseflasso= np.append(Mseflasso, resultflasso[0])
            Pattflasso = np.append(Pattflasso, resultflasso[1])
            #Supportflasso = np.append(Supportflasso, resultflasso[2])

        if glasso == True:
            if A_glasso is None:
                A_glasso = Acustom(a=np.ones(len(b_0)), b=np.ones(len(b_0) - 1))
            resultglasso = patternMSE(b_0=b_0, C=C, Cov=Cov, lambdas=x[i] * lambdas, n=n, genlasso=True,
                                      A=x[i] * A_glasso)
            Mseglasso= np.append(Mseglasso, resultglasso[0])
            Pattglasso = np.append(Pattglasso, resultglasso[1])
            #Supportglasso = np.append(Supportglasso, resultglasso[2])

        resultOLS = 0.5*(MseSLOPE[0] + MseLasso[0])
    if smooth == True:
        # Spline interpolation for smoother curve
        #x_smooth = np.concatenate((x, np.linspace(x.min(), x.max(), 10*(len(x)-1)+1)))
        #x_smooth = np.sort(x_smooth)
        x_smooth = np.linspace(x.min(), x.max(), 4 * (len(x) - 1) + 1)

        spl1 = PchipInterpolator(x, MseSLOPE)  #
        spl2 = PchipInterpolator(x, PattSLOPE)  #
        spl3 = PchipInterpolator(x, MseLasso)  #

        MseSLOPE = spl1(x_smooth)
        PattSLOPE = spl2(x_smooth)
        MseLasso = spl3(x_smooth)
        if flasso == True:
            spl4 = PchipInterpolator(x, Mseflasso)
            spl5 = PchipInterpolator(x, Pattflasso)
            Mseflasso = spl4(x_smooth)
            Pattflasso = spl5(x_smooth)
        if glasso == True:
            spl6 = PchipInterpolator(x, Mseglasso)
            spl7 = PchipInterpolator(x, Pattglasso)
            Mseglasso = spl6(x_smooth)
            Pattglasso = spl7(x_smooth)

        x = x_smooth

    # Plot the functions on the same graph
    plt.figure(figsize=(6, 6))
    plt.plot(x, MseLasso, label='RMSE Lasso', color='blue', lw=1.5, alpha=0.9)  # Plot RMSE of Lasso
    plt.plot(x, MseSLOPE, label='RMSE SLOPE', color='green', lw=1.5, alpha=0.9)  # Plot RMSE of SLOPE
    plt.plot(x, PattSLOPE, label='recovery SLOPE', color='green', linestyle='dashed', lw=1.5)  # Plot probability of pattern recovery by SLOPE
    if flasso == True:
        plt.plot(x, Mseflasso, label='RMSE FLasso', color='orange', lw=1.5, alpha=0.9)
        plt.plot(x, Pattflasso, label='recovery FLasso', color='orange', linestyle='dashed', lw=1.5)

    if glasso == True:
        plt.plot(x, Mseglasso, label='RMSE ConFLasso', color='purple', lw=1.5, alpha=0.9)
        plt.plot(x, Pattglasso, label='recovery ConFLasso', color='purple', linestyle='dashed', lw=1.5)

    #plt.plot(x, PattLasso, label='pattern recovery Lasso', color='blue', linestyle='dashed', lw=1.5)  # Plot prob of pattern by Lasso
    #plt.plot(x, SupportSLOPE, label='support recovery SLOPE', color='green', linestyle='-.', lw=1.5, alpha=0.5)  # Plot prob of support recovery by SLOPE
    #plt.plot(x, SupportLasso, label='support recovery Lasso', color='blue', linestyle='-.', lw=1.5, alpha=0.5)  # Plot prob of support recovery by Lasso

    plt.scatter(0, resultOLS, color='red', label='RMSE OLS') # Plot RMSE of OLS as a scatter point at 0

    # Increase the size of x-axis and y-axis tick labels
    plt.xticks(fontsize=14)  # font size for x-axis tick labels
    plt.yticks(fontsize=14)  # font size for y-axis tick labels

    plt.xlabel(r'$\alpha$', fontsize=16)  # penalty scaling
    #plt.ylabel('Performance')
    #plt.title('Pattern Recovery and RMSE')
    caption_text = f'$b^0$ = {b_0}, $\lambda = \sigma$ {lambdas}' #compound or block diagonal C block diagonal with one compound 0.8 block for each cluster, and penalty scaling
    #plt.figtext(0.5, 0.01, caption_text, wrap=True, horizontalalignment='center', fontsize=10, color='black')
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True, ncol=3)

    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Example usage:
# Define b_0, C, lambdas, and x before calling the function
# print(np.linspace(0, 3, 9)) #
x = np.linspace(0, 2, 20) # 25 = (9-1)*3+1, linspace (n-1)*k + 1 refines linspace n
#print('x:', np.round(x,2))
x1 = np.linspace(0,1,10)
#print('x1:', x1)
x2 = np.delete(np.linspace(1,2, 5),0)
#print('x2:', x2)
#print('deltest:', np.delete(np.array([1,2,3,]), 2))
xconcat = np.concatenate((x1, x2))
#print('xconcat:', xconcat)

#plot_performance(b_0=np.array([1, 0]), C=np.identity(2), lambdas=np.array([1.4, 0.6]), x=x, n=500)
#plot_performance(b_0=np.array([0, 1]), C=np.array([[1, 0.8], [0.8, 1]]), lambdas=np.array([1.2, 0.8]), x=x, n=2000)
#plot_performance(b_0=np.array([1, 1]), C=np.array([[1, 0.8], [0.8, 1]]), lambdas=np.array([1.2, 0.8]), x=x, n=2000)
#plot_performance(b_0=np.array([0, 0, 1, 1]), C=C_block, lambdas=np.array([1.3, 1.1, 0.9, 0.7]), x=x, n=100)
#plot_performance(b_0=np.array([0, 0, 1, 1]), C=C_block, lambdas=np.array([1.3, 1.1, 0.9, 0.7]), x=x, n=100)
#plot_performance(b_0=np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), C=block_diag_matrix9, lambdas=np.array([1.4, 1.3, 1.2, 1.1, 1, 0.9, 0.8, 0.7, 0.6]), x=x, n=200)

b = np.array([1, 1.15, 1.25, 1.3, 1.3, 1.25, 1.15, 1])
A9Bcustom = Acustom(a=1.4*np.ones(9), b=b[:8])
#A9Bcustom = Acustom(a=np.ones(9), b=np.ones(8))

bump_quadratic = lambda curvature, p: np.array([1+curvature*i*(p-i) for i in range(1, p)])
print('bump_quadratic', bump_quadratic(0,9))

#a_bump = 2*np.max(bump_quadratic(curvature=1, p=9)) + 1
#print('a_bump:', a_bump)
curvature = 0.06  # 0.06  # curvature 0.06 in A_glasso corresp to 1.9 in A_flasso
cluster_scaling = 0.65
A9bump = Acustom(a=np.ones(9), b=cluster_scaling*bump_quadratic(curvature=curvature, p=9))
print('A9bump:\n', np.round(A9bump,3))
flassoA = Acustom(a=np.ones(9), b=np.ones(8)*sum(A9bump[i][i] for i in range(9))*(1/8))
print('flassoA:\n', np.round(flassoA,3))
#print('bump_quadratic:', bump_quadratic(5,9))

# plot_performance(b_0=np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
#                  C=block_diag_matrix9,
#                  lambdas=np.array([1.4, 1.3, 1.2, 1.1, 1, 0.9, 0.8, 0.7, 0.6]),
#                  x=np.linspace(0, 2, 20),  # np.linspace(0.48, 0.55, 10)
#                  n=100,
#                  Cov=0.2**2*block_diag_matrix9,
#                  flasso=True,
#                  A_flasso=flassoA,
#                  #glasso=True,
#                  #A_glasso=A9bump,  #Acustom(a=np.ones(9), b=np.ones(8)), #(1/np.sum(A9Bcustom))*9*A9Bcustom,
#                  smooth=True)



# plot_performance(b_0=np.array([1, 0]),
#                  C=np.array([[1, 0.8], [0.8, 1]]),
#                  lambdas=np.array([1.2, 0.8]),
#                  x=np.linspace(0, 3, 20), #np.linspace(0.48, 0.55, 10)
#                  n=100,
#                  Cov=0.2**2*np.array([[1, 0.8], [0.8, 1]]),
#                  flasso=True,
#                  A_flasso=Acustom(a=np.ones(2), b=np.ones(1)),
#                  glasso=True,
#                  A_glasso=Acustom(a=np.ones(2), b=np.ones(1)),
#                  smooth=True)

rho = 0.8

# plot_performance(b_0=np.array([0, 2, 0, 2]), #interesting [1,1,0,1] slope best, [1,1,1,1] flasso best, [0,1,0,1] slope best [0,1,1,0] lasso best
#                  C=np.array([[1,0,rho,0],[0,1,0,rho],[rho,0,1,0],[0,rho,0,1]]), #(1-rho) * np.identity(4) + rho * np.ones((4, 4)),
#                  lambdas=np.array([1.6, 1.2, 0.8, 0.4]),
#                  x=np.linspace(0,1,20), #np.linspace(0.48, 0.55, 10)
#                  n=100,
#                  Cov=0.4**2*np.array([[1,0,rho,0],[0,1,0,rho],[rho,0,1,0],[0,rho,0,1]]), #(1-rho) * np.identity(4) + rho * np.ones((4, 4)),
#                  flasso=True,
#                  A_flasso=Acustom(a=np.ones(4), b=0.5 * np.ones(3)),
#                  #glasso=True,
#                  #A_glasso=Acustom(a=np.ones(4), b=1 * np.array([0.9, 1.2, 0.9])),
#                  smooth=True)


# plot_performance(b_0=np.array([1, 1, 1, 1]), #interesting [1,1,0,1] slope best, [1,1,1,1] flasso best, [0,1,0,1], [0,0,1,0] lasso best
#                  C=np.array([[1,rho,0,0],[rho,1,0, 0],[0,0,1,rho],[0,0,rho,1]]), #(1-rho) * np.identity(4) + rho * np.ones((4, 4)),
#                  lambdas=np.array([1.6, 1.2, 0.6, 0.4]),
#                  x=np.linspace(0,1,20), #np.linspace(0.48, 0.55, 10)
#                  n=200,
#                  Cov=0.4**2*np.array([[1,rho,0,0],[rho,1,0, 0],[0,0,1,rho],[0,0,rho,1]]), #(1-rho) * np.identity(4) + rho * np.ones((4, 4)),
#                  flasso=True,
#                  A_flasso=Acustom(a=np.ones(4), b=0.5 * np.ones(3)),
#                  #glasso=True,
#                  #A_glasso=Acustom(a=np.ones(4), b=0.6 * np.array([0.9, 1.2, 0.9])),
#                  smooth=True)

# rho = 0.5
# plot_performance(b_0=np.array([2, 2, 2, 2]), #interesting [1,1,0,1] slope best, [1,1,1,1] flasso best, [0,1,0,1] lasso best
#                  C=np.array([[1,rho,rho,rho],[rho,1,rho, rho],[rho,rho,1,rho],[rho,rho,rho,1]]), #(1-rho) * np.identity(4) + rho * np.ones((4, 4)),
#                  lambdas=np.array([1.6, 1.2, 0.8, 0.4]),
#                  x=np.linspace(0,1,20), #np.linspace(0.48, 0.55, 10)
#                  n=100,
#                  Cov=0.4**2*np.array([[1,rho,rho,rho],[rho,1,rho, rho],[rho,rho,1,rho],[rho,rho,rho,1]]), #(1-rho) * np.identity(4) + rho * np.ones((4, 4)),
#                  flasso=True,
#                  A_flasso=Acustom(a=np.ones(4), b=0.5 * np.ones(3)),
#                  #glasso=True,
#                  #A_glasso=Acustom(a=np.ones(4), b=0.6 * np.array([0.9, 1.2, 0.9])),
#                  smooth=True)

rho = 0.3
plot_performance(b_0=np.array([1, 1, 2, 1, 1, 1, 2, 1, 1]), #interesting [1,1,0,1] slope best, [1,1,1,1] flasso best, [0,1,0,1] lasso best
                 C=(1-rho) * np.identity(9) + rho * np.ones((9, 9)),
                 lambdas=np.array([1.4, 1.3, 1.2, 1.1, 1, 0.9, 0.8, 0.7, 0.6]),
                 x=np.linspace(0,1,20), #np.linspace(0.48, 0.55, 10)
                 n=300,
                 Cov=0.4**2*(1-rho) * np.identity(9) + rho * np.ones((9, 9)), #(1-rho) * np.identity(4) + rho * np.ones((4, 4)),
                 flasso=True,
                 A_flasso=Acustom(a=np.ones(9), b=0.5 * np.ones(8)),
                 #glasso=True,
                 #A_glasso=Acustom(a=np.ones(4), b=0.6 * np.array([0.9, 1.2, 0.9])),
                 smooth=True)



#phase transition in pattern recovery for SLOPE as correlation increases
def plot_performance_tripple(b_0, C1, C2, C3, lambdas, x, n, Cov1=None, Cov2=None, Cov3=None):
    PattSLOPE1 = np.empty(shape=(0,))
    PattSLOPE2 = np.empty(shape=(0,))
    PattSLOPE3 = np.empty(shape=(0,))

    for i in range(len(x)):
        resultSLOPE1 = patternMSE(b_0=b_0, C=C1, Cov=Cov1, lambdas=x[i] * lambdas, n=n)
        resultSLOPE2 = patternMSE(b_0=b_0, C=C2, Cov=Cov2, lambdas=x[i] * lambdas, n=n)
        resultSLOPE3 = patternMSE(b_0=b_0, C=C3, Cov=Cov3, lambdas=x[i] * lambdas, n=n)

        PattSLOPE1 = np.append(PattSLOPE1, resultSLOPE1[1])
        PattSLOPE2 = np.append(PattSLOPE2, resultSLOPE2[1])
        PattSLOPE3 = np.append(PattSLOPE3, resultSLOPE3[1])

    print(PattSLOPE1)
    print(PattSLOPE2)
    print(PattSLOPE3)
    # Spline interpolation for smoother curve
    x_smooth = np.concatenate((x, np.linspace(x.min(), x.max(), 100))) #np.array([0, 0.05, 0.1, 0.14, 0.18, 0.5, 0.8, 1.2, 2])#np.linspace(x.min(), x.max(), 20)  # Generate more points for smoothness
    x_smooth = np.sort(x_smooth)

    spl1 = make_interp_spline(x, PattSLOPE1)
    spl1 = PchipInterpolator(x, PattSLOPE1)  #
    spl2 = make_interp_spline(x, PattSLOPE2)
    spl2 = PchipInterpolator(x, PattSLOPE2)  #
    spl3 = make_interp_spline(x, PattSLOPE3)
    spl3 = PchipInterpolator(x, PattSLOPE3)  #

    PattSLOPE1_smooth = spl1(x_smooth)
    PattSLOPE2_smooth = spl2(x_smooth)
    PattSLOPE3_smooth = spl3(x_smooth)

    plt.figure(figsize=(6, 6))
    #plt.plot(x, MseSLOPE, label='RMSE SLOPE', color='green', lw=1.5, alpha=0.9)  # Plot RMSE of SLOPE
    #plt.plot(x, MseLasso, label='RMSE Lasso', color='blue', lw=1.5, alpha=0.9)  # Plot RMSE of Lasso


    #plt.plot(x, PattSLOPE1, label=r'$\rho = 2/3-0.05$', color='green', linestyle='dashed', lw=1.5)  # Plot probability of pattern recovery by SLOPE
    plt.plot(x_smooth, PattSLOPE1_smooth, label=r'$\rho = 2/3-0.05$', color='green', linestyle='dashed', lw=1.5)
    #plt.plot(x, PattSLOPE2, label=r'$\rho = 2/3$', color='blue', linestyle='dashed', lw=1.5)
    plt.plot(x_smooth, PattSLOPE2_smooth, label=r'$\rho = 2/3$', color='blue', linestyle='dashed', lw=1.5)
    #plt.plot(x, PattSLOPE3, label=r'$\rho = 2/3+0.05$', color='red', linestyle='dashed', lw=1.5)
    plt.plot(x_smooth, PattSLOPE3_smooth, label=r'$\rho = 2/3+0.05$', color='red', linestyle='dashed', lw=1.5)

    #plt.plot(x, PattLasso, label='pattern recovery Lasso', color='blue', linestyle='dashed', lw=1.5)  # Plot prob of pattern by Lasso
    #plt.plot(x, SupportSLOPE, label='support recovery SLOPE', color='green', linestyle='-.', lw=1.5, alpha=0.5)  # Plot prob of support recovery by SLOPE
    #plt.plot(x, SupportLasso, label='support recovery Lasso', color='blue', linestyle='-.', lw=1.5, alpha=0.5)  # Plot prob of support recovery by Lasso

    #plt.scatter(0, resultOLS, color='red', label='RMSE OLS')

    # Increase the size of x-axis and y-axis tick labels
    plt.xticks(fontsize=14)  # Change 12 to the desired font size for x-axis tick labels
    plt.yticks(fontsize=14)  # Change 12 to the desired font size for y-axis tick labels

    plt.xlabel(r'$\alpha$', fontsize=16) #penalty scaling
    plt.ylabel('pattern recovery', fontsize=16)
    #plt.title('Pattern Recovery and RMSE')
    caption_text = f'$b^0$ = {b_0}, $\lambda = \sigma$ {lambdas}' #compound or block diagonal C block diagonal with one compound 0.8 block for each cluster, and penalty scaling
    #plt.figtext(0.5, 0.01, caption_text, wrap=True, horizontalalignment='center', fontsize=10, color='black')
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True, ncol=3)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


alpha1 = 2/3-0.05
alpha2 = 2/3
alpha3 = 2/3+0.05
C1 = np.array([[1, alpha1], [alpha1, 1]])
C2 = np.array([[1, alpha2], [alpha2, 1]])
C3 = np.array([[1, alpha3], [alpha3, 1]])
sigma=0.2
x = np.linspace(0, 2, 11) #31 point between 0 and 10 was default
#print('x:', x)
# Custom points to be added
custom_points = np.array([(x[0]+x[1])/2, (x[1]+x[2])/2, (x[2]+x[3])/2])
# Concatenate the custom points with the linspace array
x_with_custom_points = np.concatenate((x, custom_points))
# Sort the array for better visualization (optional)
x = np.sort(x_with_custom_points)
#print('x:', x)
#x = np.linspace(0, 10, 31)
custom_points = np.array([0, 0.05, 0.18, 0.5, 1.2, 2])

custom_smooth = np.concatenate((custom_points, np.array([0.1, 0.14, 0.8])))
custom_smooth = np.sort(custom_smooth)
#print('concatenated', custom_smooth)

#plot_performance_tripple(b_0=np.array([1, 0]), C1=C1, C2=C2, C3=C3, lambdas=np.array([3, 2]), x = custom_points, n=100, Cov1=sigma ** 2 * C1, Cov2=sigma ** 2 * C2, Cov3=sigma ** 2 * C3) #, Cov1=sigma**2*C1, Cov2=sigma**2*C2, Cov3=sigma**2*C3)

'''
test_mean = 0
for i in range(20):
    a=patternMSE(b_0=np.array([1, 0]), C=C1, lambdas=np.array([3, 2]), n=1000, Cov=sigma**2*C1)
    print('recovery prob', a[2])
    test_mean = test_mean + a[2]
    print('test_mean', test_mean/(i+1))
print('test_mean', test_mean/20)
'''



'''
# Define the range of x values
x = np.linspace(0, 1, 20)  # Generates 10 points between 0 and 5

# Calculate y values for each function
alpha = 2/3
PattSLOPE = np.empty(shape=(0, ))
MseSLOPE = np.empty(shape=(0, ))
PattLasso = np.empty(shape=(0, ))
MseLasso = np.empty(shape=(0, ))
for i in range(len(x)):
    result = patternMSE(b_0 = np.array([1, 0]), C = np.array([[1, alpha], [alpha, 1]]), lambdas = x[i]*np.array([0.8, 1.2]), n = 50)
    resultLasso = patternMSE(b_0 = np.array([1, 0]), C = np.array([[1, alpha], [alpha, 1]]), lambdas = x[i]*np.array([1, 1]), n = 50)
    PattSLOPE = np.append(PattSLOPE, result[0])
    MseSLOPE = np.append(MseSLOPE, result[1])
    PattLasso = np.append(PattLasso, resultLasso[0])
    MseLasso = np.append(MseLasso, resultLasso[1])
#print(PattLasso)
#print(MseLasso)

# Plot the functions on the same graph
plt.figure(figsize=(3, 6))
plt.plot(x, PattSLOPE, label='SLOPE pattern recovery', color='blue')  # Plot probability of pattern recovery by SLOPE in blue
plt.plot(x, MseSLOPE, label='SLOPE MSE', color='green')  # Plot MSE of SLOPE in green
plt.plot(x, PattLasso, label='Lasso pattern recovery', color='red')
plt.plot(x, MseLasso, label='Lasso MSE', color='orange') # Plot MSE of Lasso in red
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


'''
#first simulation attempts for MSE and Pattern recovery
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
