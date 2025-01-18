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



def patternMSE(b_0, C, lambdas, n, Cov=None, genlasso=False, A = None, tol=1e-4):
    """
    Calculate mean squared error (MSE), probability of pattern and support recovery, and dimension reduction
    for a given pattern vector 'b_0', covariance C, and penalty sequence lambdas, for SLOPE, or possibly Generalized/Fused Lasso.

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

    for i in range(n):
        # Generate random data with given covariance matrix
        W = np.random.multivariate_normal(np.zeros(p), Cov)

        # Perform SLOPE optimization using PGD and FISTA algorithm
        if genlasso == True:
            if A is None:
                A = Acustom(a=np.ones(len(b_0)), b=np.ones(len(b_0)-1))
            u_hat = admm_glasso(C=C, A=A, w=W, beta0=b_0, lambdas=1.0)
        else:
            u_hat = pgd_slope_b_0_FISTA(C=C, W=W, b_0=b_0, lambdas=lambdas, tol=tol ) # 20 iterations for FISTA, might need more in some cases (like in graphical examples we need around n=200)

            #u_hatminus = pgd_slope_b_0_FISTA(C=C, W=W, b_0=b_0, lambdas=lambdas, t=stepsize_t, n=20)

            #print('u_hat:', u_hat, pattern(u_hat))
            #print('u_hatminus:', u_hatminus, pattern(u_hatminus))
            #print('pat_error:', pattern(u_hat)-pattern(u_hatminus))
            #print('hat_error:', np.round(u_hat-u_hatminus,4))

        # Calculate MSE
        norm2 = np.linalg.norm(u_hat) ** 2
        MSE += norm2

        # Calculate dimension reduction
        dim_reduction += p - len(np.unique(u_hat))

        # Check pattern and support recoveries
        #if genlasso == True:
            #print('u_hat', u_hat)
            #print('pattern(u_hat)', pattern(u_hat))
            #print('b_0+eps_pat(u_hat)', pattern(b_0 + 0.01 * pattern(u_hat)))
            #print('b_0+eps_pat(np.round(u_hat,5))', pattern(b_0 + 0.01 * pattern(np.round(u_hat,5))))
        if all(pattern(b_0 + (1/(2*p+1)) * pattern(np.round(u_hat,5))) == b_0): # rounding to prevent numerical errors of order 1e-16
            correct_pattern_recovery += 1
        if all(np.sign(pattern(b_0 + (1/(2*p+1)) * pattern(np.round(u_hat,5)))) == np.sign(b_0)):
            correct_support_recovery += 1

    # Calculate average metrics over simulations
    rmse = np.sqrt(MSE / n)
    pattern_recovery_rate = correct_pattern_recovery / n
    support_recovery_rate = correct_support_recovery / n
    avg_dim_reduction = dim_reduction / n

    return rmse, pattern_recovery_rate, support_recovery_rate, avg_dim_reduction


def reducedOLSerror(b_0, C, n=10000, sigma = 1):
    U_b0_SLOPE = pattern_matrix(b_0)
    p_red_SLOPE = np.shape(U_b0_SLOPE)[1]
    redC_SLOPE = U_b0_SLOPE.T @ C @ U_b0_SLOPE  # reduced covariance matrix
    redCinv_SLOPE = np.linalg.inv(redC_SLOPE)  # reduced error u_red has normal distribution with covariance sigma^2 * redCinv_SLOPE

    U_b0_Lasso = pattern_matrix_Lasso(b_0)
    p_red_Lasso = np.shape(U_b0_Lasso)[1]
    redC_Lasso = U_b0_Lasso.T @ C @ U_b0_Lasso  # reduced covariance matrix
    redCinv_Lasso = np.linalg.inv(redC_Lasso)  # reduced error u_red has normal distribution with covariance sigma^2 * redCinv_Lasso

    U_b0_Flasso = pattern_matrix_FLasso(b_0)
    p_red_Flasso = np.shape(U_b0_Flasso)[1]
    redC_Flasso = U_b0_Flasso.T @ C @ U_b0_Flasso  # reduced covariance matrix
    redCinv_Flasso = np.linalg.inv(redC_Flasso)  # reduced error u_red has normal distribution with covariance sigma^2 * redCinv_Flasso

    MSE_OLS = 0
    redMSE_SLOPE = 0
    redMSE_Lasso = 0
    redMSE_Flasso = 0
    for i in range(n):
        u_OLS = np.random.multivariate_normal(np.zeros(len(b_0)), sigma ** 2 * np.linalg.inv(C))
        norm2_OLS = np.linalg.norm(u_OLS) ** 2
        MSE_OLS = MSE_OLS + norm2_OLS

        u_red_SLOPE = np.random.multivariate_normal(np.zeros(p_red_SLOPE), sigma**2 * redCinv_SLOPE)
        norm2_SLOPE = np.linalg.norm(u_red_SLOPE) ** 2
        redMSE_SLOPE = redMSE_SLOPE + norm2_SLOPE

        u_red_Lasso = np.random.multivariate_normal(np.zeros(p_red_Lasso), sigma**2 * redCinv_Lasso)
        norm2_Lasso = np.linalg.norm(u_red_Lasso) ** 2
        redMSE_Lasso = redMSE_Lasso + norm2_Lasso

        u_red_Flasso = np.random.multivariate_normal(np.zeros(p_red_Flasso), sigma**2 * redCinv_Flasso)
        norm2_Flasso = np.linalg.norm(u_red_Flasso) ** 2
        redMSE_Flasso = redMSE_Flasso + norm2_Flasso

    return np.sqrt(MSE_OLS / n), np.sqrt(redMSE_Lasso / n), np.sqrt(redMSE_Flasso / n), np.sqrt(redMSE_SLOPE / n)
#print('reducedOLSerror:', reducedOLSerror(b_0=np.array([1,0,1,1,0]), C=np.identity(5)))


def plot_performance(b_0, C, lambdas, x, n, Cov=None, Lasso=True, SLOPE=True, flasso=False, A_flasso=None, glasso=False, A_glasso=None, reducedOLS=None, sigma=None, smooth=None, legend = True, tol=1e-3):
    PattSLOPE = np.empty(shape=(0,))
    MseSLOPE = np.empty(shape=(0,))

    PattLasso = np.empty(shape=(0,))
    MseLasso = np.empty(shape=(0,))

    Pattglasso = np.empty(shape=(0,))
    Mseglasso = np.empty(shape=(0,))

    Pattflasso = np.empty(shape=(0,))
    Mseflasso = np.empty(shape=(0,))
    #resultOLS = 0
    counter = 0

    #resultOLS = 0
    #for i in range(n):
    #    C_inv = np.linalg.inv(C)
    #    u_OLS = np.random.multivariate_normal(np.zeros(len(b_0)), C_inv @ Cov @ C_inv)
    #    norm2OLS = np.linalg.norm(u_OLS) ** 2
    #    resultOLS = resultOLS + norm2OLS
    #resultOLS= np.sqrt(resultOLS / n)

    resultOLS=0


    if SLOPE:
        for i in range(len(x)):
            resultSLOPE = patternMSE(b_0=b_0, C=C, Cov=Cov, lambdas=x[i] * lambdas, n=n, tol=tol)
            MseSLOPE = np.append(MseSLOPE, resultSLOPE[0])
            PattSLOPE = np.append(PattSLOPE, resultSLOPE[1])
        resultOLS = resultOLS + MseSLOPE[0]
        counter = counter + 1

    if Lasso:
        for i in range(len(x)):
            resultLasso = patternMSE(b_0=b_0, C=C, Cov=Cov, lambdas=x[i] * lambdas, n=n, genlasso=True, A=x[i]*Acustom(a=np.ones(len(b_0)), b=np.zeros(len(b_0)-1)), tol=tol) #lambdas are irrelevant here, since genlasso=True overrides them with A, yes the code is messy here
            MseLasso = np.append(MseLasso, resultLasso[0])
            PattLasso = np.append(PattLasso, resultLasso[1])
        resultOLS = resultOLS + MseLasso[0]
        counter = counter + 1

    if flasso:
        for i in range(len(x)):
            if A_flasso is None:
                A_flasso = Acustom(a=np.ones(len(b_0)), b=np.ones(len(b_0) - 1))
            resultflasso = patternMSE(b_0=b_0, C=C, Cov=Cov, lambdas=x[i] * lambdas, n=n, genlasso=True, A=x[i]*A_flasso)
            Mseflasso= np.append(Mseflasso, resultflasso[0])
            Pattflasso = np.append(Pattflasso, resultflasso[1])
        resultOLS = resultOLS + Mseflasso[0]
        counter = counter + 1

    if glasso:
        for i in range(len(x)):
            if A_glasso is None:
                A_glasso = Acustom(a=np.ones(len(b_0)), b=np.ones(len(b_0) - 1))
            resultglasso = patternMSE(b_0=b_0, C=C, Cov=Cov, lambdas=x[i] * lambdas, n=n, genlasso=True,
                                      A=x[i] * A_glasso)
            Mseglasso= np.append(Mseglasso, resultglasso[0])
            Pattglasso = np.append(Pattglasso, resultglasso[1])
        resultOLS = resultOLS + Mseglasso[0]
        counter = counter + 1

    resultOLS= resultOLS / counter


    if smooth:
        x_smooth = np.linspace(x.min(), x.max(), 20 * (len(x) - 1) + 1)
        if SLOPE:
            spl1 = PchipInterpolator(x, MseSLOPE)
            spl2 = PchipInterpolator(x, PattSLOPE)
            MseSLOPE = spl1(x_smooth)
            PattSLOPE = spl2(x_smooth)

        if Lasso:
            spl3 = PchipInterpolator(x, MseLasso)
            spl4 = PchipInterpolator(x, PattLasso)
            MseLasso = spl3(x_smooth)
            PattLasso = spl4(x_smooth)

        if flasso:
            spl5 = PchipInterpolator(x, Mseflasso)
            spl6 = PchipInterpolator(x, Pattflasso)
            Mseflasso = spl5(x_smooth)
            Pattflasso = spl6(x_smooth)

        if glasso:
            spl7 = PchipInterpolator(x, Mseglasso)
            spl8 = PchipInterpolator(x, Pattglasso)
            Mseglasso = spl7(x_smooth)
            Pattglasso = spl8(x_smooth)

        x = x_smooth

    plt.figure(figsize=(6, 6))
    if Lasso:
        plt.plot(x, MseLasso, label='RMSE Lasso', color='blue', lw=1.5, alpha=0.9)
        plt.plot(x, PattLasso, label='recovery Lasso', color='blue', linestyle='dashed', lw=1.5)
    if SLOPE:
        plt.plot(x, MseSLOPE, label='RMSE SLOPE', color='green', lw=1.5, alpha=0.9)
        plt.plot(x, PattSLOPE, label='recovery SLOPE', color='green', linestyle='dashed', lw=1.5)
    if flasso:
        plt.plot(x, Mseflasso, label='RMSE FLasso', color='orange', lw=1.5, alpha=0.9) #comment out to supress RMSE FLasso
        plt.plot(x, Pattflasso+0.02, label='recovery FLasso', color='orange', linestyle='dashed', lw=1.5, alpha = 0.8)
    if glasso:
        plt.plot(x, Mseglasso, label='RMSE ConFLasso', color='purple', lw=1.5, alpha=0.9)
        plt.plot(x, Pattglasso, label='recovery ConFLasso', color='purple', linestyle='dashed', lw=1.5, alpha = 0.8)
    if reducedOLS:
        reducedOLS = reducedOLSerror(b_0, C, n=100000, sigma=sigma)
        plt.scatter(-0.025, reducedOLS[1], color='blue', alpha=0.7, s=70)
        plt.scatter(0, reducedOLS[2], color='orange', alpha=0.7, s=70)
        plt.scatter(0, reducedOLS[3], color='green', alpha=0.7, s=70)
    plt.scatter(0, resultOLS, color='red', label='RMSE OLS', alpha=0.9, s=70) #comment out to supress reducedOLS

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(r'$\alpha$', fontsize=16)

    if legend:
        plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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

    plt.xlabel(r'$\sigma$', fontsize=16) #penalty scaling
    plt.ylabel('pattern recovery', fontsize=16)
    #plt.title('Pattern Recovery and RMSE')
    caption_text = r'$b^0$ = {b_0}, $\lambda = \sigma$ {lambdas}' #compound or block diagonal C block diagonal with one compound 0.8 block for each cluster, and penalty scaling
    #plt.figtext(0.5, 0.01, caption_text, wrap=True, horizontalalignment='center', fontsize=10, color='black')
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True, ncol=3)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()



