from src.slope.solvers import*
from src.slope.admm_glasso import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator




def mat_to_vec(arr):
    p = arr.shape[0]  # assuming the input array is square (p x p)
    flattened_arr = arr.flatten(order='F')  # column-wise flattening (Fortran order)
    stacked_arr = flattened_arr.reshape((p ** 2, 1))  # reshaping to p^2 x 1
    return stacked_arr
def mat_to_vech(arr, strict_low=False):  # input: p x p matrix output: low-diag p(p+1)/2 x 1 vector
    p = arr.shape[0]
    stack_low = np.zeros(int(p*(p+1)/2))
    counter = 0
    for j in range(p):
        for i in range(j, p):
            stack_low[counter] = arr[i][j]
            counter = counter + 1
    return stack_low
def vech_to_mat(vech):  # input: p(p+1)/2 x 1 vector output: p x p matrix
    p = int(-0.5 + np.sqrt(0.25 + 2*len(vech)))
    mat = np.zeros((p, p))
    counter = 0
    for j in range(p):
        for i in range(j, p):
            mat[i][j] = vech[counter]
            mat[j][i] = vech[counter]
            counter = counter + 1
    return mat
#print('mat_to_vec:\n', mat_to_vec(np.array([[2, 0], [0, 2]])))
#print('vech_to_mat:\n',vech_to_mat(np.array([1, 0.3, 0.4, 1, 0, 2])))


def D(p): # output: p^2 x p(p+1)/2 duplication matrix; i.e. D(p) @ vec(X) = vech(X)
    enumerator_full = []
    enumerator_reduced = []
    for i in range(p):
        for j in range(p):
            enumerator_full.append((j + 1, i + 1))
            if j >= i:
                enumerator_reduced.append((j + 1, i + 1))

    l1 = len(enumerator_full)
    l2 = len(enumerator_reduced)
    D_p = np.zeros((l1, l2))

    for i in range(l1):
        for j in range(l2):
            if enumerator_full[i] == enumerator_reduced[j] or tuple(reversed(enumerator_full[i])) == enumerator_reduced[j]:
                D_p[i][j] = 1
    return D_p
#print('D:\n', D(3))
#print('D^TD:\n', D(3).T @ D(3))

def Hessian(Sigma):
    p = len(Sigma[0])
    return D(p).T @ np.kron(Sigma, Sigma) @ D(p) * 0.5

#Sigma3 = np.array([[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]])
#print('Sigma3:\n', Sigma3)
#print('C_tilde:\n', Hessian(Sigma3))


'''
lambdas_low = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
truth = vech_to_mat(pgd_slope_b_0_FISTA(C=C_tilde, W=np.array([1, 0.5, 0.3, 1, 0.85, -0.6, 1.1, 0.2, -0.4, 0.1]),
                                        b_0=pat_Theta4c, lambdas=0.1 * lambdas_low, t=0.2, n=10000))
for n in range(150,151):
    print('gslp_Fista:\n', vech_to_mat(pgd_slope_b_0_FISTA(C=C_tilde, W=np.array([1, 0.5, 0.3, 1, 0.85, -0.6, 1.1, 0.2, -0.4, 0.1]),
                                                           b_0=pat_Theta4c, lambdas=0.1 * lambdas_low, t=0.2, n=n)))
    #print('diff_Fista:', n, '\n', np.round(vech_to_mat(pgd_slope_b_0_FISTA(C=C_tilde, W=np.array([1, 0.5, 0.3, 1, 0.85, -0.6, 1.1, 0.2, -0.4, 0.1]),
    #                                   b_0=pat_Theta4c, lambdas=0.1 * lambdas, t=0.2, n=n))-vech_to_mat(pgd_slope_b_0_FISTA(C=C_tilde, W=np.array([1, 0.5, 0.3, 1, 0.85, -0.6, 1.1, 0.2, -0.4, 0.1]),
    #                                   b_0=pat_Theta4c, lambdas=0.1 * lambdas, t=0.2, n=n-1)),5))
    print('error_Fista:\n',
          np.linalg.norm(truth - vech_to_mat(pgd_slope_b_0_FISTA(C=C_tilde, W=np.array([1, 0.5, 0.3, 1, 0.85, -0.6, 1.1, 0.2, -0.4, 0.1]),
                                                                 b_0=pat_Theta4c, lambdas=0.1 * lambdas_low, t=0.2, n=n))))
    print('gslp_Ista:\n',
          vech_to_mat(pgd_slope_b_0_ISTA(C=C_tilde, W=np.array([1, 0.5, 0.3, 1, 0.85, -0.6, 1.1, 0.2, -0.4, 0.1]),
                                         b_0=pat_Theta4c, lambdas=0.1 * lambdas_low, t=0.2, n=n)))
    #print('diff_Ista:', n, '\n', np.round(
    #    vech_to_mat(pgd_slope_b_0_ISTA(C=C_tilde, W=np.array([1, 0.5, 0.3, 1, 0.85, -0.6, 1.1, 0.2, -0.4, 0.1]),
    #                                    b_0=pat_Theta4c, lambdas=0.1 * lambdas, t=0.2, n=n)) - vech_to_mat(
    #        pgd_slope_b_0_ISTA(C=C_tilde, W=np.array([1, 0.5, 0.3, 1, 0.85, -0.6, 1.1, 0.2, -0.4, 0.1]),
    #                            b_0=pat_Theta4c, lambdas=0.1 * lambdas, t=0.2, n=n - 1)), 5))
    print('error_Ista:\n',
          np.linalg.norm(truth - vech_to_mat(
              pgd_slope_b_0_ISTA(C=C_tilde, W=np.array([1, 0.5, 0.3, 1, 0.85, -0.6, 1.1, 0.2, -0.4, 0.1]),
                                 b_0=pat_Theta4c, lambdas=0.1 * lambdas_low, t=0.2, n=n))))
'''

#print('p:', int(-0.5 + np.sqrt(0.25 + 2 * len(mat_to_vech(Theta4c)))))
def split_diag_and_low(vechTheta): # returns diagonal and lower triangular part of a symmetric matrix ( as two np.vectors )
    #  input can be a matrix or vectorized lower triangular part of that matrix
    if len(vechTheta.shape) == 2:
        vechTheta = mat_to_vech(vechTheta)
    p = int(-0.5 + np.sqrt(0.25 + 2 * len(vechTheta)))
    all_indices = list(range(len(vechTheta)))
    diagonal_indices = []
    for i in range(1,p+1):
        diagonal_indices.append(len(vechTheta) - int(i * (i + 1) / 2))
        diagonal_indices.sort()
    #return diagonal_indices
    diag_Theta = vechTheta[diagonal_indices]
    low_indices = [i not in diagonal_indices for i in all_indices]
    low_Theta = vechTheta[low_indices]
    return diag_Theta, low_Theta, type(diag_Theta), type(vechTheta)
#print('split_diag_and_low:\n',Theta4c,"\n", split_diag_and_low(Theta4c))

def join_diag_and_low(diag_Theta, low_Theta): # returns a matrix (as a vech vector) from its diagonal and lower triangular part # inverse of split_diag_and_low
    p = len(diag_Theta)
    diagonal_indices = []
    Theta = np.zeros(int(p*(p+1)/2))
    for i in range(1, p+1):
        diagonal_indices.append(p*(p+1)/2 - int(i * (i + 1) / 2))
        diagonal_indices.sort()
    diag_counter = 0
    low_counter = 0
    for i in range(len(Theta)):
        if i in diagonal_indices:
            Theta[i] = diag_Theta[diag_counter].real
            diag_counter += 1
        else:
            Theta[i] = low_Theta[low_counter].real
            low_counter += 1
    return Theta
#print('join_diag_and_low:\n', join_diag_and_low(split_diag_and_low(Theta4c)[0], split_diag_and_low(Theta4c)[1]))


def pgd_gslope_Theta0_ISTA(C, W, Theta0, lambdas, t, n):
    """Minimizes: 1/2 u^T*C*u-u^T*W+J'_{lambda}(vechTheta0; u),
     where J'_{lambda}(b^0; u) is the directional SLOPE derivative
       Parameters
       ----------
       C: np.array
           covariance matrix of the data
       W: np.array
           p-dimensional vector, in our paper it arises from normal N(0, \sigma^2 * C ),
           where \sigma^2 is variance of the noise
       Theta0: np.array
           true precision matrix
       lambdas : np.array
           vector of regularization weights
       t: np.float
           step size
       n: integer
           number of steps before termination

       Returns
       -------
       array
           the unique solution to the minimization problem, given by a vector u.
       """
    #W = [np.float(i) for i in W]
    #lambdas = [np.float(i) for i in lambdas]
    vechTheta0 = pattern(np.round(mat_to_vech(Theta0),3))
    u_0 = np.zeros(len(vechTheta0))
    prox_step = u_0
    stepsize_t = np.float64(t)
    for i in range(n):
        grad_step = prox_step - stepsize_t * (C @ prox_step - W)
        grad_step_diag = split_diag_and_low(grad_step)[0]
        grad_step_low = split_diag_and_low(grad_step)[1]

        prox_step_low = prox_slope_b_0(split_diag_and_low(vechTheta0)[1], grad_step_low, lambdas * stepsize_t) #prox step only on the lower diagonal entries
        prox_step_diag = grad_step_diag
        prox_step = join_diag_and_low(prox_step_diag, prox_step_low)
    return(prox_step)

def pgd_gslope_Theta0_FISTA(C, W, Theta0, lambdas, n, t=None):
    """Minimizes: 1/2 u^T*C*u-u^T*W+J'_{lambda}(vechTheta0; u),
     where J'_{lambda}(b^0; u) is the directional SLOPE derivative
       Parameters
       ----------
       C: np.array
           covariance matrix of the data
       W: np.array
           p-dimensional vector, in our paper it arises from normal N(0, \sigma^2 * C ),
           where \sigma^2 is variance of the noise
       Theta0: np.array
           pattern vector of the true signal
       lambdas : np.array
           vector of regularization weights
       t: np.float
           step size
       n: integer
           number of steps before termination

       Returns
       -------
       array
           the unique solution to the minimization problem, given by a vector u.
       """
    vechTheta0 = pattern(np.round(mat_to_vech(Theta0),3))
    u_k = np.zeros(len(vechTheta0))  # initial point
    u_kmin2 = u_k  # lagged iterate u_{k-2}
    u_kmin1 = u_k  # lagged iterate u_{k-1}
    v = u_k
    if t==None:
        t = 1/np.max(np.linalg.eigvals(C))  # default stepsize = 1/max(eigenvalues of C) to guarantee O(1/n^2) convergence
        #t = np.float32(t) #np.real(t)
        t=np.float32(np.real(t))
    stepsize_t = t #np.float32(t)
    k=1
    #u_k=np.zeros(len(vechTheta0))
    for k in range(n):
        v = u_kmin1 + ((k-2)/(k+1))*(u_kmin1-u_kmin2)
        grad_step = v - stepsize_t * (C @ v - W)
        grad_step_diag = split_diag_and_low(grad_step)[0]
        grad_step_low = split_diag_and_low(grad_step)[1]

        prox_step_low = prox_slope_b_0(split_diag_and_low(vechTheta0)[1], grad_step_low, lambdas * stepsize_t) #prox step only on the lower diagonal entries
        prox_step_diag = grad_step_diag
        u_k = join_diag_and_low(prox_step_diag, prox_step_low)
        u_kmin2 = u_kmin1
        u_kmin1 = u_k
    return u_k




lambdas_low = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]) / 3.5
print('evals:\n', np.linalg.eigvals(C_tilde))
print('1/max(eval):\n', 1/max(np.linalg.eigvals(C_tilde)))

#convergence speed issues, number of iterates required is in the order of hundreds
'''
W=np.array([1, 0.5, 0.3, 1, 0.85, -0.6, 1.1, 0.2, -0.4, 0.1])
est_truth_gslp= vech_to_mat(pgd_slope_b_0_FISTA(C=C_tilde, W=W,
                                       b_0=signal, lambdas=8*0.1 * np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])/55, t=0.2, n=1000))
est_truth_low= vech_to_mat(pgd_gslope_Theta0_ISTA(C=C_tilde, W=W,
                                                  Theta0=Theta4c, lambdas=8 * 0.1 * lambdas_low, t=0.2, n=1000))

print('est_truth_gslp:\n', est_truth_gslp, '\n', 'est_truth_low:\n', est_truth_low)
print('err_est_truth_gslp:\n', est_truth_gslp - Theta4c, '\n', 'err_est_truth_low:\n', est_truth_low - Theta4c)
for i in range(500, 501):
    Fista = vech_to_mat(pgd_gslope_Theta0_FISTA(C=C_tilde, W=W,
                                                Theta0=Theta4c, lambdas=8 * 0.1 * lambdas_low, t=0.2, n=i))
    Fistadefault = vech_to_mat(pgd_gslope_Theta0_FISTA(C=C_tilde, W=W,
                                                       Theta0=Theta4c, lambdas=8 * 0.1 * lambdas_low, n=i))

    print('Fista:\n', np.linalg.norm(Fista - est_truth_low), '\n', 'Fistadefault:\n', np.linalg.norm(Fistadefault - est_truth_low))
'''

'''
for n in range(200,201):
    print('gslp_onlylow_pen:\n', vech_to_mat(pgd_gslope_Theta0_ISTA(C=C_tilde, W=np.array([1, 0.5, 0.3, 1, 0.85, -0.6, 1.1, 0.2, -0.4, 0.1]),
                                       vechTheta0=signal, lambdas=0.1 * lambdas, t=0.2, n=n)))
    print('diffIsta:', n, '\n', np.round(truthIsta-vech_to_mat(pgd_gslope_Theta0_ISTA(C=C_tilde, W=np.array([1, 0.5, 0.3, 1, 0.85, -0.6, 1.1, 0.2, -0.4, 0.1]),
                                       vechTheta0=signal, lambdas=0.1 * lambdas, t=0.2, n=n)),9))
    print('gslp_onlylow_penFista:\n',
          vech_to_mat(pgd_gslope_Theta0_FISTA(C=C_tilde, W=np.array([1, 0.5, 0.3, 1, 0.85, -0.6, 1.1, 0.2, -0.4, 0.1]),
                                             vechTheta0=signal, lambdas=0.1 * lambdas, n=n)))
    print('gslp_onlylow_penFista_with_t:\n',
          vech_to_mat(pgd_gslope_Theta0_FISTA(C=C_tilde, W=np.array([1, 0.5, 0.3, 1, 0.85, -0.6, 1.1, 0.2, -0.4, 0.1]),
                                              vechTheta0=signal, lambdas=0.1 * lambdas, n=n, t=0.2)))
    print('diffFista_with_t:', n, '\n', np.round(truthFista - vech_to_mat(
        pgd_gslope_Theta0_FISTA(C=C_tilde, W=np.array([1, 0.5, 0.3, 1, 0.85, -0.6, 1.1, 0.2, -0.4, 0.1]),
                                vechTheta0=signal, lambdas=0.1 * lambdas, n=n, t=0.2)), 9))
    print('diffFista:', n, '\n', np.round(truthFista - vech_to_mat(
        pgd_gslope_Theta0_FISTA(C=C_tilde, W=np.array([1, 0.5, 0.3, 1, 0.85, -0.6, 1.1, 0.2, -0.4, 0.1]),
                               vechTheta0=signal, lambdas=0.1 * lambdas, n=n)), 9))
'''

def gpatternMSE(Theta0, lambdas_low, n, C=None, Cov=None, genlasso=False, A = None):
    """
    Calculate root mean squared error (RMSE), probability of pattern recovery, and PatternRMSE
    by sampling the optimization error, which minimizes u^T*C*u/2-u^T*W+J'_{lambdas_low}(vechTheta0; u), W~N(0,Cov)

    Parameters:
        Theta0 (array-like): true precision matrix
        C (array-like): Covariance matrix used in the optimization.
        lambdas_low (array-like): Regularization parameter for SLOPE optimization.
        n (int): Number of simulations to run for performance evaluation.
        Cov (array-like, optional): Covariance matrix used for data generation. Defaults to None,
                                    which means it's the same as 'C'.

    Returns:
        tuple: A tuple containing three performance metrics:
            - Root Mean Squared Error (RMSE) of the optimization result.
            - Proportion of correct pattern recoveries.
            - PatternRMSE error, proportion of the MSE that is due to the pattern error, i.e. distance from the pattern space of Theta0.
    """
    vechTheta0 = mat_to_vech(Theta0)
    Sigma0 = np.linalg.inv(Theta0)
    if C is None:
        C = Hessian(Sigma0) #0.5 D^T (Theta0^{-1} \otimes Theta0^{-1}) D given by Kronecker product
    if Cov is None:
        Cov = Hessian(Sigma0)

    p = len(vechTheta0)
    pat_signal = pattern(np.round(vechTheta0,3))
    pat_proj = proj_onto_pattern_space(pat_signal)
    correct_pattern_recovery = 0
    #correct_support_recovery = 0
    MSE = 0
    patMSE = 0
    stepsize_t = 1/max(np.linalg.eigvals(C))  # stepsize <= 1/max eigenvalue of C;
    # (max eval of C is the Lipschitz constant of grad(1/2 uCu - uW)=(Cu-W));
    # gives O(1/n^2) convergence;

    for i in range(n):
        # Generate random data with given covariance matrix
        W = np.random.multivariate_normal(np.zeros(p), Cov)

        # Perform SLOPE optimization using PGD and FISTA algorithm
        if genlasso == True:
            if A is None:
                A = Acustom(a=np.ones(len(vechTheta0)), b=np.ones(len(vechTheta0)-1))
            u_hat = admm_glasso(C=C, A=A, w=W, beta0=pat_signal, lambdas=1.0)
        else:
            u_hat = pgd_gslope_Theta0_FISTA(C=C, W=W, Theta0=Theta0, lambdas=lambdas_low, t=stepsize_t, n=400)
            #print('u_hat\n:', vech_to_mat(np.round(u_hat, 2))) # uncomment line for debugging, and understanding pattern error

        # Calculate MSE
        norm2 = np.linalg.norm(u_hat) ** 2
        MSE += norm2
        # Calculate pattern error: MSE = patMSE + restMSE
        pat_norm2 = np.linalg.norm((np.identity(len(vechTheta0))-pat_proj) @ u_hat) ** 2
        patMSE += pat_norm2  # patMSE is the Euclidean distance of u_hat from the pattern space of vechTheta0


        # Check pattern recovery on off-diagonal entries
        pat_signal_low = split_diag_and_low(pat_signal)[1]
        u_hat_low = split_diag_and_low(u_hat)[1]
        if all(pattern(pat_signal_low + 0.001 * np.round(u_hat_low, 2)) == pat_signal_low):
            correct_pattern_recovery += 1
        #if all(np.sign(pattern(b_0 + 0.0001 * np.round(u_hat, 3))) == np.sign(b_0)):
        #    correct_support_recovery += 1

    # Calculate average metrics over simulations
    rmse = np.sqrt(MSE / n)
    pattern_recovery_rate = correct_pattern_recovery / n
    pat_rmse = np.sqrt(patMSE / n)
    #support_recovery_rate = correct_support_recovery / n
    #avg_dim_reduction = dim_reduction / n

    return rmse, pattern_recovery_rate, pat_rmse

Sigma4_custom = np.array([[1, 0.8, 0.8, 0.8], [0.8, 1, 0.8, 0.8], [0.8, 0.8, 1, 0.8], [0.8, 0.8, 0.8, 1]])
Theta4_custom = np.linalg.inv(Sigma4_custom)
print('Theta4_custom:\n', Theta4_custom)
print('gpatternMSE:\n', gpatternMSE(Theta0=Theta4_custom, lambdas_low = lin_lambdas(6), n=10))
Theta4_band = np.array([[1, 0.9, 0.8, 0.7], [0.9, 1, 0.9, 0.8], [0.8, 0.9, 1, 0.9], [0.7, 0.8, 0.9, 1]])
print('gpatternMSE:\n', gpatternMSE(Theta0=Theta4_band, lambdas_low = 5*lin_lambdas(6), n=10))
print('lin_lambdas\n:',lin_lambdas(6))


Theta4_AR = np.array([[3, -1, 0, 0], [-1, 4, -1, 0], [0, -1, 4, -1], [0, 0, -1, 3]])
print('Theta4_AR:\n', Theta4_AR)
print('Sigma4_AR:\n', np.linalg.inv(Theta4_AR))

print('gpatternMSE:\n', gpatternMSE(Theta0=Theta4_AR, lambdas_low = 500*np.array([2, 1.8, 1.6, 1.4, 1.2, 1]), n=10)) # recovers pattern
#print('gpatternMSE:\n', gpatternMSE(Theta0=Theta4_AR, lambdas_low = 50*np.ones(6), n=10))
print('gpatternMSE:\n', gpatternMSE(Theta0=np.identity(4), lambdas_low = 50*np.ones(6), n=10))


#print('gpatternMSE:\n', gpatternMSE(Theta0=Theta4c, lambdas_low = 0* lambdas_low, n=100))
#print('gpatternMSE:\n', gpatternMSE(Theta0=Theta4c, lambdas_low = 3 * 0.1 * lambdas_low, n=100))
#print('gpatternMSE:\n', gpatternMSE(Theta0=Theta4c, lambdas_low = 3 * 0.1 * np.ones(6), n=100))


def plot_performance(Theta0, x, n, lambdas_low=None, C=None, Cov=None, patMSE=False, flasso=False, A_flasso = None, glasso=False, A_glasso = None, smooth = None):
    """
    Plot performance metrics for different regularization methods.

    Parameters:
    - Theta0 : numpy.ndarray
        The precision matrix. Should be the inverse of Sigma0.
    - x : numpy.ndarray
        Array of interpolation values for penalty scaling factors 'alpha'.
    - n : int
        Number of samples.
    - lambdas_low : numpy.ndarray, optional
        Array of penalty values for the lower triangular part of the matrix.
        Default is None, which generates a linear sequence of lambdas with average penalty equal to 1.
    - C : numpy.ndarray, optional
        Hessian matrix. Default is None, which calculates Hessian based on Theta0.
    - Cov : numpy.ndarray, optional
        Covariance matrix. Default is None, which calculates Cov based on Theta0.
    - patMSE : bool, optional
        If True, plot pattern-wise root mean squared error. Default is False.
    - flasso : bool, optional
        If True, include FLasso (Fused Lasso) in the plot. Default is False.
    - A_flasso : numpy.ndarray, optional
        Penalty matrix for FLasso. Default is None.
    - glasso : bool, optional
        If True, include ConFLasso (Concave Fused Lasso) in the plot. Default is False.
    - A_glasso : numpy.ndarray, optional
        Penalty matrix for ConFLasso. Default is None.
    - smooth : bool, optional
        If True, apply spline interpolation for smoother curves. Default is None.

    Returns:
    None

    Note:
    - The function plots various performance metrics such as RMSE (Root Mean Squared Error) and pattern recovery
      for different regularization methods including Lasso, SLOPE (Sorted L-One Penalized Estimation), FLasso, and ConFLasso.
    - The performance metrics are plotted against the penalty scaling factor 'alpha'.

    """
    #Theta0 = np.linalg.inv(Sigma0)
    Sigma0 = np.linalg.inv(Theta0)
    vechTheta0 = mat_to_vech(Theta0)
    vechTheta0_low = split_diag_and_low(vechTheta0)[1]
    if C is None:
        C = Hessian(Sigma0)  # 0.5 D^T (Theta0^{-1} \otimes Theta0^{-1}) D given by Kronecker product
    if Cov is None:
        Cov = Hessian(Sigma0)
    if lambdas_low is None: # default linear sequence of lambdas, average penalty is 1
        lambdas_low = lin_lambdas(len(vechTheta0_low))

    PattSLOPE = np.empty(shape=(0,))
    MseSLOPE = np.empty(shape=(0,))
    patMSESLOPE = np.empty(shape=(0,))

    PattLasso = np.empty(shape=(0,))
    MseLasso = np.empty(shape=(0,))
    patMSELasso = np.empty(shape=(0,))

    Pattglasso = np.empty(shape=(0,))
    Mseglasso = np.empty(shape=(0,))


    Pattflasso = np.empty(shape=(0,))
    Mseflasso = np.empty(shape=(0,))


    for i in range(len(x)):
        resultSLOPE = gpatternMSE(Theta0=Theta0, C=C, Cov=Cov, lambdas_low=x[i] * lambdas_low, n=n)
        resultLasso = gpatternMSE(Theta0=Theta0, C=C, Cov=Cov, lambdas_low=x[i] * np.ones(len(vechTheta0_low)), n=n)
        #resultLasso = gpatternMSE(Theta0=Theta0, C=C, Cov=Cov, lambdas=x[i] * lambdas, n=n, genlasso=True, A=x[i]*Acustom(a=np.ones(len(b_0)), b=np.zeros(len(b_0)-1)))
        # using admm for lasso instead of pgd_slope_b_0_FISTA

        MseSLOPE = np.append(MseSLOPE, resultSLOPE[0])
        PattSLOPE = np.append(PattSLOPE, resultSLOPE[1])
        MseLasso = np.append(MseLasso, resultLasso[0])
        PattLasso = np.append(PattLasso, resultLasso[1])

        if flasso == True:
            if A_flasso is None:
                A_flasso = Acustom(a=np.ones(len(vechTheta0_low)), b=np.ones(len(vechTheta0_low) - 1))
            resultflasso = gpatternMSE(Theta0=Theta0, C=C, Cov=Cov, lambdas_low=x[i] * lambdas_low, n=n, genlasso=True, A=x[i]*A_flasso)
            Mseflasso = np.append(Mseflasso, resultflasso[0])
            Pattflasso = np.append(Pattflasso, resultflasso[1])

        if glasso == True:
            if A_glasso is None:
                A_glasso = Acustom(a=np.ones(len(vechTheta0_low)), b=np.ones(len(vechTheta0_low) - 1))
            resultglasso = gpatternMSE(Theta0=Theta0, C=C, Cov=Cov, lambdas_low=x[i] * lambdas_low, n=n, genlasso=True, A=x[i] * A_glasso)
            Mseglasso= np.append(Mseglasso, resultglasso[0])
            Pattglasso = np.append(Pattglasso, resultglasso[1])
            #  Supportglasso = np.append(Supportglasso, resultglasso[2])

        resultOLS = 0.5*(MseSLOPE[0] + MseLasso[0])

        if patMSE == True:
            patMSESLOPE = np.append(patMSESLOPE, resultSLOPE[2])
        #SupportSLOPE = np.append(SupportSLOPE, resultSLOPE[2])
            patMSELasso = np.append(patMSELasso, resultLasso[2])
    if smooth == True:
        #  Spline interpolation for smoother curve
        x_smooth = np.linspace(x.min(), x.max(), 20 * (len(x) - 1) + 1)

        spl1 = PchipInterpolator(x, MseSLOPE)  #
        spl2 = PchipInterpolator(x, PattSLOPE)  #
        spl3 = PchipInterpolator(x, MseLasso)  #

        MseSLOPE = spl1(x_smooth)
        PattSLOPE = spl2(x_smooth)
        MseLasso = spl3(x_smooth)
        if patMSE == True:
            spl11 = PchipInterpolator(x, patMSESLOPE)
            spl12 = PchipInterpolator(x, patMSELasso)
            patMSESLOPE = spl11(x_smooth)
            patMSELasso = spl12(x_smooth)
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
    if patMSE == True:
        plt.plot(x, patMSESLOPE, label='patRMSE SLOPE', color='green', linestyle='dotted', lw=1.5)
        plt.plot(x, patMSELasso, label='patRMSE Lasso', color='blue', linestyle='dotted', lw=1.5)

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

    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


rho = 0.8
Sigma4c = np.array([[1, rho, rho, 0], [rho, 1, rho, 0], [rho, rho, 1, 0], [0, 0, 0, 1]])
C4c = Hessian(Sigma4c)
print('Theta4c:\n', np.linalg.inv(Sigma4c))

Sigma4 = (1-rho)*np.identity(4)+rho*np.ones((4,4))
Theta4 = np.linalg.inv(Sigma4)
print('Theta4:\n', np.linalg.inv(Sigma4))
Sigma9 = (1-rho)*np.identity(9)+rho*np.ones((9,9))
Theta9 = np.linalg.inv(Sigma9)
print('Sigma9:\n', Sigma9)
#print('Theta9:\n', np.linalg.inv(Sigma9))

Theta_test = np.linalg.inv(((1-rho)*np.identity(4)+rho*np.ones((4,4)))) + np.diag([0.1, 0.05, 0, -0.05])
print('Theta_test:\n', Theta_test)
Sigma_test = np.linalg.inv(Theta_test)
print('Sigma_test:\n', Sigma_test)
print('Theta_test:\n', np.linalg.inv(Sigma_test))

#plot_performance(Sigma0=Sigma4, x=np.linspace(0, 0.6, 10), patMSE=True, Cov=1**2*Hessian(Sigma4), n=1000, smooth=True)
#plot_performance(Sigma0=Sigma_test, x=np.linspace(0, 0.6, 10), patMSE=True, Cov=1**2*Hessian(Sigma4), n=100, smooth=True) # patMSE goes to zero if the diagonal is not clustered

#np.set_printoptions(linewidth=np.inf)
np.set_printoptions(threshold=np.inf)

print('Hessian(Sigma9):\n', Hessian(Sigma9))
print('evals9', np.linalg.eigvals(Hessian(Sigma9)))
print('evals4', np.linalg.eigvals(Hessian(Sigma4)))
plot_performance(Theta0=Theta9, x=np.linspace(0, 0.6, 5), patMSE=True, Cov=1**2*Hessian(np.linalg.inv(Theta9)), n=50, smooth=True) #good simulation
#plot_performance(Theta0=Theta4, x=np.linspace(0, 0.6, 5), patMSE=True, Cov=1**2*Hessian(np.linalg.inv(Theta4)), n=100, smooth=True) #


def create_band_matrix(n, diag_val=1.0, first_off_diag=0.9, second_off_diag=0.8, third_off_diag=0.7):
    """Creates a symmetric band matrix with specified diagonals in NumPy.

    Args:
        n: Size of the square matrix (n x n).
        diag_val: Value for the main diagonal (default: 1.0).
        first_off_diag: Value for the first off-diagonal (default: 0.9).
        second_off_diag: Value for the second off-diagonal (default: 0.8).
        third_off_diag: Value for the third off-diagonal (default: 0.7).

    Returns:
        A NumPy array representing the symmetric band matrix.
    """

    # Create a zero-filled matrix
    matrix = np.zeros((n, n))

    # Fill the diagonal
    np.fill_diagonal(matrix, diag_val)

    # Fill the first off-diagonal (above and below)
    matrix += np.diag(np.full(n - 1, first_off_diag), k=1)  # k=1 for first off-diagonal
    matrix += np.diag(np.full(n - 1, first_off_diag), k=-1)  # k=-1 for first off-diagonal below

    # Fill the second off-diagonal (above and below)
    if n > 2:  # Check if matrix size allows for second off-diagonal
        matrix += np.diag(np.full(n - 2, second_off_diag), k=2)  # k=2 for second off-diagonal
        matrix += np.diag(np.full(n - 2, second_off_diag), k=-2)  # k=-2 for second off-diagonal below

    # Fill the third off-diagonal (above and below) - optional (modify if needed)
    if n > 3:  # Check if matrix size allows for third off-diagonal
        matrix += np.diag(np.full(n - 3, third_off_diag), k=3)  # k=3 for third off-diagonal
        matrix += np.diag(np.full(n - 3, third_off_diag), k=-3)  # k=-3 for third off-diagonal below

    return matrix


# Example usage
print('band9:\n', create_band_matrix(9))
print('band4:\n', create_band_matrix(4))
Sigma4band = create_band_matrix(4)
print('Theta4band:\n', np.round(np.linalg.inv(Sigma4band),2))
#plot_performance(Sigma0=np.linalg.inv(Sigma4band), x=np.linspace(0, 1.2, 10), patMSE=True, Cov=1**2*Hessian(np.linalg.inv(Sigma4band)), n=50, smooth=True) #penalization fails, SLOPE surprisingly even worse than Lasso
Theta4_custom = np.array([[1, 0.8, 0.8, 0.8], [0.8, 1, 0.8, 0.8], [0.8, 0.8, 1, 0.8], [0.8, 0.8, 0.8, 1]])
print('Sigma4_custom:\n', Sigma4_custom)
#print('Theta4_custom:\n', np.linalg.inv(Sigma4_custom))

#plot_performance(Theta0=Theta4_custom, x=np.linspace(0, 0.6, 10), patMSE=True, Cov=1**2*Hessian(Theta4_custom), n=50, smooth=True) #good simulation


