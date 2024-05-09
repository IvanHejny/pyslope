from src.slope.solvers import *
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

def split_diag_and_low(vechTheta): # returns diagonal and lower triangular part of a symmetric matrix ( as two np.vectors )
    #  input can be a matrix or vectorized lower triangular part of that matrix
    if len(vechTheta.shape) == 2:  # if the input is a matrix, vectorize it
        vechTheta = mat_to_vech(vechTheta)
    p = int(-0.5 + np.sqrt(0.25 + 2 * len(vechTheta)))
    # print('p:', int(-0.5 + np.sqrt(0.25 + 2 * len(mat_to_vech(Theta4c)))))
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


def pgd_gslope_Theta0_ISTA(C, W, Theta0, lambdas, n, t=None):
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
    if t==None:
        t = 1/np.max(np.linalg.eigvals(C))  # default stepsize = 1/max(eigenvalues of C) to guarantee O(1/n^2) convergence
        #t = np.float32(t) #np.real(t)
        t=np.float32(np.real(t))
    stepsize_t = t #np.float32(t)
    for i in range(n):
        grad_step = prox_step - stepsize_t * (C @ prox_step - W)
        grad_step_diag = split_diag_and_low(grad_step)[0]
        grad_step_low = split_diag_and_low(grad_step)[1]

        prox_step_low = prox_slope_b_0(split_diag_and_low(vechTheta0)[1], grad_step_low, lambdas * stepsize_t) #prox step only on the lower diagonal entries
        prox_step_diag = grad_step_diag
        prox_step = join_diag_and_low(prox_step_diag, prox_step_low)
    return(prox_step)

def pgd_gslope_Theta0_FISTA(C, W, Theta0, lambdas_low, n=None, t=None, tol=1e-4, max_iter=2000):
    """
    Minimizes the objective function:
        1/2 u^T*C*u - u^T*W + J'_{lambdas_low}(vechTheta0; u),

    where J'_{lambdas_low}(b^0; u) is the directional SLOPE derivative,
    lambdas_low penalizes only the lower triangular part of the precision matrix.

    Parameters
    ----------
    C : np.array
        Covariance matrix of the data.
    W : np.array
        p-dimensional vector. In our paper, it arises from a normal distribution
        N(0, \sigma^2 * C), where \sigma^2 is the variance of the noise.
    Theta0 : np.array
        Pattern vector of the true signal.
    lambdas_low : np.array
        Vector of regularization weights.
    t : np.float, optional
        Step size. Default is None.
    n : integer, optional
        Number of steps before termination. Default is None.
    tol : float, optional
        Tolerance level for termination based on the norm between consecutive iterates.
        Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. Default is 1000.

    Returns
    -------
    array
        The unique solution to the minimization problem, given by a vector u.
    """
    vechTheta0 = pattern(np.round(mat_to_vech(Theta0),3))
    u_k = np.zeros(len(vechTheta0))  # initial point
    u_kmin2 = u_k  # Lagged iterate u_{k-2}
    u_kmin1 = u_k  # Lagged iterate u_{k-1}
    v = u_k
    if t==None:
        t = 1/np.max(np.linalg.eigvals(C))  # default stepsize = 1/max(eigenvalues of C) to guarantee O(1/n^2) convergence
        #t = np.float32(t) #np.real(t)
        t=np.float32(np.real(t))
    stepsize_t = t #np.float32(t)
    #u_k=np.zeros(len(vechTheta0))
    if n is None:
        max_iter = max_iter
    else:
        max_iter = n
    for k in range(2, max_iter):
        v = u_kmin1 + ((k-2)/(k+1))*(u_kmin1-u_kmin2)
        grad_step = v - stepsize_t * (C @ v - W)
        grad_step_diag = split_diag_and_low(grad_step)[0]
        grad_step_low = split_diag_and_low(grad_step)[1]

        prox_step_low = prox_slope_b_0(split_diag_and_low(vechTheta0)[1], grad_step_low, lambdas_low * stepsize_t) #prox step only on the lower diagonal entries
        prox_step_diag = grad_step_diag
        u_k = join_diag_and_low(prox_step_diag, prox_step_low)
        u_kmin2 = u_kmin1
        u_kmin1 = u_k
        if n is None:
            p = Theta0.shape[0]
            norm_diff = np.linalg.norm(u_kmin1 - u_kmin2) / p
            if norm_diff < tol:
                break
            elif k == max_iter - 1:
                print('Warning: Maximum number of iterations', max_iter, ' reached. Convergence of FISTA is slow. The stepsize 1/max(eigenvalues of C) = ', stepsize_t ,' might be too small. Also, Theta0 might be close to singular.')
    print('final_iter:', k)  # uncomment line for final iterate
    return u_k



#convergence speed issues, number of iterates required might be in order of hundreds if 1/max(eigenvalues of C) is small < 0.2, or if Theta0 is close to singular
'''
#d=2
#Theta4c = np.array([[d, 1.2, 1.2, 0], [1.2, d, 1.2, 0], [1.2, 1.2, d, 0], [0, 0, 0, d]]) # increasing diagonal d from 1.5, to 2, to 3, to 4 increases stepsize from 0.107, to 0.75 to 3.75 to 8.96, and final_iter with tol=1e-4, decrese 333, 141 to 77 to 51.
rho = 0.8
Sigma4c = np.array([[1, rho, rho, rho],[rho, 1, rho, rho],[rho, rho, 1, rho],[rho, rho, rho, 1]])
Theta4c = np.linalg.inv(Sigma4c)

C_tilde = Hessian(np.linalg.inv(Theta4c))
lambdas_low = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]) / 3.5
print('evals:\n', np.linalg.eigvals(C_tilde))
print('1/max(eval):\n', 1/max(np.linalg.eigvals(C_tilde)))
# we want stepsize as large as possible, i.e. max(eval(Hessian)) as small as possible, i.e. 'precision Theta0 as 'large' as possible on the diagonal'

print('Theta4c:\n', Theta4c)
W = np.array([1, 0.5, 0.3, 1, 0.85, -0.6, 1.1, 0.2, -0.4, 0.1])
#est_truth_gslp = vech_to_mat(pgd_slope_b_0_FISTA(C=C_tilde, W=W, b_0=mat_to_vech(Theta4c), lambdas=8*0.1 * np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])/55, t=0.2, n=1000))
#print('est_truth_gslp:\n', est_truth_gslp)

est_truth_fista4 = vech_to_mat(pgd_gslope_Theta0_FISTA(C=C_tilde, W=W, Theta0=Theta4c, lambdas_low=8 * 0.1 * lambdas_low, n=10000))
print('est_truth_low_fista:\n', est_truth_fista4)
terminated_fista = vech_to_mat(pgd_gslope_Theta0_FISTA(C=C_tilde, W=W, Theta0=Theta4c, lambdas_low=8 * 0.1 * lambdas_low))
print('terminated_fista:\n', terminated_fista)
print('error:\n', terminated_fista - est_truth_fista4)


Sigma8_block = np.block([[Sigma4c, np.zeros((4, 4))], [np.zeros((4, 4)), Sigma4c]])
Theta8_block = np.linalg.inv(Sigma8_block)
print('Theta8_block:\n', np.round(Theta8_block,2))
W = np.random.multivariate_normal(np.zeros(36), np.identity(36))
est_truth_fista_block8 = vech_to_mat(pgd_gslope_Theta0_FISTA(C=Hessian(np.linalg.inv(Theta8_block)), W=W, Theta0=Theta8_block, lambdas_low=8 * 0.1 * lin_lambdas(8 * 7 / 2), n=10000))
terminated_fista_block8 = vech_to_mat(pgd_gslope_Theta0_FISTA(C=Hessian(np.linalg.inv(Theta8_block)), W=W, Theta0=Theta8_block, lambdas_low=8 * 0.1 * lin_lambdas(8 * 7 / 2)))
print('est_truth_fista_block8:\n', np.round(est_truth_fista_block8, 4))
print('terminated_fista_block8:\n', np.round(terminated_fista_block8, 4))
print('error_block8:\n', np.round(terminated_fista_block8 - est_truth_fista_block8, 4))

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

def gpatternMSE(Theta0, lambdas_low, n, C=None, Cov=None, genlasso=False, A = None , tol=1e-4, max_iter=2000):
    """
    Calculate root mean squared error (RMSE) E[||u_hat||^2]^{1/2}, probability of pattern recovery, and PatternRMSE
    by sampling the optimization error, which minimizes u^T*C*u/2-u^T*W+J'_{lambdas_low}(vechTheta0; u), W~N(0,Cov).
    Here the penalty lambdas_low only penalizes the lower triangular part of the precision matrix.

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
            - Proportion of correct pattern recoveries. (Of the sub-diagonal pattern of the precision matrix)
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
    #stepsize_t = 1/max(np.linalg.eigvals(C))  # stepsize <= 1/max eigenvalue of C;
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
            u_hat = pgd_gslope_Theta0_FISTA(C=C, W=W, Theta0=Theta0, lambdas_low=lambdas_low, tol=tol, max_iter=max_iter) # before n was 400, now its adaptive to the stopping criterium
            #print('u_hat\n:', vech_to_mat(np.round(u_hat, 4))) # uncomment line for debugging, and understanding pattern error

        # Calculate MSE
        norm2 = np.linalg.norm(u_hat) ** 2
        MSE += norm2
        # Calculate pattern error: MSE = patMSE + restMSE
        pat_norm2 = np.linalg.norm((np.identity(len(vechTheta0))-pat_proj) @ u_hat) ** 2
        patMSE += pat_norm2  # patMSE is the Euclidean distance of u_hat from the pattern space of vechTheta0

        # Check pattern recovery on off-diagonal entries
        pat_signal_low = split_diag_and_low(pat_signal)[1]
        u_hat_low = split_diag_and_low(u_hat)[1]
        if all(pattern(pat_signal_low + 0.000001 * np.round(u_hat_low, 3)) == pat_signal_low):
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

# PERFORMANCE METRICS FOR VARIOUS PRECISION MATRICES
# Here we test asymptotic error, and asymptotic pattern recovery of graphical SLOPE and Lasso for various precision matrices

# Compound precision matrix
rho = 0.4
Sigma4_compound = np.array([[1, rho, rho, rho],[rho, 1, rho, rho],[rho, rho, 1, rho],[rho, rho, rho, 1]])
Theta4_compound = np.linalg.inv(Sigma4_compound)
#print('Theta4_compound:\n', Theta4_compound)
#print('stepsize=1/max(eval):\n', np.linalg.eigvals(Hessian(Sigma4_compound)), 1/max(np.linalg.eigvals(Hessian(Sigma4_compound))))
#print('gpatternMSE_compound:\n', gpatternMSE(Theta0=Theta4_compound, lambdas_low = 0.5*lin_lambdas(6), n=10)) # recovers pattern in compound symmetric perfectly for large penalty

#Block compound precision matrix
Sigma8_block = np.block([[Sigma4_compound, np.zeros((4, 4))], [np.zeros((4, 4)), Sigma4_compound]])
Theta8_block = np.linalg.inv(Sigma8_block)
#print('Theta8_block:\n', np.round(Theta8_block,2))
#print('gpatternMSE_block:\n', gpatternMSE(Theta0=Theta8_block, lambdas_low = 0.5*lin_lambdas(8*7/2), n=10))
#print('gpatternMSE_block:\n', gpatternMSE(Theta0=Theta8_block, lambdas_low = 1*lin_lambdas(8*7/2), n=10))
#print('gpatternMSE_block:\n', gpatternMSE(Theta0=Theta8_block, lambdas_low = 2*lin_lambdas(8*7/2), n=10))

#Band precision matrix
Theta4_band = np.array([[1, 0.9, 0.8, 0.7], [0.9, 1, 0.9, 0.8], [0.8, 0.9, 1, 0.9], [0.7, 0.8, 0.9, 1]])
#print('gpatternMSE_band:\n', gpatternMSE(Theta0=Theta4_band, lambdas_low = 5*lin_lambdas(6), n=10)) # does not recover patterns in the band matrix properly
#print('lin_lambdas\n:', lin_lambdas(6))

#AR_precision matrix
Theta4_AR = np.array([[3, -1, 0, 0], [-1, 4, -1, 0], [0, -1, 4, -1], [0, 0, -1, 3]])
#print('Theta4_AR:\n', Theta4_AR)
#print('Sigma4_AR:\n', np.linalg.inv(Theta4_AR))
#print('gpatternMSE_AR:\n', gpatternMSE(Theta0=Theta4_AR, lambdas_low = 30*np.array([2, 1.5, 1.3, 1.1, 1.05, 1]), n=10)) # recovers pattern perfectly for large penalty, non-linear penalty
#print('gpatternMSE_AR:\n', gpatternMSE(Theta0=Theta4_AR, lambdas_low = 30*np.array([2, 1.8, 1.6, 1.4, 1.2, 1]), n=10)) # much worse, but still recovers pattern for large penalty
#print('gpatternMSE:\n', gpatternMSE(Theta0=Theta4_AR, lambdas_low = 50*np.ones(6), n=10))




def plot_gperformance(Theta0, x, n, lambdas_low=None, C=None, Cov=None, patMSE=False, flasso=False, A_flasso = None, glasso=False, A_glasso = None, smooth = None, tol=1e-4, max_iter=2000):
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
        resultSLOPE = gpatternMSE(Theta0=Theta0, C=C, Cov=Cov, lambdas_low=x[i] * lambdas_low, n=n, tol=tol, max_iter=max_iter)
        resultLasso = gpatternMSE(Theta0=Theta0, C=C, Cov=Cov, lambdas_low=x[i] * np.ones(len(vechTheta0_low)), n=n, tol=tol, max_iter=max_iter)
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
    #plt.plot(x, PattSLOPE, label='recovery SLOPE', color='green', linestyle='dashed', lw=1.5)  # Plot probability of pattern recovery by SLOPE
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
def plot_pattern_recovery(Theta0, x, n, lambdas_low=None, C=None, Cov=None, flasso=False, A_flasso = None, glasso=False, A_glasso = None, smooth = None):
    """
    Plot probability of pattern recovery for different regularization methods.

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
    - The function plots pattern recovery for SLOPE (Sorted L-One Penalized Estimation), FLasso, and ConFLasso.
    - The pattern recovery is plotted against the penalty scaling factor 'alpha'.
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
    PattLasso = np.empty(shape=(0,))
    Pattglasso = np.empty(shape=(0,))
    Pattflasso = np.empty(shape=(0,))

    for i in range(len(x)):
        resultSLOPE = gpatternMSE(Theta0=Theta0, C=C, Cov=Cov, lambdas_low=x[i] * lambdas_low, n=n)
        resultLasso = gpatternMSE(Theta0=Theta0, C=C, Cov=Cov, lambdas_low=x[i] * np.ones(len(vechTheta0_low)), n=n)
        #resultLasso = gpatternMSE(Theta0=Theta0, C=C, Cov=Cov, lambdas=x[i] * lambdas, n=n, genlasso=True, A=x[i]*Acustom(a=np.ones(len(b_0)), b=np.zeros(len(b_0)-1)))
        # using admm for lasso instead of pgd_slope_b_0_FISTA

        PattSLOPE = np.append(PattSLOPE, resultSLOPE[1])
        PattLasso = np.append(PattLasso, resultLasso[1])

        if flasso == True:
            if A_flasso is None:
                A_flasso = Acustom(a=np.ones(len(vechTheta0_low)), b=np.ones(len(vechTheta0_low) - 1))
            resultflasso = gpatternMSE(Theta0=Theta0, C=C, Cov=Cov, lambdas_low=x[i] * lambdas_low, n=n, genlasso=True, A=x[i]*A_flasso)
            Pattflasso = np.append(Pattflasso, resultflasso[1])

        if glasso == True:
            if A_glasso is None:
                A_glasso = Acustom(a=np.ones(len(vechTheta0_low)), b=np.ones(len(vechTheta0_low) - 1))
            resultglasso = gpatternMSE(Theta0=Theta0, C=C, Cov=Cov, lambdas_low=x[i] * lambdas_low, n=n, genlasso=True, A=x[i] * A_glasso)
            Pattglasso = np.append(Pattglasso, resultglasso[1])
            #  Supportglasso = np.append(Supportglasso, resultglasso[2])
    if smooth == True:
        #  Spline interpolation for smoother curve
        x_smooth = np.linspace(x.min(), x.max(), 20 * (len(x) - 1) + 1)
        spl2 = PchipInterpolator(x, PattSLOPE)  #
        PattSLOPE = spl2(x_smooth)

        if flasso == True:
            spl5 = PchipInterpolator(x, Pattflasso)
            Pattflasso = spl5(x_smooth)
        if glasso == True:
            spl7 = PchipInterpolator(x, Pattglasso)
            Pattglasso = spl7(x_smooth)

        x = x_smooth
    # Plot the functions on the same graph
    plt.figure(figsize=(6, 6))
    plt.plot(x, PattSLOPE, label='recovery SLOPE', color='green', linestyle='dashed', lw=1.5)  # Plot probability of pattern recovery by SLOPE
    if flasso == True:
        plt.plot(x, Pattflasso, label='recovery FLasso', color='orange', linestyle='dashed', lw=1.5)
    if glasso == True:
        plt.plot(x, Pattglasso, label='recovery ConFLasso', color='purple', linestyle='dashed', lw=1.5)
    #plt.plot(x, PattLasso, label='pattern recovery Lasso', color='blue', linestyle='dashed', lw=1.5)  # Plot prob of pattern by Lasso
    #plt.plot(x, SupportSLOPE, label='support recovery SLOPE', color='green', linestyle='-.', lw=1.5, alpha=0.5)  # Plot prob of support recovery by SLOPE
    #plt.plot(x, SupportLasso, label='support recovery Lasso', color='blue', linestyle='-.', lw=1.5, alpha=0.5)  # Plot prob of support recovery by Lasso

    # Increase the size of x-axis and y-axis tick labels
    plt.xticks(fontsize=14)  # font size for x-axis tick labels
    plt.yticks(fontsize=14)  # font size for y-axis tick labels
    plt.xlabel(r'$\alpha$', fontsize=16)  # penalty scaling

    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Compound symmetric precision matrix


#np.set_printoptions(linewidth=np.inf)
np.set_printoptions(threshold=np.inf)


# PERFORMANCE PLOTS
# Here we seperately plot rmse performance and pattern recovery for SLOPE and Lasso
# The plots are very rough, for better results, increase the number of simulations n, for quicker results, decrease n, or p or increase tol
# To display pattern recovery for SLOPE in the same plot, uncomment the corresponding line in the plot_gperformance function. Or Use plot_pattern_recovery function.
# If you do not wish to see the number of iterates in FISTA, comment second to last line in pgd_gslope_Theta0_FISTA function.


# Compound symmetric precision matrix

Sigma9 = comp_sym_corr(0.1, 9)
Theta9 = np.linalg.inv(Sigma9)
print('Theta9:\n', np.round(Theta9, 2))
print('stepsize in FISTA:\n', 1/max(np.linalg.eigvals(Hessian(Sigma9))))
#plot_gperformance(Theta0=Theta9, lambdas_low=lin_lambdas(9 * 8 / 2), x=np.linspace(0, 2, 7), patMSE=True, n=100, tol=1e-4, smooth=True) #G # rho=0.1 SLOPE beats Lasso
#plot_gperformance(Theta0=Theta9, lambdas_low=bh_lambdas(9*8/2, q=0.9), x=np.linspace(0, 2, 5), patMSE=True, n=50, smooth=True) # BH does not improve linear sequence

Sigma20 = comp_sym_corr(0.1, 20)
Theta20 = np.linalg.inv(Sigma20)
#print('Theta20:\n', np.round(Theta20, 2))
#print('eval20:\n', 1/max(np.linalg.eigvals(Hessian(Sigma20))))
#plot_gperformance(Theta0=Theta20, lambdas_low=lin_lambdas(20*19/2), x=np.linspace(0, 2, 5), patMSE=True, n=50, smooth=True) # rho=0.1 SLOPE beats Lasso
#plot_gperformance(Theta0=Theta20, lambdas_low=bh_lambdas(20*19/2, 0.5), x=np.linspace(0, 2, 5), patMSE=True, n=10, smooth=True) # bh not better than linear sequence

# Block diagonal precision matrix


# 20x20 block matrix consisting of two 10x10 compound symmetric blocks
Sigma20_block = np.block([[comp_sym_corr(0.2,10), np.zeros((10, 10))], [np.zeros((10, 10)), comp_sym_corr(0.2,10)]])
print('Theta20_block:\n', np.round(np.linalg.inv(Sigma20_block), 2))
print('lin_lambdas:\n', lin_lambdas(20*19/2))
#plot_gperformance(Theta0=np.linalg.inv(Sigma20_block), lambdas_low=lin_lambdas(20 * 19 / 2), x=np.linspace(0, 2, 7), patMSE=True, n=300, smooth=True, tol=1e-4)

print('bh_lambdas0.05:\n', bh_lambdas(20*19/2, 0.05))
print('bh_lambdas0.5:\n', bh_lambdas(20*19/2, 0.5))
#plot_gperformance(Theta0=np.linalg.inv(Sigma20_block), lambdas_low=bh_lambdas(20 * 19 / 2, 0.5), x=np.linspace(0, 2, 7), patMSE=True, n=200, smooth=True, tol=1e-4) # rho 0.2 SLOPE beats Lasso

my_lambdas = np.ones(int(20*19/2))
my_lambdas[0] = my_lambdas[0] + 10
my_lambdas[1] = my_lambdas[1] + 4
my_lambdas = my_lambdas * (20*19/2)/((20*19-2+my_lambdas[0]+my_lambdas[1])/2)
print('my_lambdas:\n', my_lambdas)
#plot_gperformance(Theta0=np.linalg.inv(Sigma20_block), lambdas_low=my_lambdas, x=np.linspace(0, 2, 7), patMSE=True, n=100, smooth=True, tol=1e-4) #


# Pattern recovery
plot_pattern_recovery(Theta0=np.linalg.inv(Sigma20_block), lambdas_low=lin_lambdas(20*19/2), x=np.linspace(0, 25, 6), n=20, smooth=True) # recovers at 15
plot_pattern_recovery(Theta0=np.linalg.inv(Sigma20_block), lambdas_low=bh_lambdas(20*19/2, 0.05), x=np.linspace(0, 25, 6), n=20, smooth=True) # recovers at 22
plot_pattern_recovery(Theta0=np.linalg.inv(Sigma20_block), lambdas_low=bh_lambdas(20*19/2, 0.5), x=np.linspace(0, 25, 6), n=20, smooth=True) # recovers at 10
plot_pattern_recovery(Theta0=np.linalg.inv(Sigma20_block), lambdas_low=my_lambdas, x=np.linspace(0, 25, 6), n=20, smooth=True) # recovers at 15


'''
Sigma8_block = np.block([[comp_sym_corr(0.8,4), np.zeros((4, 4))], [np.zeros((4, 4)), comp_sym_corr(0.8,4)]])
Theta8_block = np.linalg.inv(Sigma8_block) + 2*np.identity(8)
print('Theta8_block:\n', np.round(Theta8_block, 2))
print('evals8_block:\n', 1/max(np.linalg.eigvals(Hessian(Sigma8_block))))
# Here the BH_sequence is superior to the linear sequence in terms of error
print('bh_lambdas:\n', bh_lambdas(9*8/2, 0.00001))
print('bh_lambdas:\n', bh_lambdas(9*8/2, 0.01))
print('bh_lambdas:\n', bh_lambdas(9*8/2, 0.1))
print('1e-3:', 1e-3)


#plot_gperformance(Theta0=Theta8_block, lambdas_low=lin_lambdas(9*8/2), x=np.linspace(0, 0.6, 5), patMSE=True, n=30, smooth=True, tol=1e-3) #meh # Lasso beats SLOPE
#plot_gperformance(Theta0=Theta8_block, lambdas_low=bh_lambdas(9*8/2, 0.08), x=np.linspace(0, 0.3, 5), patMSE=True, n=100, smooth=False, tol=1e-3) #meh # rho 0.8 meh SLOPE beats Lasso


#plot_gperformance(Theta0=Theta8_block, lambdas_low=lin_lambdas(9*8/2), x=np.linspace(0, 2, 5), patMSE=True, n=200, smooth=True) # rho 0.1
#plot_gperformance(Theta0=Theta8_block, lambdas_low=bh_lambdas(9*8/2, 0.5), x=np.linspace(0, 2, 5), patMSE=True, n=200, smooth=True) # rho 0.1
#plot_pattern_recovery(Theta0=Theta8_block, lambdas_low=bh_lambdas(9*8/2, 0.2), x=np.linspace(0, 5, 5), Cov=0.2**2*Hessian(np.linalg.inv(Theta8_block)), n=100, smooth=True) # does not recover the pattern
'''


# Band precision matrix
p=20
a=-0.4
Theta_p_band = band_mat(p, 5, a, a, a)
print('band_p:\n', band_mat(p, 5, a, a, a))
print('stepsize_band_p:\n', 1 / max(np.linalg.eigvals(Hessian(np.linalg.inv(band_mat(p, 5, a, a, a))))))

#plot_gperformance(Theta0=Theta_p_band, lambdas_low=bh_lambdas(20*19/2, 0.05), x=np.linspace(0, 0.3, 7), patMSE=True, n=20, smooth=False) # not clear
#plot_gperformance(Theta0=Theta_p_band, lambdas_low=lin_lambdas(20*19/2), x=np.linspace(0, 0.4, 10), patMSE=True, n=10, smooth=False) # SLOPE not so good

my_lambdas = np.ones(int(p*(p-1)/2))
my_lambdas[0] = my_lambdas[0] + 10
my_lambdas[1] = my_lambdas[1] + 4
#plot_gperformance(Theta0=Theta_p_band, lambdas_low=my_lambdas, x=np.linspace(0, 0.4, 10), n=30, patMSE=True, smooth=True) # SLOPE a bit better than Lasso


# AR precision matrix
print('Theta4_AR:\n', Theta4_AR)
# comparing how fast the pattern is recovered in AR matrix for different SLOPE penalties
#plot_pattern_recovery(Theta0=Theta4_AR, lambdas_low=np.array([2, 1.8, 1.6, 1.4, 1.2, 1]), x=np.linspace(0, 200, 5), n=30, smooth=True) # slow recovery
#plot_pattern_recovery(Theta0=Theta4_AR, lambdas_low=np.array([2, 1.5, 1.3, 1.1, 1.05, 1]), x=np.linspace(0, 200, 5), n=30, smooth=True) # faster recovery

# recovery with BH_sequence
#print('bh_lambdas_q=0.1\n:',bh_lambdas(6, 0.1), '\n', 'bh_lambdas_q=0.2\n:', bh_lambdas(6, 0.2), '\n', 'bh_lambdas_q=0.5\n:', bh_lambdas(6, 0.6))
#plot_pattern_recovery(Theta0=Theta4_AR, lambdas_low=bh_lambdas(6, 0.1), x=np.linspace(0, 100, 5), n=30, smooth=True) # does not recover the pattern
#plot_pattern_recovery(Theta0=Theta4_AR, lambdas_low=bh_lambdas(6, 0.2), x=np.linspace(0, 100, 5), n=30, smooth=True) # recovers the pattern
#plot_pattern_recovery(Theta0=Theta4_AR, lambdas_low=bh_lambdas(6, 0.5), x=np.linspace(0, 100, 5), n=30, smooth=True) # recovers the pattern
#plot_pattern_recovery(Theta0=Theta4_AR, lambdas_low=bh_lambdas(6, 0.6), x=np.linspace(0, 100, 5), n=30, smooth=True) # does not recover the pattern

#plot_gperformance(Theta0=Theta4_AR, lambdas_low=np.array([2, 1.8, 1.6, 1.4, 1.2, 1])/1.5, x=np.linspace(0, 0.2, 5), n=200, smooth=False)
#plot_gperformance(Theta0=Theta4_AR, lambdas_low=np.array([2, 1.5, 1.3, 1.1, 1.05, 1])/1.325, x=np.linspace(0, 0.2, 10), n=500, smooth=True)
