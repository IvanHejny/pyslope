from src.slope.solvers import*
import numpy as np
import random

Sigma2 = np.array([[2, 0], [0, 2]])
Sigma3 = np.array([[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]])
Sigma3i = np.identity(3)
Sigma4 = np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0.5], [0, 0, 0.5, 1]])
Sigma4b = np.array([[5, 1, 1, 0], [1, 2, 1, 1], [1, 1, 1, 0.5], [0, 1, 0.5, 1]])

#print(np.linalg.det(Sigma4b))


# print(sqsq(Sigma3))
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
print(vech_to_mat(np.array([1, 0.3, 0.4, 1, 0, 2])))
print(':\n', mat_to_vec(Sigma2))

def D(p):
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


def Hessian(Sigma):
    p = len(Sigma[0])
    return np.round(D(p).T @ np.kron(Sigma, Sigma) @ D(p) * 0.5, 3)


rho=2/3
Sigma4c = np.array([[1, rho, rho, 0], [rho, 1, rho, 0], [rho, rho, 1, 0], [0, 0, 0, 1]])

#Theta4c = np.array([[4, 1.2, 1.2, 0], [1.2, 4, 1.2, 0], [1.2, 1.2, 4, 0], [0, 0, 0, 4]])
print('Sigma4c:\n', Sigma4c, '\n Theta4c:\n', np.linalg.inv(Sigma4c))

#print('stacked Sigma4c:\n', stack_columns(Sigma4c))
print('mat_to_vech Sigma4c:\n', mat_to_vech(Sigma4c))

print('C_tilde:\n', Hessian(Sigma4c))


print('Sigma3:\n', Sigma3)
#print('Theta3:\n', np.linalg.inv(Sigma3))
print('C_tilde:\n', Hessian(Sigma3))
print('D:\n', D(3))
print('D^TD:\n', D(3).T @ D(3))


C_tilde = Hessian(Sigma4c)
print('C_tilde:\n', C_tilde)
Theta4c = np.linalg.inv(Sigma4c)
signal = pattern(np.round(mat_to_vech(Theta4c)))
print('Theta4c:\n', Theta4c, '\n', pattern(np.round(mat_to_vech(Theta4c),4)))
lambdas = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
#for n in range(1000,1005):
#    print('gslp:\n', vech_to_mat(pgd_slope_b_0_FISTA(C=C_tilde, W=np.array([1,0.5, 0.3, 1, 0.85, -0.6, 1.1, 0.2, -0.4, 0.1]), b_0=signal, lambdas=0*lambdas, t=0.2, n=n)))
for n in range(200,201):
    print('gslp:\n', vech_to_mat(pgd_slope_b_0_FISTA(C=C_tilde, W=np.array([1, 0.5, 0.3, 1, 0.85, -0.6, 1.1, 0.2, -0.4, 0.1]),
                                       b_0=signal, lambdas=0.1 * lambdas, t=0.2, n=n)))
    print('diff:', n, '\n', np.round(vech_to_mat(pgd_slope_b_0_FISTA(C=C_tilde, W=np.array([1, 0.5, 0.3, 1, 0.85, -0.6, 1.1, 0.2, -0.4, 0.1]),
                                       b_0=signal, lambdas=0.1 * lambdas, t=0.2, n=n))-vech_to_mat(pgd_slope_b_0_FISTA(C=C_tilde, W=np.array([1, 0.5, 0.3, 1, 0.85, -0.6, 1.1, 0.2, -0.4, 0.1]),
                                       b_0=signal, lambdas=0.1 * lambdas, t=0.2, n=n-1)),5))


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
            Theta[i] = diag_Theta[diag_counter]
            diag_counter += 1
        else:
            Theta[i] = low_Theta[low_counter]
            low_counter += 1
    return Theta
#print('join_diag_and_low:\n', join_diag_and_low(split_diag_and_low(Theta4c)[0], split_diag_and_low(Theta4c)[1]))


def pgd_gslope_Theta0_ISTA(C, W, vechTheta0, lambdas, t, n):
    """Minimizes: 1/2 u^T*C*u-u^T*W+J'_{lambda}(vechTheta0; u),
     where J'_{lambda}(b^0; u) is the directional SLOPE derivative
       Parameters
       ----------
       C: np.array
           covariance matrix of the data
       W: np.array
           p-dimensional vector, in our paper it arises from normal N(0, \sigma^2 * C ),
           where \sigma^2 is variance of the noise
       b_0: np.array
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
    #W = [np.float(i) for i in W]
    #lambdas = [np.float(i) for i in lambdas]
    u_0 = np.zeros(len(vechTheta0))
    prox_step = u_0
    stepsize_t = np.float64(t)
    for i in range(n):
        grad_step = prox_step - stepsize_t * (C @ prox_step - W)
        grad_step_diag = split_diag_and_low(grad_step)[0]
        grad_step_low = split_diag_and_low(grad_step)[1]

        prox_step_low = prox_slope_b_0(split_diag_and_low(vechTheta0)[1], grad_step_low, split_diag_and_low(lambdas)[1] * stepsize_t) #prox step only on the lower diagonal entries
        prox_step_diag = grad_step_diag
        prox_step = join_diag_and_low(prox_step_diag, prox_step_low)
    return(prox_step)

print('evals:\n', np.linalg.eigvals(C_tilde))
for n in range(200,201):
    print('gslp_onlylow_pen:\n', vech_to_mat(pgd_gslope_Theta0_ISTA(C=C_tilde, W=np.array([1, 0.5, 0.3, 1, 0.85, -0.6, 1.1, 0.2, -0.4, 0.1]),
                                       vechTheta0=signal, lambdas=0.1 * lambdas, t=0.2, n=n)))
    print('diff:', n, '\n', np.round(vech_to_mat(pgd_gslope_Theta0_ISTA(C=C_tilde, W=np.array([1, 0.5, 0.3, 1, 0.85, -0.6, 1.1, 0.2, -0.4, 0.1]),
                                       vechTheta0=signal, lambdas=0.1 * lambdas, t=0.2, n=n))-vech_to_mat(pgd_gslope_Theta0_ISTA(C=C_tilde, W=np.array([1, 0.5, 0.3, 1, 0.85, -0.6, 1.1, 0.2, -0.4, 0.1]),
                                       vechTheta0=signal, lambdas=0.1 * lambdas, t=0.2, n=n-1)),9))