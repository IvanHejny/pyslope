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
def mat_to_vech(arr):  # input: p x p matrix output: p(p+1)/2 x 1 vector
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
print('C_tilde', C_tilde)
Theta4c = np.linalg.inv(Sigma4c)
signal = pattern(np.round(mat_to_vech(Theta4c)))
print(Theta4c, '\n', pattern(np.round(mat_to_vech(Theta4c),4)))
lambdas = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
print('gslp:', vech_to_mat(pgd_slope_b_0_FISTA(C=C_tilde, W=np.array([1,0.5, 0.3, 1, 0.85, -0.6, 1.1, 0.2, -0.4, 0.1]), b_0=signal, lambdas=5*lambdas, t=0.30, n=20)))

