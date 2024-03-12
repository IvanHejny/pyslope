import numpy as np


def soft_thresholding(a, kappa):
    """
    Applies soft thresholding to an array.
    """
    return np.sign(a) * np.maximum(np.abs(a) - kappa, 0)

#print('threshold_test:', soft_thresholding(a=np.array([-3, -2, 0, 1.4, 2.7]), kappa=1))

'''
def loss(u, C, A, w, beta0, lambdas):
    """
    Calculates the loss function.
    """
    E_1 = np.diag(np.sign(A @ beta0))  #sgn(A beta0)
    ind = np.where(np.diag(E_1) == 0)[0]
    F_0 = A[ind, :]  #(I-|sgn(A*beta0)|)A
    w_tilde = w - lambdas * np.ones(p).T @ E_1 @ A  #w-lambda* 1^T * sgn(A beta0)

    return 0.5 * u.T@C@u - u.T@w_tilde + lambdas * np.sum(np.abs(F_0@u))
'''

# Let f(b)=lam*||b||_1 (Lasso), f_A(b)=lam*||Ab||_1 (gen Lasso), then f_A'(b;u)=f'(Ab,Au).
# Goal:  Minimize u^TCu - u^T w + f_A'(b;u).
# Equiv: Minimize u^TCu - u^T tilde_w + lam*||F u||_1, (can be solved directly by ADMM, p.44 https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf)
# Here: tilde_w=w-lambda* 1^T * sgn(A beta0), and F = (I-|sgn(A*beta0)|)A
def admm_glasso(C, A, w, beta0, lambdas, rho=1.0, x0=None, u0=None, z0=None, iter=100):
    """
    minimizes: u^TCu -u^T w + f_A'(b;u), where f_A(b)=lam*||Ab||_1 is the generalized Lasso penalty.
    """
    p = len(beta0)
    E_1 = np.diag(np.sign(np.round(A @ beta0, 10)))  # sgn(A beta0): k x k diagonal sign matrix, rounding to prevent numerical errors like np.sign(4.4408921e-16)=1.
    ind = np.where(np.diag(E_1) == 0)[0]
    F_0 = A[ind, :]  # F_0*x = (I-|sgn(A*beta0)|)A*x

    w_tilde = w - lambdas * np.ones(A.shape[0]).T @ E_1 @ A  #w-lambda* 1^T * sgn(A beta0)A

    #minimizing x^TCx -x^T w_tilde + lam*||F_0 x||_1

    m = F_0.shape[0]  # number of rows in F0
    if x0 is None:
        x0 = np.zeros(p)
    if z0 is None:
        z0 = np.zeros(m)
    if u0 is None:
        u0 = np.zeros(m)

    x = x0
    z = z0
    u = u0

    Q = np.linalg.inv(C + rho * F_0.T @ F_0)
    #print(x)

    for i in range(iter): #ADMM iterations
        x = Q @ (w_tilde + rho * F_0.T @ (z-u))
        z = soft_thresholding(F_0 @ x + u, lambdas / rho)
        u = u + F_0 @ x - z
        # loss_x = loss(x, C, A, w, beta0, lambdas)
        # (np.round(x,3), np.round(loss_x,8))

    return x


'''
#print(admm_glasso(C=np.identity(3), A=np.array([[1, -1, 0], [0, 1.15, -1.15]]), w=np.array([1, 1, 1]), beta0=np.array([1, 4, 4]), lambdas=400, iter=100))
#print(admm_glasso(C=np.identity(3), A=np.array([[1, -1, 0], [0, 1.15, -1.15]]), w=np.array([1, 1, 1]), beta0=np.array([1, 5, 5]), lambdas=400, iter=100))
#np.diag(np.sign(A @ beta0))
#print('A@b0:\n', np.array([[1, -1, 0], [0, 1.15, -1.15]]) @ np.array([1, 5, 5]))
#print('sign(A@b0):\n', np.sign(np.array([[1, -1, 0], [0, 1.15, -1.15]]) @ np.array([1, 5, 5])))
#print('sign(A@b0):\n', np.sign(np.round(np.array([[1, -1, 0], [0, 1.15, -1.15]]) @ np.array([1, 5, 5]),10)))
#print('[1,4,4]:\n', np.diag(np.sign(np.array([[1, -1, 0], [0, 1.15, -1.15]]) @ np.array([1, 4, 4]))))
#print('[1,5,5]:\n', np.diag(np.sign(np.array([[1, -1, 0], [0, 1.15, -1.15]]) @ np.array([1, 5, 5]))))
#print(np.sign(5*1.15-5*1.15))
'''
# penalty matrices in generalized lasso
def AFL(p, a=1): #Fused Lasso + Lasso, tuned by a. For a=0, Lasso.
    A = np.zeros((2 * p - 1, p))
    for i in range(p - 1):
        A[i][i] = 1
        A[i][i + 1] = -1  # clustering penalty
    for i in range(p - 1, 2 * p - 1):
        A[i][i - (p - 1)] = a  # sparsity penalty
    normalization = np.sum(A)
    return A#p*(1/normalization)*A #A

def AFLmon(p, b=0.1): # Fused Lasso + Monotone Lasso penalty
    A = np.zeros((2 * p - 1, p))
    for i in range(p - 1):
        A[i][i] = 1
        A[i][i + 1] = -1  # clustering penalty
    for i in range(p - 1, 2 * p - 1):
        A[i][i - (p - 1)] = 1+b*(i-(p-1)+1)  # monotone sparsity penalty
        normalization = np.sum(A)
    return A#p*(1/normalization)*A #A
#print('AFLmon:\n', AFLmon(7, 0.1))

def Acustom(a , b): # a: sparsity penalty, b: clustering penalty
    p = len(a)
    A = np.zeros((2 * p - 1, p))
    for i in range(p - 1):
        A[i][i] = b[i]
        A[i][i + 1] = -b[i]  # clustering penalty
    for i in range(p):
        A[p-1+i][i - p] = a[i]  # sparsity penalty
        normalization = np.sum(A)
    return A

print('Acustom:\n', Acustom(a= [1,2,3], b= [5,1]))




#print('A:\n', A)
'''
p=9
A = np.zeros((p-1, p))
for i in range(p - 1):
    A[i][i] = 1
    A[i][i + 1] = -1
print('A:\n', A)

beta02 = np.array([0, 1])
beta0 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]) #px1 vector

AFL = np.zeros((2 * p - 1, p)) # mxp Gen Lasso matrix
for i in range(p - 1):
    AFL[i][i] = 1
    AFL[i][i + 1] = -1 #clustering penalty
for i in range(p-1, 2*p -1):
    AFL[i][i - (p-1)] = 1 #sparsity penalty
#print('AFL\n:', AFL)


p = 2
AFLsmall = np.zeros((2 * p - 1, p))
for i in range(p - 1):
    AFLsmall[i][i] = 1
    AFLsmall[i][i + 1] = -1 #clustering penalty
for i in range(p-1, 2*p -1):
    AFLsmall[i][i - (p-1)] = 10 #sparsity penalty
#print('AFL\n:', AFLsmall)


'''





