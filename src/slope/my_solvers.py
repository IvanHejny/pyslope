
import numpy as np
from numpy.linalg import norm
from scipy import sparse

from src.slope.utils import dual_norm_slope, prox_slope, sl1_norm



def prox_slope(beta, lambdas):
    """Compute the sorted L1 proximal operator.

    Parameters
    ----------
    beta : array
        vector of coefficients
    lambdas : array
        vector of regularization weights

    Returns
    -------
    array
        the result of the proximal operator
    """
    beta_sign = np.sign(beta)
    beta = np.abs(beta)
    ord = np.flip(np.argsort(beta))
    beta = beta[ord]

    p = len(beta)

    s = np.empty(p, np.float64)
    w = np.empty(p, np.float64)
    idx_i = np.empty(p, np.int64)
    idx_j = np.empty(p, np.int64)

    k = 0

    for i in range(p):
        idx_i[k] = i
        idx_j[k] = i
        s[k] = beta[i] - lambdas[i]
        w[k] = s[k]

        while (k > 0) and (w[k - 1] <= w[k]):
            k = k - 1
            idx_j[k] = i
            s[k] += s[k + 1]
            w[k] = s[k] / (i - idx_i[k] + 1)

        k = k + 1

    for j in range(k):
        d = max(w[j], 0.0)
        for i in range(idx_i[j], idx_j[j] + 1):
            beta[i] = d

    beta[ord] = beta.copy()
    beta *= beta_sign

    return beta

def prox_slope_isotonic(beta, lambdas):
    """Compute the sorted L1 proximal operator.

    Parameters
    ----------
    beta : array
        vector of coefficients
    lambdas : array
        vector of regularization weights

    Returns
    -------
    array
        the result of the proximal operator
    """
    #beta_sign = np.sign(beta)
    #beta = np.abs(beta)
    ord = np.flip(np.argsort(beta))
    beta = beta[ord]

    p = len(beta)

    s = np.empty(p, np.float64)
    w = np.empty(p, np.float64)
    idx_i = np.empty(p, np.int64)
    idx_j = np.empty(p, np.int64)

    k = 0

    for i in range(p):
        idx_i[k] = i
        idx_j[k] = i
        s[k] = beta[i] - lambdas[i]
        w[k] = s[k]

        while (k > 0) and (w[k - 1] <= w[k]):
            k = k - 1
            idx_j[k] = i
            s[k] += s[k + 1]
            w[k] = s[k] / (i - idx_i[k] + 1)

        k = k + 1

    for j in range(k):
        d=w[j]
        #d = max(w[j], 0.0)
        for i in range(idx_i[j], idx_j[j] + 1):
            beta[i] = d

    beta[ord] = beta.copy()
    #beta *= beta_sign

    return beta


'''
beta = np.array([2, -3, -1, 5])
print(beta)
beta_sign = np.sign(beta)
print(beta_sign)
#beta = np.abs(beta)
ord = np.flip(np.argsort(beta))
print(ord)
beta = beta[ord]
print(beta)
'''

def pgd_slope(
    X,
    y,
    lambdas,
    fit_intercept=True,
    gap_tol=1e-6,
    max_it=10_000,
    verbose=False,
):
    n, p = X.shape

    residual = y.copy()
    beta = np.zeros(p)
    intercept = 0.0

    primals, duals, gaps = [], [], []

    if sparse.issparse(X):
        L = sparse.linalg.svds(X, k=1)[1][0] ** 2 / n
    else:
        L = norm(X, ord=2) ** 2 / n

    for it in range(max_it):
        beta = prox_slope(beta + (X.T @ residual) / (L * n), lambdas / L)

        if fit_intercept:
            intercept = np.mean(residual)

        residual = y - X @ beta - intercept

        theta = residual / n
        theta /= max(1, dual_norm_slope(X, theta, lambdas))

        primal = 0.5 * norm(residual) ** 2 / n + sl1_norm(beta, lambdas)
        dual = 0.5 * (norm(y) ** 2 - norm(y - theta * n) ** 2) / n
        gap = primal - dual

        primals.append(primal)
        duals.append(primal)
        gaps.append(gap)

        if verbose:
            print(f"epoch: {it + 1}, loss: {primal}, gap: {gap:.2e}")

        if gap < gap_tol:
            break

    #return dict(beta=beta, intercept=intercept, primals=primals, duals=duals, gaps=gaps)
    return dict(beta=beta, intercept=intercept)





X = np.array([[1.0, 0.0], [0.0, 1.0]])

y_1 = np.array([5.0, 4.0])
lambdas_1 = np.array([3.0, 1.0])

y_2 = np.array([10.0, -5.0])
lambdas_2 = np.array([25.0, 5.0])

y_3 = np.array([60.0, 50.0, 10.0, -5.0])
lambdas_3 = np.array([65.0, 40.0, 25.0, 5.0])

y_3s = np.array([-5.0, 60.0, 50.0, 10.0])
lambdas_3s = np.array([65.0, 40.0, 25.0, 5.0])

#y_4 = np.array([60.0, 50.0, 10.0, -5.0, -7.0])
#lambdas_4 = np.array([65.0, 40.0, 25.0, 5.0, 1.0])

#print("prox_slope:", prox_slope(y_2,lambdas_2))
print("prox_slope_isotonic:", prox_slope_isotonic(y_3s, lambdas_3s))















