import numpy as np
from numpy.linalg import norm
from scipy import sparse

from src.slope.utils import dual_norm_slope, prox_slope, sl1_norm

'''
#Some functions in this file are using the code from https://github.com/jolars/pyslope.git
MIT License

Copyright (c) 2022 Johan Larsson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

def pgd_slope(X, y, lambdas, fit_intercept=True, gap_tol=1e-6, max_it=10_000, verbose=False):
    """
    Implements proximal gradient descent to minimize the SLOPE problem  0.5/n*||y-Xb||^2+ <lambda,|b|_(i)>
    """
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

def prox_slope_isotonic(y, lambdas):
    """Compute the prox operator for the generalized slope norm, with b^0 = positive constant: prox_{t*J_b^0,lambda}(y).

    Parameters
    ----------
    y : array
        vector of coefficients
    lambdas : array
        vector of regularization weights

    Returns
    -------
    array
        the result of the proximal operator
    """
    lambdas = np.flip(np.sort(lambdas))
    ord = np.flip(np.argsort(y))
    y = y[ord]

    p = len(y)

    s = np.empty(p, np.float64)
    w = np.empty(p, np.float64)
    idx_i = np.empty(p, np.int64)
    idx_j = np.empty(p, np.int64)

    k = 0

    for i in range(p):
        idx_i[k] = i
        idx_j[k] = i
        s[k] = y[i] - lambdas[i]
        w[k] = s[k]

        while (k > 0) and (w[k - 1] <= w[k]):
            k = k - 1
            idx_j[k] = i
            s[k] += s[k + 1]
            w[k] = s[k] / (i - idx_i[k] + 1)

        k = k + 1

    for j in range(k):
        d = w[j]
        # d = max(w[j], 0.0)
        for i in range(idx_i[j], idx_j[j] + 1):
            y[i] = d

    y[ord] = y.copy()
    # beta *= beta_sign

    return y

def prox_slope_on_b_0_single_cluster(b_0, y, lambdas):
    """Compute the prox operator for the generalized slope norm, with b^0 of one cluster like [2,2,-2,-2]: prox_{t*J_b^0,lambda}(y).

    Parameters
    ----------
    y : array
        vector of coefficients
    lambdas : array
        vector of regularization weights

    Returns
    -------
    array
        the result of the proximal operator
    """

    sign_b_0 = np.sign(b_0)
    S_b_0 = np.diag(sign_b_0)
    solution = S_b_0 @ prox_slope_isotonic(S_b_0 @ y, lambdas)
    return solution

def lambda_partition_by_b_0(b_0, lambdas):
    '''
    Associates lambdas to the clusters of b_0 starting from the 0 cluster.
    ----------
    b_0 : array
        pattern vector of the signal b_0
    lambdas : array
        penalty vector
    Returns
    -------
    array
        partition of lambda into clusters of b_0 starting from the 0 cluster.
    '''
    lambda_partition = []
    lambdas = np.sort(lambdas)
    b_0 = np.abs(b_0)
    nr_clusters = max(np.abs(b_0)) #len(set(np.abs(b_0)))
    index_counter = 0
    for k in range(nr_clusters + 1):
        size_of_cluster_k = np.count_nonzero(b_0 == k)
        cluster_k = []
        for i in range(size_of_cluster_k):
            cluster_k.append(lambdas[index_counter])
            index_counter = index_counter + 1
        lambda_partition.append(cluster_k)

    return lambda_partition

def y_partition_by_b_0(b_0, y):
    '''
    Associates y to the clusters of b_0 starting from the 0 cluster.
    ----------
    b_0 : array
        pattern vector of the signal b_0
    y : array
        input prox vector
    Returns
    -------
    array
        partition of y into clusters of b_0 starting from the 0 cluster.
    '''
    b_0 = np.abs(b_0)
    cluster_boxes = []
    nr_clusters = max(np.abs(b_0)) #len(set(np.abs(b_0)))

    for m in range(nr_clusters+1):
        cluster_boxes.append([])

    for i in range(len(b_0)):
        k = b_0[i]
        cluster_boxes[k].append(y[i])
    return cluster_boxes

def u_reconstruction(b_0, u_partition):
    '''
    Reconstructs y from its u_partition.
    ----------
    b_0 : array
        pattern vector of the signal b_0
    u_partition : array
        partition of y into clusters of b_0 starting from the 0 cluster.
    Returns
    -------
    array
        original vector y reconstruction
    '''
    reconstruction = np.zeros(len(b_0))
    b_0 = np.abs(b_0)
    nr_clusters = max(np.abs(b_0)) #len(set(np.abs(b_0))) #max(b_0)
    for k in range(nr_clusters +1):
        size_of_cluster_k = np.count_nonzero(b_0 == k)
        for i in range(size_of_cluster_k):
            value = u_partition[k][i]
            position = np.where(b_0 == k)[0][i]
            reconstruction[position] = value
    return reconstruction



def prox_slope_new(y, lambdas):
    """Compute the sorted L1 proximal operator.

    Parameters
    ----------
    y : np.array
        vector of coefficients
    lambdas : np.array
        vector of regularization weights

    Returns
    -------
    array
        the result of the proximal operator
    """
    lambdas = np.flip(np.sort(lambdas))
    y_sign = np.sign(y)
    y = np.abs(y)
    ord = np.flip(np.argsort(y))
    y = y[ord]

    p = len(y)

    s = np.empty(p, np.float64)
    w = np.empty(p, np.float64)
    idx_i = np.empty(p, np.int64)
    idx_j = np.empty(p, np.int64)

    k = 0

    for i in range(p):
        idx_i[k] = i
        idx_j[k] = i
        s[k] = y[i] - lambdas[i]
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
            y[i] = d

    y[ord] = y.copy()
    y *= y_sign

    return y

def prox_slope_b_0(b_0, y, lambdas):
    """Compute the prox operator for the SLOPE directional derivative J'_{lambda}, i.e; prox_{J'_{lambda}(b^0; * )}(y) = argmin_u (1/2)||u-y||^2+J'_{lambda}(b^0; u)

       Parameters
       ----------
       b_0: np.array
           pattern vector of the true signal
       y : np.array
           input vector of coefficients
       lambdas : np.array
           vector of regularization weights

       Returns
       -------
       array
           the result of the proximal operator
       """
    y = [np.float64(i) for i in y]
    lambdas = [np.float64(i) for i in lambdas]
    # b_0 = [0, 2, 0, 2, -2, -2, 1, 1]
    # y = [5.0, 60.0, 4.0, 50.0, 10.0, -5.0, 12.0, 17.0]
    # lambdas = [65.0, 42.0, 40.0, 20.0, 18.0, 15.0, 3.0, 1.0]
    nr_clusters = max(np.abs(b_0))  # 2
    y_partition = y_partition_by_b_0(b_0, y)  # [[5.0, 4.0], [12.0, 17.0], [60.0, 50.0, 10.0, -5.0]]
    lambda_partition = lambda_partition_by_b_0(b_0, lambdas)  # [[1.0, 3.0], [15.0, 17.0], [20.0, 40.0, 42.0, 65.0]]
    b_0_partition = y_partition_by_b_0(b_0, b_0) #[[0,0],[1,1],[2,2,-2,-2]]
    prox_0_cluster = prox_slope_new(y=y_partition[0], lambdas=lambda_partition[0])
    prox_k_clusters = [prox_0_cluster]  # [[2.5,2.5]] then [[2.5, 2.5], [-3.0, -1.]] then [[2.5, 2.5], [-3.0, -1.], [1.5, 1.5, 32.5, 32.5]]

    for k in range(nr_clusters):
        b_0_kth = np.array(b_0_partition[k + 1])
        y_kth = np.array(y_partition[k + 1])
        lambda_kth = np.array(lambda_partition[k + 1])
        prox_kth_cluster = prox_slope_on_b_0_single_cluster(b_0=b_0_kth, y=y_kth, lambdas=lambda_kth)
        prox_k_clusters.append(prox_kth_cluster)

    solution = u_reconstruction(b_0, prox_k_clusters)  # [2.5, 1.5, 2.5, 1.5, 32.5, 32.5, -3.0, -1.0]
    return solution

#print(prox_slope_b_0([0, 2, 0, 2, -2, -2, 1, 1], [5.0, 60.0, 4.0, 50.0, 10.0, -5.0, 12.0, 17.0], [65.0, 42.0, 40.0, 20.0, 18.0, 15.0, 3.0, 1.0]))

def pgd_slope_b_0_ISTA(C, W, b_0, lambdas, t, n):
    """Minimizes: 1/2 u^T*C*u-u^T*W+J'_{lambda}(b^0; u),
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
    u_0 = np.zeros(len(b_0))
    prox_step = u_0
    stepsize_t = np.float64(t)
    for i in range(n):
        grad_step = prox_step - stepsize_t * (C @ prox_step - W)
        prox_step = prox_slope_b_0(b_0, grad_step, lambdas * stepsize_t)
    return(prox_step)

def pgd_slope_b_0_FISTA(C, W, b_0, lambdas, t, n):
    """Minimizes: 1/2 u^T*C*u-u^T*W+J'_{lambda}(b^0; u),
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
    u_0 = np.zeros(len(b_0))
    u_kmin2 = u_0
    u_kmin1 = u_0
    v = u_0
    stepsize_t = np.float64(t)
    k=1
    for k in range(n):
        v = u_kmin1 + ((k-2)/(k+1))*(u_kmin1-u_kmin2)
        grad_step = v - stepsize_t * (C @ v - W)
        u_k = prox_slope_b_0(b_0, grad_step, stepsize_t * lambdas)
        u_kmin2 = u_kmin1
        u_kmin1 = u_k
    return (u_k)

def pattern(u):
    """
    Calculate the SLOPE pattern of a vector: rank(abs(u_i)) * sgn(u_i).

    Parameters:
        u (np.ndarray): Input vector.

    Returns:
        np.ndarray: Integer-valued vector.
    """
    # Calculate the absolute values of the elements in u
    abs_u = np.abs(u)

    # Calculate the unique ranks of the absolute values (ascending order)
    if 0 in abs_u:
        unique_ranks = np.unique(abs_u, return_inverse=True)[1]
    else:
        unique_ranks = np.unique(abs_u, return_inverse=True)[1] + 1

    # Calculate the sign of each element in u
    sign_u = np.sign(u)

    # Calculate the SLOPE pattern
    result = unique_ranks * sign_u

    return result.astype(int)

