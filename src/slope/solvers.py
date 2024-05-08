import numpy as np
from numpy.linalg import norm
from scipy import sparse, stats

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
    y = [np.float32(i) for i in np.real(y)]
    lambdas = [np.float32(i) for i in np.real(lambdas)]
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

def pgd_slope_b_0_FISTA(C, W, b_0, lambdas, n=None, t=None, tol=1e-4, max_iter=2000):
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
    b_0 = pattern(b_0)
    u_0 = np.zeros(len(b_0))
    u_kmin2 = u_0
    u_kmin1 = u_0
    v = u_0
    if n is None:
        max_iter = max_iter
    else:
        max_iter = n
    if t == None:
        t = 1 / np.max(np.linalg.eigvals(C))  # default stepsize = 1/max(eigenvalues of C) to guarantee O(1/n^2) convergence
        t = np.float32(np.real(t))

    for k in range(1, max_iter):
        v = u_kmin1 + ((k-2)/(k+1))*(u_kmin1-u_kmin2)
        grad_step = v - t * (C @ v - W)
        u_k = prox_slope_b_0(b_0, grad_step, t * lambdas)
        u_kmin2 = u_kmin1
        u_kmin1 = u_k
        if n is None:
            p = len(b_0)
            norm_diff = np.linalg.norm(u_kmin1 - u_kmin2) / np.sqrt(p)
            if norm_diff < tol: #and k > 4:
                #print('final_iter:', k)  # uncomment line for final iterate
                break
            elif k == max_iter - 1:
                print('Warning: Maximum number of iterations', max_iter, ' reached. Convergence of FISTA might be slow or tol too low.')

    #print('stpsize:', t)  # uncomment line for stepsize
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

def pattern_matrix(vector):
  """Creates a (SLOPE) pattern matrix representing non-zero clusters in the input vector.

  Args:
      vector: A NumPy vector.

  Returns:
      A NumPy matrix where each row represents a basis vector corresponding to a non-zero cluster.
  """
  sign = np.sign(vector)
  sign_matrix = np.diag(sign)
  vector = np.abs(vector)
  unique_values, counts = np.unique(vector[np.nonzero(vector)], return_counts=True)  # Get unique non-zero values and counts
  pattern_matrix = np.zeros((len(unique_values), len(vector)))  # Initialize pattern matrix

  for i, (value, count) in enumerate(zip(unique_values, counts)):
    pattern_matrix[i, vector == value] = 1  # Set 1s for matching values in each cluster
    # Limit the number of 1s to the count of the unique value
    # pattern_matrix[i, pattern_matrix[i] == 1] = np.arange(1, count +1)  # Set unique markers for duplicates

  return  (pattern_matrix @ sign_matrix).T  # .astype(int) convert to integer type

#print('pattern matrix:\n', pattern_matrix(np.array([0, 2, 0, -2, 2, 1, 1])))

def pattern_matrix_Lasso(vector):
  """Creates a Lasso pattern matrix for the input vector.

  Args:
      vector: A NumPy vector.

  Returns:
      A NumPy matrix Lasso pattern matrix.
  """
  p=len(vector)
  m=np.count_nonzero(vector)
  pattern_matrix = np.zeros((p, m))  # Initialize pattern matrix
  sign = np.sign(vector)
  sign_matrix = np.diag(sign)
  for i in range(m):
      row_index = np.nonzero(vector)[0][i]
      pattern_matrix[row_index, i] = 1  # Set 1s for matching values in each cluste
  return  sign_matrix @ pattern_matrix   # .astype(int) convert to integer type
#print('pattern_matrix_Lasso:\n', pattern_matrix_Lasso(np.array([0, 2, 0, -2, 2, 1, 1])))
#print(np.nonzero(np.array([0, 2, 0, -2, 2, 1, 1])))

def consecutive_cluster_sizes(b0):
  """
  Counts the lengths of consecutive clusters in a NumPy array.

  Args:
      b0: A NumPy array representing the cluster labels.

  Returns:
      A NumPy array containing the lengths of consecutive clusters.
  """

  # Find differences between consecutive elements
  diffs = np.diff(b0)
  diffs = np.concatenate(([1], diffs))

  # Handle potential single-element cluster at the beginning
  #if b0[0] != b0[1]:  # Check difference between first two elements
  #  cluster_starts = np.array([0])
  #else:
  cluster_starts = np.where(diffs != 0)[0] + 1  # + 1 to exclude first element

  # Cluster ends based on non-zero differences (or array length)
  cluster_ends = np.append(cluster_starts[1:], len(b0)+1)

  # Cluster lengths
  cluster_lengths = cluster_ends - cluster_starts

  return cluster_lengths
#print('cluster_lengths:', consecutive_cluster_sizes(np.array([1, 1, 1, 0, 0, 1, 1, 3, 3, 3, 3])))

def pattern_matrix_FLasso(vector):
  """Creates a Fused Lasso pattern matrix for the input vector.

  Args:
      vector: A NumPy vector.

  Returns:
      A NumPy matrix Lasso pattern matrix.
  """
  sign = np.sign(vector)
  sign_matrix = np.diag(sign)

  p = len(vector)
  cluster_sizes = consecutive_cluster_sizes(vector)
  m = len(cluster_sizes)
  cluster_matrix = np.zeros((p, m))  # Initialize pattern matrix

  counter = 0
  for i in range(m):
      for j in range(cluster_sizes[i]):
          cluster_matrix[counter, i] = 1
          counter = counter + 1
  pattern_matrix_Flasso = sign_matrix @ cluster_matrix
  zero_cols = np.all(pattern_matrix_Flasso == 0, axis=0)
  return  pattern_matrix_Flasso[:, ~ zero_cols] # removes zero columns

'''
print('pattern_matrix_Lasso:\n', pattern_matrix_Lasso(np.array([1,1,0,2,0,2])))
print('pattern_matrix_FusedLasso:\n', pattern_matrix_FLasso(np.array([1,1,0,2,0,2])))
print('pattern_matrix_SLOPE:\n', pattern_matrix(np.array([1,1,0,2,0,2])))

print('pattern_matrix_Lasso:\n', pattern_matrix_Lasso(np.array([0, 2, 0, -2, 2, 1, 1])))
print('pattern_matrix_FusedLasso:\n', pattern_matrix_FLasso(np.array([0, 2, 0, -2, 2, 1, 1])))
print('pattern_matrix_SLOPE:\n', pattern_matrix(np.array([0, 2, 0, -2, 2, 1, 1])))

print('pattern_matrix_Lasso:\n', pattern_matrix_Lasso(np.array([1, 0, 1, 0])))
print('pattern_matrix_FusedLasso:\n', pattern_matrix_FLasso(np.array([1, 0, 1, 0])))
print('pattern_matrix_SLOPE:\n', pattern_matrix(np.array([1, 0, 1, 0])))
'''
def proj_onto_pattern_space(vector):
    """Projection matrix onto the (SLOPE) pattern space of some vector

    Args:
        vector: A NumPy vector.

    Returns:
        A NumPy projection matrix
    """
    U = pattern_matrix(vector)
    return U @ np.linalg.inv(U.T @ U) @ U.T


def lin_lambdas(p):
    """
    Generate linear penalty sequence.

    This function generates a linear penalty sequence for regularization
    purposes, normalized such that the average penalty is 1.

    Parameters:
        p (int): The length of the penalty sequence.

    Returns:
        array_like: Array of linearly decreasing penalty values.
    """
    return np.flip(np.arange(1, p + 1)) / ((p + 1) / 2)


def bh_lambdas(p, q=0.05):
    """
    Generate Benjamini-Hochberg (BH) penalty sequence.

    This function generates a penalty sequence following the Benjamini-Hochberg
    (BH) method, normalized such that the average penalty is 1.

    Parameters:
        p (int): The length of the penalty sequence.
        q (float, optional): The false discovery rate (FDR) control level.

    Returns:
        array_like: Array of penalty values following the BH method.
    """
    randnorm = stats.norm(loc=0, scale=1)
    lambdas = randnorm.ppf(1 - np.arange(1, p + 1) * q / (2 * p))
    return lambdas / (np.sum(lambdas) / p)

#print('bh_lambdas', bh_lambdas(6,0.1))


def comp_sym_corr(rho, p):
    return (1-rho)*np.identity(p)+rho*np.ones((p, p))
def band_mat(n, diag_val=1.0, first_off_diag=0.9, second_off_diag=0.8, third_off_diag=0.7):
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
