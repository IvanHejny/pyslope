
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
    #beta_sign = np.sign(beta)
    #beta = np.abs(beta)
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
        d=w[j]
        #d = max(w[j], 0.0)
        for i in range(idx_i[j], idx_j[j] + 1):
            y[i] = d

    y[ord] = y.copy()
    #beta *= beta_sign

    return y


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


def prox_slope_on_b_0_cluster(b_0, y, lambdas):
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
    y = S_b_0 @ y
    solution = S_b_0 @ prox_slope_isotonic(y, lambdas)
    return solution


X = np.array([[1.0, 0.0], [0.0, 1.0]])

y_1 = np.array([5.0, 4.0])
lambdas_1 = np.array([3.0, 1.0])

y_2 = np.array([10.0, -5.0])
lambdas_2 = np.array([25.0, 5.0])


lambdas_3 = np.array([65.0, 40.0, 25.0, 5.0])
#lambdas_3s = np.array([5.0, 40.0, 65.0, 25.0]) resorting lambdas messes up with the solution, keep the lambdas sorted

y_3 = np.array([60.0, 50.0, 10.0, -5.0])
#y_3s = np.array([-5.0, 60.0, 50.0, 10.0]) #reshuffling y

y_3t = np.array([60.0, 50.0, -10.0, 5.0]) #swapping last two signs in y_3 multiplying y S_b_0 with b_0=[2,2,-2,-2]

b_0 = np.array([2, 2, -2, -2])

#y_4 = np.array([60.0, 50.0, 10.0, -5.0, -7.0])
#lambdas_4 = np.array([65.0, 40.0, 25.0, 5.0, 1.0])

#print("prox_slope:", prox_slope(y_2,lambdas_2))

print("prox_slope_isotonic:", prox_slope_isotonic(y_3t, lambdas_3))
#print("prox_slope_isotonic:", prox_slope_isotonic(y_3t, lambdas_3s))

print("prox_slope_on_b_0_cluster:", prox_slope_on_b_0_cluster(b_0, y_3, lambdas_3))


"""
b_0=np.array([2, 2, -2, -2])
sign_b_0=np.sign(b_0)
print(b_0)
print(sign_b_0)
identity=np.identity(4)
print(np.diag(sign_b_0))
#print(b_0 @ np.diag(sign_b_0))
"""


b_0_test1 = [0, 0, 1, 1, 2, -2, 2]
lambda_test1 = [7, 6, 5, 4, 3, 2, 1]
#nr_clusters = max(np.abs(b_0_test1))
#print(np.arange(nr_clusters+1))
lambda_test1 = np.flip(lambda_test1)
b_0_test1 = np.abs(b_0_test1)
print(b_0_test1)

'''
k = 0
lambda_partition = []
one_cluster = []
for i in range(len(b_0_test1)):
    if i == len(b_0_test1)-1:
        one_cluster.append(lambda_test1[i])
        lambda_partition.append(one_cluster)
        break
    elif b_0_test1[i] == k:
        one_cluster.append(lambda_test1[i])
    else:
        lambda_partition.append(one_cluster)
        one_cluster = []
        k = k+1
        one_cluster.append(lambda_test1[i])

print(lambda_partition)
'''
'''
cluster_boxes = []
for m in range(nr_clusters+1):
    cluster_boxes.append([])
print(cluster_boxes)

b_0_test2 = [0, 2, 0, -2, 2, 1, 1]
y_test2 = [80, 21, -5, 7, 12, 18, 33]

for i in range(len(b_0_test2)):
    k = b_0_test2[i]
    cluster_boxes[k].append(y_test2[i])
print(cluster_boxes)
'''

b_0_test2 = [0, 2, 0, -2, 2, 1, 1]
y_test2 = [80, 21, -5, 7, 12, 18, 33]
b_0_test2 = np.abs(b_0_test2)
#print(b_0_test2.count(-2))
#print(np.count_nonzero(b_0_test2 == 2))
def lambda_partition_by_b_0(b_0, lambdas):
    lambda_partition = []
    lambdas = np.sort(lambdas)
    b_0 = np.abs(b_0)
    nr_clusters = max(np.abs(b_0))
    index_counter = 0
    for k in range(nr_clusters+1):
        size_of_cluster_k = np.count_nonzero(b_0 == k)
        cluster_k = []
        for i in range(size_of_cluster_k):
            cluster_k.append(lambdas[index_counter])
            index_counter = index_counter + 1
        lambda_partition.append(cluster_k)

    return lambda_partition


def y_partition_by_b_0(b_0, y):
    b_0 = np.abs(b_0)
    cluster_boxes = []
    nr_clusters = max(np.abs(b_0))

    for m in range(nr_clusters + 1):
        cluster_boxes.append([])

    for i in range(len(b_0)):
        k = b_0[i]
        cluster_boxes[k].append(y[i])
    return cluster_boxes

#print(y_partition_by_b_0(b_0_test2, y_test2))

#print(y_partition_by_b_0(b_0_test2, lambda_test1))
print(lambda_partition_by_b_0(b_0_test2, lambda_test1))


