from src.slope.solvers import*
from admm_glasso import*

#ADMM_GLASSO
# penalty matrices in generalized lasso
def AFL(p, a=1): #Fused Lasso + Lasso, tuned by a
    A = np.zeros((2 * p - 1, p))
    for i in range(p - 1):
        A[i][i] = 1
        A[i][i + 1] = -1  # clustering penalty
    for i in range(p - 1, 2 * p - 1):
        A[i][i - (p - 1)] = a  # sparsity penalty
    return A

def AFLmon(p, b=0.1): # Fused Lasso + Monotone Lasso penalty
    A = np.zeros((2 * p - 1, p))
    for i in range(p - 1):
        A[i][i] = 1
        A[i][i + 1] = -1  # clustering penalty
    for i in range(p - 1, 2 * p - 1):
        A[i][i - (p - 1)] = 1+b*(i-(p-1)+1)  # sparsity penalty
    return A
print('AFLmon:\n', AFLmon(7, 0.1))

'''
p = 2
AFL2 = np.zeros((2 * p - 1, p))
for i in range(p - 1):
    AFL2[i][i] = 1
    AFL2[i][i + 1] = -1 #clustering penalty
for i in range(p-1, 2*p -1):
    AFL2[i][i - (p - 1)] = 1.7 #sparsity penalty
#print('AFL\n:', AFL2)

p = 9
A = np.zeros((p, p))
for i in range(p - 1):
    A[i][i] = 1
    A[i][i + 1] = -1 #enables clustering
A[p - 1][0] = 1 #enables sparsity
#w = np.array([1, 1.1, 0.9, 2, 1, -2, 0, 1, 1]) #just a fixed arbitrary vector


AFL9 = np.zeros((2 * p - 1, p))
for i in range(p - 1):
    AFL9[i][i] = 1
    AFL9[i][i + 1] = -1 #clustering penalty
for i in range(p-1, 2*p -1):
    AFL9[i][i - (p - 1)] = 1 #sparsity penalty
#print('AFL\n:', AFL)

p=7
Amon = np.zeros((2 * p - 1, p))
for i in range(p - 1):
    Amon[i][i] = 1
    Amon[i][i + 1] = -1  # clustering penalty
for i in range(p - 1, 2 * p - 1):
    Amon[i][i - (p - 1)] = 1+0.1*(i-(p-1)+1)  # sparsity penalty
print('Amon:\n', Amon)
'''


# signal vector betas
beta02 = np.array([0, 1])  # recovery for tuned AFL a>>1
beta09 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])  # recovery seemingly impossible irrespective of tuning AFL
# the error in pattern always occurs in the middle cluster next to zero cluster, never in the last cluster
beta03 = np.array([0,1,1])  # recovery for tuned AFL a>>1
beta030 = np.array([0, 0, 1])  # recovery for tuned AFL a>>1
beta04 = np.array([0, 0, 1, 1])  # recovery for tuned AFL a>>1
beta043 = np.array([0, 1, 1, 2])  # fails to recover the middle cluster
beta07 = np.array([0, 1, 1, 2, 2, 3, 3])  # fails to recover the two middle clusters, but recovers the 0 and the last

# printing different quantities appearing in the ADMM algorithm
my_beta0 = beta043
my_A = AFLmon(4, 0.1) # AFL(4,1.3) # AFL(7,1)

#my_beta0 = beta0
#my_A = AFL
#my_A = np.identity(9) #Lasso penalty
#my_A = A #Fused Lasso penalty
print('my_beta0:', my_beta0)
print('my_A:\n', my_A)
E_1 = np.diag(np.sign(my_A @ my_beta0))  # sgn(A*beta0) mxm matrix
print('E_1:\n', E_1)
ind = np.where(np.diag(E_1) == 0)[0]
print('ind:', ind)
F_0 = my_A[ind, :]
print('F_0:\n', F_0)
print('colsum:\n', E_1 @ my_A, np.ones(my_A.shape[0]).T @ E_1 @ my_A)

# covariance matrix
alpha = 0.7
C2 = np.array([[1, alpha], [alpha, 1]])
C9_comp = alpha * np.ones(9) + (1-alpha) * np.identity(9)  # compound symmetric covariance matrix

c_block = (1 - alpha) * np.identity(3) + alpha * np.ones((3, 3))
C9_block = np.block([[c_block, np.zeros((3, 3)), np.zeros((3, 3))], # block diagonal with 3 compound blocks 3x3
                     [np.zeros((3,3)), c_block, np.zeros((3, 3))],
                     [np.zeros((3,3)), np.zeros((3,3)), c_block]])

def glasso_sampler(C, A, beta0, lambdas, iter=100, n=20): #sampling asymptotic error from the glasso
    p = len(beta0)
    for i in range(n):
        W = np.random.multivariate_normal(np.zeros(p), C)
        glasso_sample = admm_glasso(C, A, W, beta0, lambdas, iter)
        #print('glasso_sol:', np.round(glasso_sample, 3))
        print('glasso_sol:', pattern(np.round(glasso_sample, 3)))

# print(glasso_sampler(C2, AFL(2, a=1.5), beta02, 40))
# print(glasso_sampler(np.identity(3), AFL(3, a=1.5), beta030, 40))
# print(glasso_sampler(C9_comp, AFL(9), beta09, 40))
# print(glasso_sampler(np.identity(4), AFL(4, a=1.2), beta043, 40))
# print(glasso_sampler(np.identity(7), AFL(7, a=1.2), beta07, 40))

#print(glasso_sampler(np.identity(7), AFLmon(7,0.1), beta07, 40))
print(glasso_sampler(C9_block, AFLmon(9, 0.1), beta09, 40))

#for i in range(20):
#    W = np.random.multivariate_normal(np.zeros(p), C)
#    glasso_AFLsmall = admm_glasso(C, AFL2, W, beta02, 40, iter=100)
#    print('admm_solution:', np.round(glasso_AFLsmall, 3))




#print('admm_solution:\n', np.round(admm_glasso(C, AL, w, beta0, lambdas), 4))
'''
A0 = np.zeros((p-1, p))
for i in range(p - 1):
    A0[i][i] = 1
    A0[i][i + 1] = -1
print(A0)
A2 = np.zeros((p+1, p))
for i in range(1, p):
    A2[i][i-1] = 1
    A2[i][i] = -1
A2[0][0] = 1
A2[p][p-1] = 1
print('A0:\n', A0)
#print(A0.shape[0],A0.shape[1])
print('A2:\n', A2)
print('admm_solutionA0:\n', admm_glasso(C, A0, w, beta0, lambdas))
print('admm_solutionA2:\n', admm_glasso(C, A2, w, beta0, lambdas))

AFL3 = np.zeros((p+2, p)) #Fused Lasso + Lasso on the first three
for i in range(p - 1):
    AFL3[i][i] = 1
    AFL3[i][i + 1] = -1 #clustering penalty
for i in range(p-1, p+2):
    AFL3[i][i - (p-1)] = 0.8 #sparsity penalty
print('AFL3\n:', AFL3)

AFL9tuned = np.zeros((2 * p - 1, p)) #does not recover the pattern for beta0 = [0, 0, 0, 1, 1, 1, 2, 2, 2] despite the tuning
for i in range(p - 1):
    AFL9tuned[i][i] = 1
    AFL9tuned[i][i + 1] = -1 #clustering penalty
for i in range(p-1, p+2):
    AFL9tuned[i][i - (p-1)] = 5 #sparsity penalty strong for first 3
for i in range(p+2, 2*p -1):
    AFL9tuned[i][i - (p-1)] = 0.1 #sparsity penalty weak for last 6
print('AFLtuned\n:', AFL9tuned)
'''


# compound symmetric covariance matrix, simulations with Lasso/Fused Lasso Penalty
#alpha = 0.1
#C = alpha * np.ones(9) + (1-alpha) * np.identity(9)
#for i in range(20):
#    W = np.random.multivariate_normal(np.zeros(p), C)
#    print('admm_AFL:', np.round(admm_glasso(C, AFL, W, beta0, lambdas), 3))

# block diagonal with 3 compound blocks 3x3, simulations with Lasso/Fused Lasso Penalty
'''
alpha = 0.5
compound_block = (1-alpha) * np.identity(3) + alpha * np.ones((3, 3))
block_diag_matrix9 = np.block([[compound_block, np.zeros((3,3)), np.zeros((3,3))],
                              [np.zeros((3,3)), compound_block, np.zeros((3,3))],
                               [np.zeros((3,3)), np.zeros((3,3)), compound_block]])
print('block_diag_matrix9:\n', block_diag_matrix9)
for i in range(20):
    W = np.random.multivariate_normal(np.zeros(p), block_diag_matrix9)
    admm_AFL_comp = admm_glasso(block_diag_matrix9, AFL, W, beta0, 40, iter=100)
    pgd_slope_comp = pgd_slope_b_0_FISTA(C=block_diag_matrix9, W=W, b_0=beta0, lambdas=40 * np.array([1.4,1.3,1.2,1.1,1.0,0.9,0.8,0.7,0.6]), t=0.35, n=150)
    print('admm_AFL_comp:', np.round(admm_AFL_comp, 3))
    #print('pgd_slope_comp', np.round(pgd_slope_comp, 3))
'''




#testing if admm lasso = pgd lasso, sanity check! Works fine
'''
for i in range(20):
    W = np.random.multivariate_normal(np.zeros(p), block_diag_matrix9)
    admm_ALasso_comp = admm_glasso(block_diag_matrix9, np.identity(p), W, beta0, 40, iter=100)
    pgd_slope_comp = pgd_slope_b_0_FISTA(C=block_diag_matrix9, W=W, b_0=beta0, lambdas=40 * np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]), t=0.35, n=150)

    #print('admm_ALasso_comp:', np.round(admm_ALasso_comp,3))
    #print('pgd_slope_comp', np.round(pgd_slope_comp, 3))
    print('difference:', np.round(admm_ALasso_comp - pgd_slope_comp,4))# correct, the algorithms admm and pgd converge return the same
    # prox step:

C=np.array([[1, 0], [ 0, 1]])
print(C)
print(C@np.array([2, 3]))
W = np.array([5,4])
lambdas = np.array([1,3])
b_00 = np.array([0,0])
u_0 = np.array([0,0])
stepsize_t = 0.4
prox_step = prox_slope_b_0(b_00, u_0-stepsize_t*(C@u_0-W), lambdas*stepsize_t)
print(prox_step)
prox_step = prox_slope_b_0(b_00, prox_step-stepsize_t*(C@prox_step-W), lambdas*stepsize_t)
print(prox_step)
'''

    #ISTA/FISTA
'''
C = np.array([[1, 0.5], [0.5, 1]])
W = np.array([5.0, 4.0])
lambdas = np.array([0.9, 0.2])
b_00 = np.array([1, 1])
stepsize_t = 0.35 # to guarantee convergence take stepsize < 1/max eigenvalue of C (max eval of C is the Lipschitz constant of grad(1/2 uCu - uW)=(Cu-W))


print("prox_slope_b_0:", prox_slope_b_0(b_00, W, lambdas)) # first simple verification of ISTA, FISTA
print("pdg_slope_b_0_ISTA:", pgd_slope_b_0_ISTA(C, W, b_00, lambdas, stepsize_t, 20))
print("pdg_slope_b_0_FISTA:", pgd_slope_b_0_FISTA(C, W, b_00, lambdas, stepsize_t, 20))
#print("pdg_slope_b_0_ISTA:", pgd_slope_b_0_ISTA(C, [5,4], b_00, [1,1], stepsize_t, 20)) #typeproblem
'''

#pattern attainability:
'''
for i in range(40):
    one_solution = pgd_slope_b_0_ISTA(C = C, W = np.random.multivariate_normal([0, 0], [[1.7, 0], [0, 1.9]]), b_0 = np.array([0, 0]), lambdas = lambdas, t = stepsize_t, n= 20)
    print("pdg_slope_b_0_ISTA: [0,0]" , i+1, one_solution) #b_0 = [0,0] all patterns attainable

for i in range(40):
    one_solution = pgd_slope_b_0_ISTA(C = C, W = np.random.multivariate_normal([0, 0], [[1.7, 0], [0, 1.9]]), b_0 = np.array([1, 1]), lambdas = lambdas, t = stepsize_t, n = 20)
    print("pdg_slope_b_0_ISTA: [1, 1]", i+1, one_solution) #b_0 = [1,1] all positive patterns attainable

for i in range(40):
    one_solution = pgd_slope_b_0_ISTA(C = C, W = np.random.multivariate_normal([0, 0], [[1.7, 0], [0, 1.9]]), b_0 = np.array([0, 1]), lambdas = lambdas, t = stepsize_t, n = 20)
    print("pdg_slope_b_0_ISTA: [0, 1]", i+1, one_solution) #b_0 = [0,1] all patterns with positive second entry attainable
'''

#MAIN TEST
#prox_slope, ISTA, FISTA
'''
b_0_test0 = np.array([0, 0, 0, 0])
b_0_test1 = np.array([1, 1, -1, -1])
b_0_test01 = np.array([0, 0, 1, 1])
y_test1 = np.array([60.0, 50.0, 10.0, -5.0])

lambdas_test1 = np.array([65.0, 42.0, 40.0, 20.0])
lambdas_test01 = np.array([42.0, 42.0, 20.0, 20.0])

print("prox_slope_b_0_1:", prox_slope_b_0(b_0_test0, y_test1, lambdas_test1))
print("prox_slope_b_0_1:", prox_slope_b_0(b_0_test1, y_test1, lambdas_test1))

b_0_test1x = np.array([1, 1, -1, 1])
y_test1x = np.array([60.0, 50.0, -5.0, 10.0])
lambdas_test1x = np.array([65.0, 42.0, 40.0, 40.0])

print("prox_slope_b_0_x:", prox_slope_b_0(b_0_test1x, y_test1x, lambdas_test1x)) #(1.5, 1.5, 35, -30) coincides with theory
print("prox_slope_b_0_x:", prox_slope_b_0(b_0_test1x, [60.0, 50.0, -5.0, 10.0], [65.0, 42.0, 40.0, 40.0])) #correct
print("prox_slope_b_0_x:", prox_slope_b_0(b_0_test1x, [60, 50, -5, 10], [65, 42, 40, 40])) #correct after resolved type issues
print("pdg_slope_b_0_ISTA_x:", pgd_slope_b_0_ISTA( C = np.identity(4), W = y_test1x, b_0 = b_0_test1x, lambdas = lambdas_test1x, t = 0.35, n = 50))
print("pdg_slope_b_0_FISTA_x:", pgd_slope_b_0_FISTA( C = np.identity(4), W = y_test1x, b_0 = b_0_test1x, lambdas = lambdas_test1x, t = 0.35, n = 50))
'''


#Further ISTA, FISTA examples
'''
b_0_test3 = np.array([0, 2, 0, 2, -2, -2, 1, 1])
y_test3 = np.array([5.0, 60.0, 4.0, 50.0, 10.0, -5.0, 12.0, 17.0])

lambda_test3 = [65.0, 42.0, 40.0, 20.0, 18.0, 15.0, 3.0, 1.0]
lambda_test4 = [35.0, 35.0, 5.2, 5.2, 5.2, 5.2, 5.2, 5.2]
lambda_test5 = [35.0, 35.0, 4.8, 4.8, 4.8, 4.8, 4.8, 4.8]

print('prox_slope_b_0_3:', prox_slope_b_0(b_0_test3, y_test3, lambda_test3))
print('prox_slope_b_0:', prox_slope_b_0(b_0_test3, y_test3, np.flip(lambda_test3)))# flipping lambdas has no effect (sanity check)
print('prox_slope_b_0_4:', prox_slope_b_0(b_0_test3, y_test3, lambda_test4))
print('prox_slope_b_0_5:', prox_slope_b_0(b_0_test3, y_test3, lambda_test5))

C_test3 = np.identity(8)
W_test3 = np.array([5.0, -2.0, 3.0, 3.1, -2.5, -5.2, 0.7, -7.0])
print("pdg_slope_b_0_FISTA:", pgd_slope_b_0_FISTA(C_test3, W_test3, b_0_test3, lambda_test3, stepsize_t, 20))

# Compare with
print('b_0:', b_0_test3)
print("zero-cluster:", prox_slope(y=np.array([5.0, 4.0]), lambdas=np.array([3.0, 1.0])),
      "one-cluster:", prox_slope_on_b_0_single_cluster(b_0=np.array([1, 1]), y=np.array([12.0, 17.0]),
                                                       lambdas=np.array(np.array([18.0, 15.0]))),
      "two-cluster:",
      prox_slope_on_b_0_single_cluster(b_0=np.array([2, 2, -2, -2]), y=np.array([60.0, 50.0, 10.0, -5.0]),
                                       lambdas=np.array([65.0, 42.0, 40.0, 20.0])))
'''


#permutation dependencies for prox_slope
'''
lambdas_3s = np.array([5.0, 40.0, 65.0, 25.0]) resorting lambdas no longer messes up with the solution, lambdas are always sorted by the functions

y_test1s = np.array([-5.0, 60.0, 50.0, 10.0])  # reshuffling y
y_test1t = np.array([60.0, 50.0, -10.0, 5.0])  # swapping last two signs in y_3 multiplying y S_b_0 with b_0=[2,2,-2,-2]

y_4 = np.array([60.0, 50.0, 10.0, -5.0, -7.0])
lambdas_4 = np.array([65.0, 40.0, 25.0, 5.0, 1.0])

print("prox_slope:", prox_slope(y_test1, lambdas_test1))
print("prox_slope_b_0, b_0= [0,0,0,0] is:", prox_slope_b_0(b_0_test0, y_test1, lambdas_test1))

print("prox_slope_isotonic:", prox_slope_isotonic(y_test1, lambdas_test1))
print("prox_slope_isotonic:", prox_slope_isotonic(y_test1s, lambdas_test1))#reshuffling y only reshuffles the solution (keeping lambdas fixed)
print("prox_slope_isotonic:", prox_slope_isotonic(y_test1t, lambdas_test1))#swapping last two signs in y changes the solution significantly. Compare with prox_slope_on_b_0_single_cluster for b_0=[2,2,-2,-2]

print("prox_slope_on_b_0_cluster:", prox_slope_on_b_0_single_cluster(b_0_test1, y_test1, lambdas_test1))
'''

#clustering
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
#partition and reconstruction
'''
#b_0_test1 = [0, 0, 1, 1, 2, -2, 2]
lambda_test1 = [7, 6, 5, 4, 3, 2, 1]

b_0_test2 = np.array([0, 2, 0, -2, 2, 1, 1])
y_test2 = np.array([80, 21, -5, 7, 12, 18, 33])
lambda_test2 = [7, 6, 5, 4, 3, 2, 1]

print(b_0_test2)
print(y_test2)

partition_test2 = y_partition_by_b_0(b_0_test2, y_test2)
print('y_partition_by_b_0:', y_partition_by_b_0(b_0_test2, y_test2))

print('u_reconstruction:', u_reconstruction(b_0_test2, partition_test2))

print('lambda:', lambda_test1)
print('lambda_partition_by_b_0:', lambda_partition_by_b_0(b_0_test2, lambda_test1))

print('b_0_test3:', b_0_test3)
print('y_test3', y_test3)

partition_test3 = y_partition_by_b_0(b_0_test3, y_test3)
print('y_partition_by_b_0:', y_partition_by_b_0(b_0_test3, y_test3))

print('u_reconstruction:', u_reconstruction(b_0_test3, partition_test3))

print('lambda:', lambda_test3)
print('lambda_partition_by_b_0:', lambda_partition_by_b_0(b_0_test3, lambda_test3))

y_partition = y_partition_by_b_0(b_0_test3, y_test3)  # [[5.0, 4.0], [12.0, 18.0], [60.0, 50.0, 10.0, -5.0]]
lambda_partition = lambda_partition_by_b_0(b_0_test3, lambda_test3) # [[1.0, 3.0], [15.0, 18.0], [20.0, 40.0, 42.0, 65.0]]
b_0_partition = y_partition_by_b_0(b_0_test3, b_0_test3)
prox_0_cluster = prox_slope(y=y_partition[0], lambdas=lambda_partition[0])
prox_k_clusters = [prox_0_cluster]

for k in range(2):
    b_0_kth = np.array(b_0_partition[k+1])
    y_kth = np.array(y_partition[k + 1])
    lambda_kth = np.array(lambda_partition[k + 1])
    prox_kth_cluster = prox_slope_on_b_0_single_cluster(b_0=b_0_kth, y=y_kth, lambdas=lambda_kth)
    prox_k_clusters.append(prox_kth_cluster)

#print(prox_k_clusters)
#print(u_reconstruction(b_0_test3, prox_k_clusters))
#print(b_0_partition)
#print(y_partition_by_b_0(b_0_test3, y_test3))
'''

