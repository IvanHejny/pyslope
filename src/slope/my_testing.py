from src.slope.solvers import*

'''
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


C = np.array([[1, 0.5], [0.5, 1]])
W = np.array([5.0, 4.0])
lambdas = np.array([0.9, 0.2])
b_00 = np.array([1, 1])
stepsize_t = 0.35 # to guarantee convergence take stepsize < 1/max eigenvalue of C (max eval of C is the Lipschitz constant of grad(1/2 uCu - uW)=(Cu-W))


print("prox_slope_b_0:", prox_slope_b_0(b_00, W, lambdas)) # first simple verification of ISTA, FISTA
print("pdg_slope_b_0_ISTA:", pgd_slope_b_0_ISTA(C, W, b_00, lambdas, stepsize_t, 20))
print("pdg_slope_b_0_FISTA:", pgd_slope_b_0_FISTA(C, W, b_00, lambdas, stepsize_t, 20))
#print("pdg_slope_b_0_ISTA:", pgd_slope_b_0_ISTA(C, [5,4], b_00, [1,1], stepsize_t, 20)) #typeproblem

for i in range(40):
    one_solution = pgd_slope_b_0_ISTA(C, np.random.multivariate_normal([0, 0], [[1.7, 0], [0, 1.9]]), np.array([0, 0]), lambdas, stepsize_t, 20)
    print("pdg_slope_b_0_ISTA: [0,0]" , i+1, one_solution) #all patterns attainable

for i in range(40):
    one_solution = pgd_slope_b_0_ISTA(C, np.random.multivariate_normal([0, 0], [[1.7, 0], [0, 1.9]]), np.array([1, 1]), lambdas, stepsize_t, 20)
    print("pdg_slope_b_0_ISTA: [1, 1]", i+1, one_solution) #

for i in range(40):
    one_solution = pgd_slope_b_0_ISTA(C, np.random.multivariate_normal([0, 0], [[1.7, 0], [0, 1.9]]), np.array([0, 1]), lambdas, stepsize_t, 20)
    print("pdg_slope_b_0_ISTA: [0, 1]", i+1, one_solution)


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



#print("prox_slope_b_0:", prox_slope_b_0( b_0 = b_0_test01, y = np.array([5.0, -2.0, 3.0, 4.0]), lambdas = lambdas_test1))
#print("pdg_slope_b_0_ISTA:", pgd_slope_b_0_ISTA(C = np.identity(4), W = np.array([5.0, -2.0, 3.0, 4.0]), b_0 = b_0_test01, lambdas = lambdas_test1, t = stepsize_t, n = 20))


#for i in range(10):
#    print("pdg_slope_b_0_FISTA:", pgd_slope_b_0_FISTA(C_test1, np.random.multivariate_normal([0,0], [[20, 0], [0, 20]])[0], b_0_test01, lambdas_test01, stepsize_t, 4))


b_0_test3 = np.array([0, 2, 0, 2, -2, -2, 1, 1])
y_test3 = np.array([5.0, 60.0, 4.0, 50.0, 10.0, -5.0, 12.0, 17.0])

lambda_test3 = [65.0, 42.0, 40.0, 20.0, 18.0, 15.0, 3.0, 1.0]
lambda_test4 = [35.0, 35.0, 5.2, 5.2, 5.2, 5.2, 5.2, 5.2]
lambda_test5 = [35.0, 35.0, 4.8, 4.8, 4.8, 4.8, 4.8, 4.8]

#print('prox_slope_b_0_3:', prox_slope_b_0(b_0_test3, y_test3, lambda_test3))
#print('prox_slope_b_0:', prox_slope_b_0(b_0_test3, y_test3, np.flip(lambda_test3)))# flipping lambdas has no effect (sanity check)
#print('prox_slope_b_0_4:', prox_slope_b_0(b_0_test3, y_test3, lambda_test4))
#print('prox_slope_b_0_5:', prox_slope_b_0(b_0_test3, y_test3, lambda_test5))

C_test3 = np.identity(8)
W_test3 = np.array([5.0, -2.0, 3.0, 3.1, -2.5, -5.2, 0.7, -7.0])
#print("pdg_slope_b_0_FISTA:", pgd_slope_b_0_FISTA(C_test3, W_test3, b_0_test3, lambda_test3, stepsize_t, 20))



'''
# Compare with
print('b_0:', b_0_test3)
print("zero-cluster:", prox_slope(y=np.array([5.0, 4.0]), lambdas=np.array([3.0, 1.0])),
      "one-cluster:", prox_slope_on_b_0_single_cluster(b_0=np.array([1, 1]), y=np.array([12.0, 17.0]),
                                                       lambdas=np.array(np.array([18.0, 15.0]))),
      "two-cluster:",
      prox_slope_on_b_0_single_cluster(b_0=np.array([2, 2, -2, -2]), y=np.array([60.0, 50.0, 10.0, -5.0]),
                                       lambdas=np.array([65.0, 42.0, 40.0, 20.0])))
'''

'''
X = np.array([[1.0, 0.0], [0.0, 1.0]])
y_1 = np.array([5.0, 4.0])
lambdas_1 = np.array([3.0, 1.0])
y_2 = np.array([10.0, -5.0])
lambdas_2 = np.array([25.0, 5.0])
'''
# lambdas_3s = np.array([5.0, 40.0, 65.0, 25.0]) resorting lambdas no longer messes up with the solution, lambdas are always sorted by the functions

#y_test1s = np.array([-5.0, 60.0, 50.0, 10.0])  # reshuffling y
#y_test1t = np.array([60.0, 50.0, -10.0, 5.0])  # swapping last two signs in y_3 multiplying y S_b_0 with b_0=[2,2,-2,-2]

# y_4 = np.array([60.0, 50.0, 10.0, -5.0, -7.0])
# lambdas_4 = np.array([65.0, 40.0, 25.0, 5.0, 1.0])

#print("prox_slope:", prox_slope(y_test1, lambdas_test1))
#print("prox_slope_b_0, b_0= [0,0,0,0] is:", prox_slope_b_0(b_0_test0, y_test1, lambdas_test1))

#print("prox_slope_isotonic:", prox_slope_isotonic(y_test1, lambdas_test1))
#print("prox_slope_isotonic:", prox_slope_isotonic(y_test1s, lambdas_test1))#reshuffling y only reshuffles the solution (keeping lambdas fixed)
#print("prox_slope_isotonic:", prox_slope_isotonic(y_test1t, lambdas_test1))#swapping last two signs in y changes the solution significantly. Compare with prox_slope_on_b_0_single_cluster for b_0=[2,2,-2,-2]

#print("prox_slope_on_b_0_cluster:", prox_slope_on_b_0_single_cluster(b_0_test1, y_test1, lambdas_test1))
"""
b_0=np.array([2, 2, -2, -2])
sign_b_0=np.sign(b_0)
print(b_0)
print(sign_b_0)
identity=np.identity(4)
print(np.diag(sign_b_0))
#print(b_0 @ np.diag(sign_b_0))
"""

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

"""
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
"""

'''
print('b_0_test3:', b_0_test3)
print('y_test3', y_test3)

partition_test3 = y_partition_by_b_0(b_0_test3, y_test3)
print('y_partition_by_b_0:', y_partition_by_b_0(b_0_test3, y_test3))

print('u_reconstruction:', u_reconstruction(b_0_test3, partition_test3))

print('lambda:', lambda_test3)
print('lambda_partition_by_b_0:', lambda_partition_by_b_0(b_0_test3, lambda_test3))
'''

'''
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
'''
#print(y_partition_by_b_0(b_0_test3, y_test3))