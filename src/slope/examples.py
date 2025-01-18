from src.slope.solvers import*
from admm_glasso import*
from my_plot import*



compound_block = comp_sym_corr(0.8, 3)
block_diag_matrix12 = np.block([[compound_block, np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3))],
                                [np.zeros((3,3)), compound_block, np.zeros((3,3)), np.zeros((3,3))],
                                [np.zeros((3,3)), np.zeros((3,3)), compound_block, np.zeros((3,3))],
                                [np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), compound_block]])

curvature = 0.04 # (0.04, 0.8) for mon beta
cluster_scaling = 0.8
A12concave = Aconcave(12, curvature, cluster_scaling)
#print('Aconcave', Aconcave(12, 0.04, 0.8))
A12flasso = Acustom(a=np.ones(12), b=np.ones(11) * sum(A12concave[i][i] for i in range(12)) * (1 / 11))
#print('A12flasso:\n', np.round(A12flasso, 3))
#print('lin_lambdas(12):', lin_lambdas(12))


#'''

# comparison simulations for [0,0,0,1,1,1,3,3,3,2,2,2]
plot_performance(b_0=np.array([0, 0, 0, 1, 1, 1, 3, 3, 3, 2, 2, 2]),  # np.array([1, 1, 1, 0, 0, 0, 3, 3, 3, 2, 2, 2])
                 C=block_diag_matrix12,
                 lambdas=lin_lambdas(12),
                 x=np.linspace(0, 2, 24),
                 n=100,
                 Cov=0.2**2*block_diag_matrix12,  #block_diag_matrix12
                 flasso=True,
                 A_flasso=A12flasso,
                 glasso=True,
                 A_glasso=A12concave,
                 smooth=True,
                 tol=1e-4)
                  #reducedOLS=True,
                  #sigma=0.2)



# comparison simulations for [0,0,1,0], [1,1,1,1], and [1,0,1,0]

rho = 0.8
plot_performance(b_0=np.array([1, 1, 1, 1]), #[1,1,0,1], [1,0,1,0] slope best, [1,1,1,1] flasso best, [0,1,1,0], [0,0,1,0] lasso best
                 C=np.array([[1, 0, rho, 0], [0, 1, 0, rho], [rho, 0, 1, 0], [0, rho, 0, 1]]), #(1-rho) * np.identity(4) + rho * np.ones((4, 4)),
                 lambdas= np.array([1.6, 1.2, 0.8, 0.4]), #np.array([4, 0, 0, 0]), #np.array([8, 4.1, 4, 3.9])/(20/4), #np.array([1.6, 1.2, 0.8, 0.4])
                 x=np.linspace(0,1,20),  # np.linspace(0.48, 0.55, 10)
                 n=1000,
                 Cov=0.4**2*np.array([[1, 0, rho, 0], [0, 1, 0, rho], [rho, 0, 1, 0], [0, rho, 0, 1]]),  # (1-rho) * np.identity(4) + rho * np.ones((4, 4)),
                 flasso=True,
                 A_flasso=Acustom(a=np.ones(4), b=1 * np.ones(3)),
                 #glasso=True,
                 #A_glasso=Acustom(a=np.ones(4), b=0.4 * np.array([1, 1, 1])),
                 #reducedOLS=True,
                 #sigma=0.4,
                 smooth=True,
                 legend=True)
#'''

#Phase Transition in Pattern recovery
'''
alpha1 = 2/3-0.05
alpha2 = 2/3
alpha3 = 2/3+0.05
C1 = np.array([[1, alpha1], [alpha1, 1]])
C2 = np.array([[1, alpha2], [alpha2, 1]])
C3 = np.array([[1, alpha3], [alpha3, 1]])
sigma=0.2
custom_points = np.array([0, 0.05, 0.18, 0.5, 1.2, 2])

#plot_performance_tripple(b_0=np.array([1, 0]), C1=C1, C2=C2, C3=C3, lambdas=np.array([3, 2]), x = np.linspace(0,0.4, 20), n=500, Cov1=sigma ** 2 * C1, Cov2= 0.2 ** 2 * C2, Cov3=sigma ** 2 * C3)
plot_performance_tripple(b_0=np.array([1, 0]), C1=C1, C2=C2, C3=C3, lambdas=np.array([3, 2]), x = custom_points, n=500, Cov1=sigma ** 2 * C1, Cov2= 0.2 ** 2 * C2, Cov3=sigma ** 2 * C3) #, Cov1=sigma**2*C1, Cov2=sigma**2*C2, Cov3=sigma**2*C3)
'''

#fused lasso simulation, for this I commented out lines 235 and 245 in my_plot.py
'''
beta_0=np.array([1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3])
#beta_0 = np.array([1, 3, 3,3,3,3,3,3,3,3,3, 2])
beta_0 = np.array([1,2,2,3])
p=len(beta_0)
plot_performance(b_0=beta_0,
                 C=np.identity(p),
                 Cov=1**2*np.identity(p),
                 x=np.linspace(0, 2, 8),
                 lambdas=lin_lambdas(p),
                 n=500,
                 flasso=True,
                 A_flasso=Acustom(a=np.zeros(p), b=np.ones(p)),
                 SLOPE=False,
                 Lasso=False,
                 smooth=False,
                 )
'''