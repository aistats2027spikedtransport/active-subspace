import numpy as np
import matplotlib.pyplot as plt
from functions import sample_target, convergence_jax, target_distribution
from OTEstimator import EntropicTransport


num_samples = [2000, 4000, 6000, 8000, 10000, 15000]
kl_errors = []
D=4
k=2

np.random.seed(10) #Using the same seed for synthetic_data.py
sigma, L, A_true = target_distribution(D, k=k, seed=10) 
eval_samples = sample_target(L, 3000).numpy()


np.random.seed(5)
for N in num_samples:
    print(f"Training with N={N}...")
    samples_N = sample_target(L, N)
    X_source_N = np.random.multivariate_normal(np.zeros(D), np.eye(D), size=N)

    tm = EntropicTransport(dx=D, estimator='OTT', eps=0.1)
    tm.fit(source=np.asarray(samples_N, dtype=np.float32),
           target=X_source_N.astype(np.float32),
           max_iter=1000)

    kl = convergence_jax(tm, eval_samples, sigma, D)
    kl_errors.append(kl)
    print(f"  KL = {kl:.4f}")

# Plot
plt.figure(figsize=(6, 4))
plt.loglog(num_samples, kl_errors, 'o-', color='steelblue')
# Reference lines for convergence rates
ns = np.array(num_samples, dtype=float)
plt.loglog(ns, kl_errors[0] * (ns[0]/ns),     '--', color='gray', alpha=0.6, label=r'$O(1/N)$')
plt.loglog(ns, kl_errors[0] * (ns[0]/ns)**0.5, ':',  color='gray', alpha=0.6, label='$O(1/\sqrt{N})$')
plt.xlabel('Number of training samples $N$', fontsize=14)
plt.ylabel(r'$D_{KL}(\rho \| \rho_{\theta})$', fontsize=14)
plt.legend(fontsize=14)
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig(f'Figures/synthetic_data_convergence.pdf', bbox_inches='tight')
plt.show()