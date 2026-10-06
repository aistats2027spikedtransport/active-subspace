import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm
from functions import target_distribution, sample_target, LiftedMap, draw_samples, grad_log_ratio_closed_form, approximate, eigendecomposition, active_subspace
from OTEstimator import EntropicTransport
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)



def T_hat_r(x, W, S_r_half):
    """
    Computes T_hat_r(x) to evalute how the bias and variance of the learned transport map in the synthetic data 
    experiment changes based on the dimension chosen for the active subspace.
    
    Input: 
    x: (N, D) array of samples to be mapped
    W: (D, r) array of active subspace directions
    S_r_half: (r, r) array of the square root of the covariance matrix
    
    Output:
    (N, D) array of mapped samples
    """
    x = np.asarray(x)
    proj = x @ W
    lifted = (proj @ S_r_half.T) @ W.T
    inactive = x - proj @ W.T
    return lifted + inactive


np.random.seed(111)
D=4
k=2
sigma, L, A_true = target_distribution(D, k=k, seed=10)
sigma_np = sigma.numpy()
original_samples = np.asarray(sample_target(L, 2000), dtype=np.float64)
X_source = np.random.multivariate_normal(np.zeros(D), np.eye(D), size=2000)

T_true = X_source @ np.real(sqrtm(sigma_np)).T
eps_ott=0.1 
n_iters_ott=1000
n_list, bias_list, var_list, total_list = [], [], [], []

for n in range(D + 1):
    if n == 0:
        T_oracle = X_source.copy()     
        T_learned = X_source.copy()
    else:
        W = None                         
        for i in range(3):
            if W is None:
                samples, X_source_n = original_samples, X_source
            else:
                samples, X_source_n = original_samples @ W, X_source @ W
            dim_r = samples.shape[1]

            transport_map = EntropicTransport(dx=dim_r, estimator='OTT', eps=eps_ott)
            transport_map.fit(source=samples.astype(np.float32),
                                target=X_source_n.astype(np.float32), max_iter=n_iters_ott)
            transport_map_d = EntropicTransport(dx=dim_r, estimator='OTT', eps=eps_ott)
            transport_map_d.fit(source=X_source_n.astype(np.float32),
                                target=samples.astype(np.float32), max_iter=n_iters_ott)

            if W is not None:
                transport_map = LiftedMap(transport_map, transport_map_d, W, D,
                                            fitted_samples=samples, X_source=X_source_n)

            
            if isinstance(transport_map, LiftedMap):
                samples_2 = transport_map.draw_samples(D, 1000, X_source)
            else:
                samples_2 = draw_samples(transport_map_d, D, 1000, X_source_n)

            Y_jax = jnp.array(transport_map.solver.target, dtype=jnp.float64)
            g_jax = jnp.array(transport_map.solver.gYjs, dtype=jnp.float64)
            eps = transport_map.solver.eps

            if hasattr(transport_map, 'W'):
                W_jax = jnp.array(transport_map.W)
                grad_fn = jax.jit(jax.vmap(
                    lambda x: W_jax @ grad_log_ratio_closed_form(x @ W_jax, Y_jax, g_jax, eps)))
            else:
                grad_fn = jax.jit(jax.vmap(
                    lambda x: grad_log_ratio_closed_form(x, Y_jax, g_jax, eps)))

            grads = np.asarray(grad_fn(jnp.array(samples_2, dtype=jnp.float64)))
            c_hat = approximate(grads)
            eigenvalues, eigenvectors = eigendecomposition(c_hat)
            W = active_subspace(n, eigenvectors)

        # use the subspace the final map was actually trained on
        W_fit = np.asarray(transport_map.W)
        S_r_half = np.real(sqrtm(W_fit.T @ sigma_np @ W_fit))
        T_oracle = T_hat_r(X_source, W_fit, S_r_half)
        T_learned = transport_map.evaluate(X_source) 

    bias_sq  = np.mean(np.sum((T_true - T_oracle) ** 2, axis=1))
    var_sq   = np.mean(np.sum((T_oracle - T_learned) ** 2, axis=1))
    total_sq = np.mean(np.sum((T_true - T_learned) ** 2, axis=1))

    n_list.append(n)
    bias_list.append(np.sqrt(bias_sq))
    var_list.append(np.sqrt(var_sq))
    total_list.append(np.sqrt(total_sq))
    
plt.close('all')                     
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(n_list, bias_list, 'o-', label=r'$\|T - \hat{T}_n\|$ (bias)', color='steelblue')
ax.plot(n_list, var_list, 's-', label=r'$\|\hat{T}_n - \hat{T}_n^N\|$ (variance)', color='tomato')
ax.set_xlabel('Subspace dimension $n$', fontsize=14)
ax.set_ylabel('$L^2$ norm', fontsize=14)
ax.set_xticks(n_list)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12, loc='upper left')
plt.tight_layout()
plt.savefig('Figures/synthetic_data_bias_variance_tradeoff.pdf', bbox_inches='tight')
plt.show()