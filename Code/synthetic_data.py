import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from OTEstimator import EntropicTransport
from functions import target_distribution, sample_target, LiftedMap, draw_samples, grad_log_ratio_closed_form, approximate, eigendecomposition, active_subspace




if __name__ == "__main__":
    D=4
    k=2
    np.random.seed(10)
    sigma, L, A_true = target_distribution(D, k=k, seed=10)
    samples = sample_target(L, 2000)
    original_samples = samples.clone()
    X_source = np.random.multivariate_normal(np.zeros(D), np.eye(D), size=2000) 
    m = 1

    for i in range(3):
        if m != 1:
            samples = original_samples @ W
            X_source_n = X_source @ W   # (N, n)
        else:
            X_source_n = X_source

        print("Doing iteration ", m)

    #=============================================
    #Create map
    #=============================================

        dim_r = D if m == 0 else samples.shape[1]
        eps_ott = .1
        n_inters_ott = 1000
        transport_map = EntropicTransport(dx=dim_r,estimator='OTT',eps=eps_ott) #inverse but that is used for everything
        transport_map.fit(source=np.asarray(samples, dtype=np.float32), target=X_source_n.astype(np.float32), max_iter=n_inters_ott)
        transport_map_d = EntropicTransport(dx=dim_r,estimator='OTT',eps=eps_ott) #direct and only used for samples
        transport_map_d.fit(source=X_source_n.astype(np.float32), target=np.asarray(samples, dtype=np.float32), max_iter=n_inters_ott)

        if m != 1:
            transport_map = LiftedMap(transport_map, transport_map_d, W, D, fitted_samples= samples, X_source= X_source_n)

    #=============================================
    #Algorithm to find subspace
    #=============================================

        if isinstance(transport_map, LiftedMap):
            samples_2 = transport_map.draw_samples(D, 1000, X_source)
        else:
            samples_2 = draw_samples(transport_map_d, D, 1000, X_source_n)

        Y_jax = jnp.array(transport_map.solver.target, dtype=jnp.float64)
        g_jax = jnp.array(transport_map.solver.gYjs,   dtype=jnp.float64)
        eps   = transport_map.solver.eps

        if hasattr(transport_map, 'W'):
            W_jax = jnp.array(transport_map.W)   # (D, n)
            grad_fn = jax.jit(jax.vmap(
                lambda x: W_jax @ grad_log_ratio_closed_form(x @ W_jax, Y_jax, g_jax, eps)
            ))
        else:
            grad_fn = jax.jit(jax.vmap(
                lambda x: grad_log_ratio_closed_form(x, Y_jax, g_jax, eps)
            ))

        grads = np.asarray(grad_fn(jnp.array(samples_2, dtype=jnp.float64)))  # (200, 4)

        if not hasattr(transport_map, 'W'):
            z_test = np.asarray(samples_2[0])
            g_approx = np.asarray(grad_log_ratio_closed_form(
                jnp.array(z_test, dtype=jnp.float64), Y_jax, g_jax, eps))
            g_true = (np.eye(D) - np.linalg.inv(sigma.numpy())) @ z_test

        if not hasattr(transport_map, 'W'):
          z_jax = jnp.array(np.asarray(samples_2[0]), dtype=jnp.float64)
          phi = (g_jax - jnp.sum((z_jax - Y_jax)**2, axis=1)) / eps
          p   = jax.nn.softmax(phi)
          S_x = p @ Y_jax
          C   = (p[:, None] * Y_jax).T @ Y_jax - jnp.outer(S_x, S_x)
          J_S = 2 * C / eps
          term1 = -J_S.T @ S_x
          term2 = z_jax

          evals = jnp.linalg.eigvalsh(C)
          cond  = jnp.max(jnp.abs(evals)) / jnp.maximum(jnp.min(jnp.abs(evals)), 1e-15)
          eff_n = jnp.exp(-jnp.sum(p * jnp.log(p + 1e-300)))

          C_inv         = jnp.linalg.inv(C)
          diff          = Y_jax - S_x[None, :]
          C_inv_applied = (Y_jax - 2*S_x[None, :]) @ C_inv
          alpha         = jnp.sum(Y_jax * C_inv_applied, axis=1)
          term3         = (2/eps) * (p * alpha) @ diff


        c_hat = approximate(grads)
        eigenvalues, eigenvectors = eigendecomposition(c_hat)    


    #======================================================================
    #Bootstrap
    #======================================================================

        M_boot = 1000
        boot_eigenvalues = np.zeros((M_boot, D))

        for i in range(M_boot):
            jk = np.random.randint(0, grads.shape[0], size=grads.shape[0])
            boot_grads = grads[jk]
            C_boot = (1/grads.shape[0]) * boot_grads.T @ boot_grads
            evals_boot, _ = np.linalg.eigh(C_boot)
            boot_eigenvalues[i] = np.sort(evals_boot)[::-1]


        lower = np.maximum(np.percentile(boot_eigenvalues, 2.5, axis=0), 1e-10)
        upper = np.maximum(np.percentile(boot_eigenvalues, 97.5, axis=0), 1e-10)

        plt.semilogy(range(1, D+1), np.maximum(eigenvalues, 1e-10), 'ko-', label='estimate')
        plt.fill_between(range(1, D+1), lower, upper, alpha=0.3, label='95\% bootstrap interval')
        plt.xlabel('dimension'); plt.ylabel('eigenvalue (log scale)')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.savefig(f'Figures/synthetic_data_eigenvalue_bootstrap_{m}.pdf', bbox_inches='tight')
        plt.show()

    #======================================================================
    #Visually find subspace
    #======================================================================
    
        n = int(input("Enter the number of active dimensions based on the graph: "))

        W = active_subspace(n, eigenvectors)
        print("Active subspace W:")
        print(W)

        m=m+1




# #========================================================================================================================
# #Checks
# #========================================================================================================================

# #========================================================================================================================
# #check transport map
# #========================================================================================================================

        if isinstance(transport_map, LiftedMap):
            pushed = transport_map.draw_samples(D, 1000, X_source)
        else:
            pushed = draw_samples(transport_map_d, D, 1000, X_source_n)
        pushed_mean = np.mean(pushed, axis=0)
        pushed_std = np.std(pushed, axis=0)

        true_mean = np.zeros(D)
        true_std = np.sqrt(np.diag(sigma.numpy()))

        print("Pushed mean:", pushed_mean)
        print("True mean:  ", true_mean)
        print("Pushed std: ", pushed_std)
        print("True std:   ", true_std)
        
        
        
#=================================================================================================================
# Projecting samples onto random directions and comparing distributions
#=================================================================================================================

n_directions = 5
n_samples    = 2000

# true target 
x_true = sample_target(L, n_samples).numpy()        


# predicted samples from learned map 
if isinstance(transport_map, LiftedMap):
    x_tilde = np.array(transport_map.draw_samples(D, n_samples, X_source))
else:
    x_tilde = np.array(draw_samples(transport_map_d, D, n_samples, X_source_n))

# 5 random directions 
np.random.seed(0) 
vs = np.random.randn(n_directions, D)
vs = vs / np.linalg.norm(vs, axis=1, keepdims=True)   


fig = plt.figure(figsize=(7, 8), layout='constrained')   
gs  = GridSpec(3, 4, figure=fig)                         

axes = [
    fig.add_subplot(gs[0, :2]),   
    fig.add_subplot(gs[0, 2:]),  
    fig.add_subplot(gs[1, :2]),   
    fig.add_subplot(gs[1, 2:]),  
    fig.add_subplot(gs[2, 1:3]),  
]

for i, (v, ax) in enumerate(zip(vs, axes)):
    xv       = x_true  @ v    # true projection
    x_tilde_v = x_tilde @ v   # predicted projection

    ax.hist(xv,         bins=40, alpha=0.5, density=True,
            label=r'$x \cdot v$',         color='steelblue')
    ax.hist(x_tilde_v,  bins=40, alpha=0.5, density=True,
            label=r'$\tilde{x} \cdot v$', color='tomato')

    ax.set_title(f'$v_{{{i+1}}}$')
    ax.set_xlabel('projection value')
    if i == 0:
        ax.set_ylabel('density')
        ax.legend()

plt.savefig(f'Figures/synthetic_data_true_vs_learned_distribution.pdf', bbox_inches='tight')
plt.show()