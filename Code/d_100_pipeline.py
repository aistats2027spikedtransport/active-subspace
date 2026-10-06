from functions import get_data, LiftedMap, draw_samples, grad_log_ratio_closed_form, approximate, eigendecomposition, active_subspace
from OTEstimator import EntropicTransport
import torch
import numpy as np
import matplotlib.pyplot as plt
import jax, jax.numpy as jnp
from pathlib import Path
jax.config.update("jax_enable_x64", True)
from matplotlib.gridspec import GridSpec

REPO_ROOT = Path(__file__).resolve().parent.parent 
DATA_DIR = REPO_ROOT / "Data"

_original_show = plt.show
def _colab_like_show(*args, **kwargs):
    # remove empty placeholder figures
    for num in plt.get_fignums():
        if not plt.figure(num).axes:
            plt.close(num)
    nums = plt.get_fignums()
    if not nums:
        return
    _original_show(block=False)                
    for num in nums:
        plt.figure(num).canvas.draw_idle()
    plt.figure(nums[-1]).canvas.start_event_loop(0.5)   
    plt.figure()                              
plt.show = _colab_like_show


if __name__ == "__main__":
    D=100
    np.random.seed(4937)
    #df = get_data('/Users/anaelisalopez/Documents/Summer 2026/Ricardo Baptista/GenerativeModelCompression/Data/citeseq.csv')
    df = get_data(DATA_DIR / "citeseq.csv")
    
    data_cols = [f'd{i}' for i in range(0, D)]
    samples = df[data_cols].values 
    samples = torch.tensor(df[data_cols].values[:500], dtype=torch.float64)
    original_samples = samples.clone()
    X_source = np.random.multivariate_normal(np.zeros(D), np.eye(D), size=2000)
    m = 1


    for i in range(3):
        if m != 1:
            samples = original_samples @ W
            X_source_n = X_source @ W
        else:
            X_source_n = X_source


        print("Doing iteration ", m)

    #=============================================
    #Create map
    #=============================================

        dim_r = D if m ==0 else samples.shape[1]
        eps_ott = 0.1 * D
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
            samples_2 = transport_map.draw_samples(D, 200, X_source)
        else:
            samples_2 = draw_samples(transport_map_d, D, 200, X_source_n)

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

        batch_size=5
        batch_size = max(1, batch_size // 4)
        grads_list =[]
        for i in range(0, len(samples_2), batch_size):
            batch = jnp.array(samples_2[i:i+batch_size], dtype=jnp.float64)
            grads_list.append(np.asarray(grad_fn(batch)))
        grads = np.concatenate(grads_list, axis=0)
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
            eff_n = jnp.exp(-jnp.sum(p * jnp.log(p + 1e-300)))   # effective sample size

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

        plt.semilogy(range(D), np.maximum(eigenvalues, 1e-10), 'ko-', label='estimate')
        plt.fill_between(range(D), lower, upper, alpha=0.3, label='95\% bootstrap interval')
        plt.xlabel('dimension'); plt.ylabel('eigenvalue (log scale)')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.savefig(f'Figures/d=100_data_cumulative_variance_{m}.pdf', bbox_inches='tight')
        plt.show()

    #======================================================================
    #Visually find subspace
    #======================================================================

        n = int(input("Enter the number of active dimensions based on the graph: "))

        evals_plot = eigenvalues[:dim_r]
        lower_plot  = lower[:dim_r]
        upper_plot  = upper[:dim_r]

        fig1, ax1 = plt.subplots(figsize=(7, 5))

        ax1.semilogy(range(dim_r), np.maximum(evals_plot, 1e-10), 'ko-', label='estimate')
        ax1.fill_between(range(dim_r), lower_plot, upper_plot, alpha=0.3, label='95\% bootstrap interval')
        ax1.axvline(x=n-1, color='tomato', linestyle='--', linewidth=1.5, label=f'n = {n}')
        ax1.set_xlabel('dimension')
        ax1.set_ylabel('eigenvalue (log scale)', fontsize =14)
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        ax1.legend()
        fig1.savefig(f'Figures/d=100_eigenvalue_bootstrap_n={n}_{m}_full.pdf', bbox_inches='tight')
        plt.show()

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.semilogy(range(n), np.maximum(evals_plot[:n], 1e-10), 'ko-', label='estimate')
        ax2.fill_between(range(n), lower_plot[:n], upper_plot[:n], alpha=0.3, label='95\% bootstrap interval')
        ax2.set_xlabel('dimension', fontsize =14)
        ax2.set_ylabel('eigenvalue (log scale)', fontsize =14)
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)
        ax2.legend()

        fig2.savefig(f'Figures/d=100_eigenvalue_bootstrap_n={n}_{m}_first_n.pdf', bbox_inches='tight')
        plt.show()


        W = active_subspace(n, eigenvectors)
        print("Active subspace W:")
        print(W)
        print("W shape:", W.shape)

        m=m+1


#=================================================================================================================
# Projecting samples onto random directions and comparing distributions
#=================================================================================================================

n_directions = 5
n_samples    = 2000


data_cols = [f'd{i}' for i in range(0, D)] 
samples = df[data_cols].values 
x_true = torch.tensor(df[data_cols].values[:500], dtype=torch.float64)

# predicted samples
if isinstance(transport_map, LiftedMap):
    x_tilde = np.array(transport_map.draw_samples(D, n_samples, X_source))
else:
    x_tilde = np.array(draw_samples(transport_map_d, D, n_samples, X_source_n))


# 5 random directions 
np.random.seed(0)
vs = np.random.randn(n_directions, D)
vs = vs / np.linalg.norm(vs, axis=1, keepdims=True)  

# Plot
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
    xv       = x_true  @ v # true projection
    x_tilde_v = x_tilde @ v # predicted projection

    ax.hist(xv,         bins=40, alpha=0.5, density=True,
            label=r'$x \cdot v$',         color='steelblue')
    ax.hist(x_tilde_v,  bins=40, alpha=0.5, density=True,
            label=r'$\tilde{x} \cdot v$', color='tomato')
    ax.set_title(f'$v_{{{i+1}}}$')
    ax.set_xlabel('projection value', fontsize=15)
    if i == 0:
        ax.set_ylabel('density', fontsize=15)
        ax.legend(fontsize=14)

plt.savefig(f'Figures/d=100_data_true_vs_learned_distribution.pdf', bbox_inches='tight')
plt.show()
_original_show()