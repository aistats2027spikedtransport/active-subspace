import torch
import numpy as np
import matplotlib.pyplot as plt
import jax, jax.numpy as jnp
from scipy.stats import multivariate_normal
import pandas as pd



def target_distribution(D, k, seed=10):
    """
    This function creates a target distribution N(0, I + A^TA) where A is a kxD matrix with k<=D.

    Input:
    D: dimension of the target distribution
    k: rank of the low-rank perturbation (k <= D)
    seed: random seed for reproducibility
    
    Output:
    sigma: covariance matrix of the target distribution
    L: Cholesky decomposition of sigma (L @ L^T = sigma)
    A_true: the low-rank matrix A used to construct sigma
    """
    torch.manual_seed(seed)
    A = torch.randn(k, D)
    sigma = torch.eye(D) + A.T @ A 
    L = torch.linalg.cholesky(sigma) 
    return sigma, L, A



def sample_target(L, n):
    """
    This function samples n points from the target distribution N(0, I + A^TA) using the Cholesky decomposition L.
    
    Input:
    L: Cholesky decomposition of the covariance matrix (L @ L^T = sigma)
    n: number of samples to draw
    
    Output:
    x: samples drawn from the target distribution, shape (n, D)
    """
    D = L.shape[0]
    z = torch.randn(n, D) 
    x = z @ L.T
    return x 

#========================================================================================================================
#Step 2: define target distribution
#========================================================================================================================

def log_target_density(transport_map, z):
    """
    Compute the log density of the target distribution at point z using the transport map using Equation 3 from 
    George P's paper. The log density is computed as:
    log p_tar(z) = log p_ref(S(z)) + log|det ∇S(z)|
    
    Input:
    transport_map: the fitted transport map that maps from the target distribution to the reference distribution
    z: point at which to evaluate the log density, shape (D,)
    
    Output:
    log_density: the log density of the target distribution at point z
    """
    z_np = np.asarray(z, dtype=np.float64).reshape(1, -1)
    Sz = transport_map.evaluate(z_np)
    log_det = transport_map.grad_map(z_np)
    D = Sz.shape[1]
    log_p_ref = -0.5 *np.sum(Sz**2, axis=1) - (D/2) * np.log(2*np.pi) #the log of the standard gaussian which is originally
    return float(np.squeeze(log_p_ref + log_det))

#========================================================================================================================
#3 find subspace
#========================================================================================================================


def draw_samples(transport_map_d, D, M, X_source, seed=10):
    """
    This function draws M samples from the target distribution using the fitted transport map. The samples are drawn
    independently according to the density function.
    
    Inputs:
    transport_map_d: the fitted transport map that maps from the reference distribution to the target distribution
    D: dimension of the target distribution
    M: number of samples to draw
    X_source: samples from the reference distribution (standard Gaussian), shape (N, D)
    
    Output:
    result: samples drawn from the target distribution, shape (M, D)
    """
    idx = np.random.choice(len(X_source), size=M, replace=False)
    result = transport_map_d.evaluate(X_source[idx])
    return np.asarray(result)


def approximate(gradients):
    """
    This function approximates the uncentered covariance matrix of the gradients. Given a set of gradients, it computes
    the uncentered covariance matrix C_hat as:
    C_hat = 1/M * sum(gradients[i] @ gradients[i].T) for i=1 to M, where M is the number of gradients.
    
    Input:
    gradients: a set of gradients, shape (M, D) where M is the number of gradients and D is the dimension of each gradient.
    
    Output:
    C_hat: the approximated uncentered covariance matrix, shape (D, D)
    """
    M = gradients.shape[0]
    return (1/M) * gradients.T @ gradients

def eigendecomposition(C_hat):
    """
    This computes the eigendecomposition of the uncentered covariance matrix C_hat. It returns the eigenvalues and eigenvectors
    sorted in descending order. The eigenvalues and eigenvectors can be used to identify the active subspace of the target distribution.
    
    Input:
    C_hat: the uncentered covariance matrix, shape (D, D)
    
    Output:
    eigenvalues: the eigenvalues of C_hat, sorted in descending order, shape (D,)
    eigenvectors: the eigenvectors of C_hat, sorted according to the eigenvalues, shape (D, D)
    """
    eigenvalues, eigenvectors = np.linalg.eigh(C_hat) #smallest to largest
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1] #reverse
    return eigenvalues, eigenvectors


def active_subspace(n, eigenvectors):
    """
    This function returns the active subspace of the target distribution given the number of active dimensions n and the eigenvectors
    
    Input:
    n: number of active dimensions
    eigenvectors: the eigenvectors of the uncentered covariance matrix C_hat
    
    Output:
    W: the active subspace, shape (D, n) where D is the dimension of the target distribution
    """
    return eigenvectors[:, :n]

class LiftedMap:
    """
    This class takes the approximated map in a lower dimension and lifts it up to the original dimension.
    """
    def __init__(self, transport_map_r, transport_map_d_r, W, D, fitted_samples, X_source):
        self.transport_map_r = transport_map_r
        self.transport_map_d_r = transport_map_d_r
        self.W = W
        self.D = D
        self.fitted_samples = fitted_samples
        self.X_source = X_source

    @property
    def solver(self):
        """
        This property returns the solver of the transport map in the reduced dimension.
        
        Output:
        solver: the solver of the transport map in the reduced dimension
        """
        return self.transport_map_r.solver

    def draw_samples(self, D, M, X_source, seed=10):
        """
        This function draws M samples from the target distribution using the fitted transport map in the reduced dimension.
        
        Inputs:
        D: dimension of the target distribution
        M: number of samples to draw
        X_source: samples from the reference distribution (standard Gaussian), shape (N, D)
        
        Output:
        result: samples drawn from the target distribution, shape (M, D)
        """
        idx = np.random.choice(len(X_source), size=M, replace=False)
        x_sub = X_source[idx]           # (M, D) Gaussian samples
        active = x_sub @ self.W         # (M, r) project to active subspace
        gen = self.transport_map_d_r.evaluate(active)  # (M, r) generate in reduced space
        inactive = x_sub @ (np.eye(D) - self.W @ self.W.T).T  # (M, D) inactive part
        return gen @ self.W.T + inactive  # lift back to full space

    def evaluate(self, X):
        """
        This function evaluates the transport map in the reduced dimension and lifts it back to the original dimension.
        
        Input:
        X: points at which to evaluate the transport map, shape (N, D)
        
        Output:
        result: evaluated points in the original dimension, shape (N, D)
        """
        X = np.array(X)
        if X.shape[1]==self.W.shape[1]:
            X = X @ self.W.T
        down = X @ self.W
        Tr_Wx = np.array(self.transport_map_r.evaluate(down))
        inactive = (np.eye(self.D) - self.W @ self.W.T) @ X.T
        return (self.W @ Tr_Wx.T + inactive).T  # (D, N) lift back up


    def grad_map(self, X):
        """
        This function computes the gradient of the transport map in the reduced dimension and lifts it back to the original dimension.
        
        Input:
        X: points at which to evaluate the gradient, shape (N, D)
        
        Output:
        result: gradient evaluated points in the original dimension, shape (N, D)
        """
        active = X @ self.W   # (n, N)
        return self.transport_map_r.grad_map(active)  # (N,)


def get_data(data_path):
    '''
    This function read the data from the given path and returns the dataframe.
    
    Input:
    data_path: path to the data file
    
    Output:
    df: dataframe containing the data
    '''
    df = pd.read_csv(data_path)
    data_cols = sorted(
    [c for c in df.columns if c.startswith('d') and c[1:].isdigit()],
    key=lambda x: int(x[1:])
)
    D = len(data_cols)
    print(f"D = {D}, columns: {data_cols[0]} to {data_cols[-1]}")
    return df


def grad_log_ratio_closed_form(x, Y, g, eps, min_eff_n=20.0):
    """
    This function computes the gradient of the log ratio of the target density and the reference density at point x using the closed form expression.
    
    Input:
    x: point at which to evaluate the gradient, shape (D,)
    Y: samples from the target distribution, shape (N, D)
    g: potentials evaluated at Y, shape (N,)
    eps: regularization parameter used in the transport map
    
    Output:
    grad: gradient of the log ratio at point x, shape (D,)
    """
    phi = (g - jnp.sum((x - Y)**2, axis=1)) / eps   
    p   = jax.nn.softmax(phi)                               
    S_x = p @ Y                                            
    C   = (p[:, None] * Y).T @ Y - jnp.outer(S_x, S_x)   
    J_S = 2 * C / eps                                           

    term1 = -J_S.T @ S_x                                  
    term2 = x
    eff_n = jnp.exp(-jnp.sum(p * jnp.log(p + 1e-300)))

    C_inv          = jnp.linalg.inv(C)                     
    diff           = Y - S_x[None, :]                      
    C_inv_applied  = (Y - 2*S_x[None, :]) @ C_inv         
    alpha          = jnp.sum(Y * C_inv_applied, axis=1)    
    term3_raw          = (2/eps) * (p * alpha) @ diff         
    weight = jnp.clip((eff_n - 2.0) / (min_eff_n - 2.0), 0.0, 1.0)
    term3 = weight * term3_raw

    return term1 + term2 +term3


def convergence_jax(transport_map, eval_samples, sigma, D):
    """
    This function computes the KL divergence between the target distribution and the distribution induced by the transport map. 

    Input:
    transport_map: the fitted transport map that maps from the reference distribution to the target distribution
    eval_samples: samples drawn from the target distribution, shape (M, D)
    sigma: covariance matrix of the target distribution, shape (D, D)
    D: dimension of the target distribution

    Output:
    kl: the KL divergence between the target distribution and the distribution induced by the transport map
    """    
    p_true = multivariate_normal(mean=np.zeros(D), cov=sigma.numpy())

    Y_jax = jnp.array(transport_map.solver.target, dtype=jnp.float64)
    g_jax = jnp.array(transport_map.solver.gYjs,   dtype=jnp.float64)
    eps   = transport_map.solver.eps

    def log_rho_theta(x):
        """
        This function computes the log density of the distribution induced by the transport map at point x using Equation 3 from George P's paper. The log density is computed as:
        log p_theta(x) = log p_ref(S(x)) + log|det ∇S(x)|
        
        Input:
        x: point at which to evaluate the log density, shape (D,)
        
        Output:
        log_density: the log density of the distribution induced by the transport map at point x
        """
        phi  = (g_jax - jnp.sum((x - Y_jax)**2, axis=1)) / eps
        p    = jax.nn.softmax(phi)
        Sx   = p @ Y_jax                              # (D,)
        C    = (p[:, None] * Y_jax).T @ Y_jax - jnp.outer(Sx, Sx)
        _, log_det = jnp.linalg.slogdet(2 * C / eps)
        log_p_ref  = -0.5 * jnp.sum(Sx**2) - (D/2) * jnp.log(2*jnp.pi)
        return log_p_ref + log_det
    log_rho_theta_vmap = jax.jit(jax.vmap(log_rho_theta))

    x_jax      = jnp.array(eval_samples, dtype=jnp.float64)
    log_p_hat  = np.array(log_rho_theta_vmap(x_jax))         
    log_p_true = p_true.logpdf(eval_samples)                 

    kl = np.mean(log_p_true - log_p_hat)
    return float(kl)