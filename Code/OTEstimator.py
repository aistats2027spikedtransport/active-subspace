import ot
import jax
import jax.numpy as jnp
import numpy as np
import ott
from ott.geometry import pointcloud
from ott.solvers import linear
from ott.solvers.linear import sinkhorn
from ott.tools import sinkhorn_divergence
from scipy.special import logsumexp

## neural OT from OTT
from ott.neural.methods import neuraldual
from ott.neural.networks import potentials
from ott.neural.networks import icnn
from ott.datasets import Dataset
import dataclasses
from scipy.stats import gaussian_kde  as orig_kde
import ott.utils as utils

import optax
from jax import random, grad, vmap
from jax.scipy.optimize import minimize

from typing import Iterator, Literal, NamedTuple, Optional, Tuple

class EntropicTransport:
    def __init__(self,dx,estimator,**args):
        self.dx  = dx
        if estimator == 'OTT':
            self.solver = EntropicTransport_OTT(**args)
        else:
            raise ValueError('Estimator is not yet implemented')

    def fit(self, source, target, **args):
        # check inputs
        assert(source.shape[1] == self.dx)
        assert(source.shape[1] == target.shape[1])
        # solve transport problem and save potentials/map
        self.solver.estimate(source, target, **args)

    def evaluate(self, source_new, **args):
        # check inputs
        assert(source_new.shape[1] == (self.dx))
        return self.solver.evaluate_map(source_new,**args)
    
    def inverse(self, target_new, **args):
        # check inputs
        assert(target_new.shape[1] == self.dx)
        # scale data
        x0 = self.scale_data(np.random.randn(target_new.shape[0], target_new.shape[1]))
        return self.solver.inverse_map(target_new,x0,**args)

    def grad_map(self, source_new):
        return self.solver.grad_map(source_new)
    
class EntropicTransport_OTT:
    def __init__(self, eps):
        self.eps = eps

    def estimate(self, source, target, tol=1e-3, max_iter=5000):
        # check inputs
        if source.shape[1] != target.shape[1]:
            raise ValueError("Source and target must have same dimensionality")
        # define geometry
        geom_data = pointcloud.PointCloud(source, target, epsilon=self.eps)
        # solve EOT and save potentials
        out = linear.solve(geom_data, threshold=tol, max_iterations=max_iter)
        self.potentials = out.to_dual_potentials()
        self.gYjs = out.potentials[1]
        self.target = target

    def evaluate_map(self, source_new):
        return self.potentials.transport(source_new)

    def grad_map(self, source_new):
        single_transport = lambda x: self.potentials.transport(x[None])[0]
        J = jax.vmap(jax.jacobian(single_transport))(source_new)  # (N, D, D)
        # then take log|det| of each:
        _, log_det = jnp.linalg.slogdet(J)  # (N,)
        return log_det
            
@dataclasses.dataclass
class KDE:
    """A mixture of Gaussians.
  
    Args:
      name: the name specifying the centers of the mixture components:
  
        - ``simple`` - data clustered in one center,
        - ``circle`` - two-dimensional Gaussians arranged on a circle,
        - ``square_five`` - two-dimensional Gaussians on a square with
          one Gaussian in the center, and
        - ``square_four`` - two-dimensional Gaussians in the corners of a
          rectangle
  
      batch_size: batch size of the samples
      rng: initial PRNG key
      scale: scale of the Gaussian means
      std: the standard deviation of the individual Gaussian samples
    """
    data: np.array
    batch_size: int
    rng: jax.Array
  
    def __post_init__(self) -> None:
        self.kde = orig_kde(self.data.T, bw_method=0.00)

    def __iter__(self) -> Iterator[jnp.array]:
        """Random sample generator from Gaussian mixture.
        Returns:
          A generator of samples from the Gaussian mixture.
        """
        return self._create_sample_generators()
    
    def _create_sample_generators(self) -> Iterator[jnp.array]:
        rng = self.rng
        while True:
          samples = self.kde.resample(self.batch_size).T
          yield samples

def minibatch_samplers(
    source_data: np.array,
    target_data: np.array,
    train_batch_size: int = 256,
    rng: Optional[jax.Array] = None,
) -> Tuple[Dataset, Dataset, int]:
    """Gaussian samplers.
    Args:
      name_source: name of the source sampler
      name_target: name of the target sampler
      train_batch_size: the training batch size
      valid_batch_size: the validation batch size
      rng: initial PRNG key
    Returns:
      The dataset and dimension of the data.
    """
    rng = utils.default_prng_key(rng)
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
    train_dataset = Dataset(
        source_iter=iter(
            KDE(source_data, batch_size=train_batch_size, rng=rng1)
        ),
        target_iter=iter(
            KDE(target_data, batch_size=train_batch_size, rng=rng2)
        )
    )
    # valid_dataset = Dataset(
    #     source_iter=iter(
    #         KDE(source_data, batch_size=valid_batch_size, rng=rng3)
    #     ),
    #     target_iter=iter(
    #         KDE(target_data, batch_size=valid_batch_size, rng=rng4)
    #     )
    # )
    return train_dataset #, valid_dataset

def training_callback(step, learned_potentials):
    if step % 50 == 0:
        print(step)
