# =========================================================
# Imports
# =========================================================
import jax.numpy as jnp
import jax.random as jr
from dataclasses import dataclass

Array = jnp.ndarray

# =========================================================
# Data classes
# =========================================================

@dataclass
class TestData:
    """
    Test data container.

    Inputs
    ----------
    Xs : jnp.ndarray, shape (n_points, dim)
        Input coordinates or features for testing.

    us : jnp.ndarray, shape (n_points,)
        Observed solution values at the points.
    """
    Xs: Array        # Shape: (n_points, dim)
    us: Array        # Shape: (n_points,)

@dataclass
class TrainData:
    """
    Training data container.

    Inputs
    ----------
    Xu : jnp.ndarray, shape (n_train_u, dim)
        Input locations for observed data.

    yu : jnp.ndarray, shape (n_train_u,)
        Observed data values at Xu.

    Xf : jnp.ndarray, shape (n_train_f, dim)
        Input locations for collocation / PDE residual points.

    yf : jnp.ndarray, shape (n_train_f,)
        PDE residual targets at Xf.
    """
    Xu: Array        # Shape: (n_train_u, dim)
    yu: Array        # Shape: (n_train_u,)
    Xf: Array        # Shape: (n_train_f, dim)
    yf: Array        # Shape: (n_train_f,)

@dataclass
class PosteriorSamples:
    """
    Container for posterior samples of solution and parameters.

    Inputs
    ----------
    us_samples : jnp.ndarray, shape (n_samples, n_points)
        Posterior samples of solution fields.

    theta_samples : jnp.ndarray, shape (n_samples, n_params)
        Posterior samples of model parameters.
    """
    us_samples: Array       # Shape: (n_samples, n_points)
    theta_samples: Array    # Shape: (n_samples, n_params)

# =========================================================
# Data Loader
# =========================================================
class DataLoader:
    """
    Data loader for batching training data.

    Supports:
        - Random mini-batches of observed and collocation points (Stage 1)
        - Fixed batches of collocation points (Stage 2)
        - Stage 2 collocation batches for PDE residual evaluation
    """
    def __init__(self, train_data, Nbatch_s1, Nbatch_s2):
        """
        Initialise the data loader.

        Inputs
        ----------
        train_data : TrainData
            Full training dataset.

        Nbatch_s1 : int
            Batch size for stage 1 training.

        Nbatch_s2 : int
            Batch size for stage 2 training.
        """
        self.Xu = train_data.Xu
        self.yu = train_data.yu
        self.Xf = train_data.Xf
        self.yf = train_data.yf

        self.Nu = self.Xu.shape[0]      # Total number of observed points
        self.Nf = self.Xf.shape[0]      # Total number of collocation points
        self.Nbatch_s1 = Nbatch_s1
        self.Nbatch_s2 = Nbatch_s2
        

    def get_batch(self, rng_key, Nbatch_u=None, Nbatch_f=None):
        """
        Get a random batch of observed and collocation points for training (Stage 1).

        Inputs
        ----------
        rng_key : jax.random.PRNGKey
            Random key for reproducible sampling.

        Nbatch_u : int or None, default=None
            Number of observed points in batch. If None, defaults to Nbatch_s1.

        Nbatch_f : int or None, default=None
            Number of collocation points in batch. If None, defaults to Nbatch_s1.

        Returns
        ----------
        TrainData
            Mini-batch with selected Xu, yu, Xf, yf.

        Notes
        ----------
        Shapes:
            Xu_batch : (Nbatch_u, dim)
            yu_batch : (Nbatch_u,)
            Xf_batch : (Nbatch_f, dim)
            yf_batch : (Nbatch_f,)
        """
        if Nbatch_u is None: Nbatch_u = self.Nbatch_s1
        if Nbatch_f is None: Nbatch_f = self.Nbatch_s1

        u_rng, f_rng = jr.split(rng_key, 2)
        
        # Sample indices without replacement
        u_indices = jr.choice(u_rng, self.Nu, shape=(Nbatch_u,), replace=False)
        f_indices = jr.choice(f_rng, self.Nf, shape=(Nbatch_f,), replace=False)
        
        return TrainData(
            Xu=self.Xu[u_indices],
            yu=self.yu[u_indices],
            Xf=self.Xf[f_indices],
            yf=self.yf[f_indices]
        )

    def get_fixed_batch(self, rng_key, Nbatch_f=None):
        """
        Get a fixed batch of collocation points for Stage 2 training.

        Inputs
        ----------
        rng_key : jax.random.PRNGKey
            Random key for reproducible sampling.

        Nbatch_f : int or None, default=None
            Number of collocation points in batch. If None, defaults to Nbatch_s2.

        Returns
        ----------
        TrainData
            Batch with full Xu, yu and selected collocation points Xf, yf.

        Notes
        ----------
        Shapes:
            Xu_full : (Nu, dim)
            yu_full : (Nu,)
            Xf_batch : (Nbatch_f, dim)
            yf_batch : (Nbatch_f,)
        """
        if Nbatch_f is None: Nbatch_f = self.Nbatch_s2

        # Sample collocation indices without replacement
        f_indices = jr.choice(rng_key, self.Nf, shape=(Nbatch_f,), replace=False)
        
        return TrainData(
            Xu=self.Xu,   # Full training set of observed points
            yu=self.yu,
            Xf=self.Xf[f_indices],
            yf=self.yf[f_indices]
        )

    def get_colloc_batch(self, rng_key, Nbatch_f=256):
        """
        Get a batch of collocation points for Stage 2 training (Section 4.1).

        Inputs
        ----------
        rng_key : jax.random.PRNGKey
            Random key for reproducible sampling.

        Nbatch_f : int, default=256
            Number of collocation points to sample.

        Returns
        ----------
        Xf_batch : jnp.ndarray, shape (Nbatch_f, dim)
            Collocation input points.

        yf_batch : jnp.ndarray, shape (Nbatch_f,)
            PDE residual targets at collocation points.
        """
        # Sample collocation indices without replacement
        f_indices = jr.choice(rng_key, self.Nf, shape=(Nbatch_f,), replace=False)
        return self.Xf[f_indices], self.yf[f_indices]