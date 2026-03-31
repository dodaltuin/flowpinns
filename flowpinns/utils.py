# =========================================================
# Imports
# =========================================================
import jax
import jax.numpy as jnp
from jax.scipy.stats.norm import pdf as norm_pdf

from scipy.stats import differential_entropy
from scipy.stats import gaussian_kde

from dataclasses import dataclass

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

import os

from .data_utils import TestData, TrainData, DataLoader, PosteriorSamples

# =========================================================
# Typing
# =========================================================
from typing import Callable, Dict, Tuple
from jax import Array
FloatArray = Array

# =========================================================
# Positive transformation function
# =========================================================
def softplus(x: Array, inv: bool = False) -> Array:
    """
    Numerically stable softplus transform and its inverse.

    The softplus function is

        softplus(x) = log(1 + exp(x))

    which maps ℝ → (0, ∞). This is commonly used to enforce positivity
    constraints on parameters.

    Parameters
    ----------
    x : array_like
        Input array.

    inv : bool, default=False
        If False, compute the softplus transform.
        If True, compute the inverse softplus. This is useful for 
        initialising parameters

    Returns
    -------
    y : array_like
        Transformed array with the same shape as `x`.
    """
    if not inv:
        # Stable implementation provided by JAX
        return jax.nn.softplus(x)

    # Inverse softplus:
    # x = log(exp(y) - 1)
    # Use expm1 for numerical stability when y is small
    return jnp.log(jnp.expm1(x))

# =========================================================
# Evaluation metrics
# =========================================================
def RMSE(true: Array, pred: Array) -> float:
    """
    Compute Root Mean Square Error.

    Inputs
    ----------
    true : (n_points,)
    pred : (n_points,)

    Returns
    -------
    rmse : float
    """
    return jnp.sqrt(((true - pred) ** 2).mean())

# Vectorized RMSE over samples
RMSE_vmap = jax.vmap(RMSE, in_axes=[None, 0])

def MPL(us_true: Array, us_samples: Array) -> Array:
    """
    Mean Predictive Likelihood.

    Inputs
    ----------
    us_true : (n_points,)
    us_samples : (n_samples, n_points)

    Returns
    -------
    norm_values : (n_points_kept,)
    """
    samples_mean = us_samples.mean(0)
    samples_std = us_samples.std(0)

    # Keep points with positive variance
    keep_indices = samples_std > 0.0
    norm_values = norm_pdf(
        us_true[keep_indices],
        samples_mean[keep_indices],
        samples_std[keep_indices]
    )
    return norm_values

def NIPG(true_samples: Array, pred_samples: Array) -> float:
    """
    Normalized inner product between variances.

    Inputs
    ----------
    true_samples : (n_samples, n_points)
    pred_samples : (n_samples, n_points)

    Returns
    -------
    nipg_val : float (percentage)
    """
    true_var = jnp.var(true_samples, axis=0)
    pred_var = jnp.var(pred_samples, axis=0)

    inner_one = jnp.inner(true_var, pred_var)
    inner_two = jnp.inner(true_var, true_var)
    inner_thr = jnp.inner(pred_var, pred_var)

    nipg_val = inner_one / jnp.sqrt(inner_two * inner_thr)
    return 100.0 * nipg_val

def entropy_calc(theta_samples: Array) -> float:
    """
    Compute differential entropy of parameter samples.

    Inputs
    ----------
    theta_samples : (n_samples, n_params)

    Returns
    -------
    entropy : float
    """
    return differential_entropy(theta_samples).sum()

def kl_divergence(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    kde_bw: float,
    n_mc: int
) -> float:
    """
    Monte Carlo estimate of KL(P||Q) using KDE.

    Inputs
    ----------
    p_samples : (n_samples, n_dim)
    q_samples : (n_samples, n_dim)
    kde_bw : float
    n_mc : int

    Returns
    -------
    kl : float
    """
    p_kde = gaussian_kde(p_samples.T, bw_method=kde_bw)
    q_kde = gaussian_kde(q_samples.T, bw_method=kde_bw)

    np.random.seed(808)
    x = p_samples[np.random.choice(len(p_samples), n_mc, replace=True)]
    log_p = p_kde.logpdf(x.T)
    log_q = q_kde.logpdf(x.T)

    return np.mean(log_p - log_q)

def jensen_shannon_divergence(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    kde_bw: float = 1.0,
    n_mc: int = 10000
) -> float:
    """
    Monte Carlo estimate of Jensen-Shannon divergence.

    Inputs
    ----------
    p_samples : (n_samples, n_dim)
    q_samples : (n_samples, n_dim)
    kde_bw : float
    n_mc : int

    Returns
    -------
    jsd : float
    """
    # KDEs for P and Q
    p_kde = gaussian_kde(p_samples.T, bw_method=kde_bw)
    q_kde = gaussian_kde(q_samples.T, bw_method=kde_bw)

    # Approximate mixture distribution
    mix_samples = np.vstack([
        p_samples[np.random.choice(len(p_samples), n_mc // 2, replace=True)],
        q_samples[np.random.choice(len(q_samples), n_mc // 2, replace=True)]
    ])
    m_kde = gaussian_kde(mix_samples.T, bw_method=kde_bw)

    kl_pm = kl_divergence(p_samples, mix_samples, kde_bw, n_mc // 2)
    kl_qm = kl_divergence(q_samples, mix_samples, kde_bw, n_mc // 2)

    return 0.5 * (kl_pm + kl_qm)

# =========================================================
# Evaluation against ground truth
# =========================================================
def ground_truth_evaluation(
    us_true: Array,
    theta_true: Array,
    us_pred_samples: Array,
    theta_pred_samples: Array
) -> dict:
    """
    Evaluate predictive and parameter estimates against ground truth.

    Inputs
    ----------
    us_true : (n_points,)
    theta_true : (n_params,)
    us_pred_samples : (n_samples, n_points)
    theta_pred_samples : (n_samples, n_params)

    Returns
    -------
    stats_dict : dict
    """
    # PDE solution evaluation
    u_rmse = RMSE_vmap(us_true, us_pred_samples)
    u_mpl = MPL(us_true, us_pred_samples)
    u_rmse_mean, u_rmse_std = u_rmse.mean(), u_rmse.std()
    u_mpl_mean, u_mpl_std = u_mpl.mean(), u_mpl.std()

    # Parameter evaluation
    theta_rmse = RMSE_vmap(theta_true, theta_pred_samples)
    theta_entr = entropy_calc(theta_pred_samples)
    theta_rmse_mean, theta_rmse_std = theta_rmse.mean(), theta_rmse.std()

    print("Results evaluation against ground truth")
    print(f"u_rmse - mean/std     : {u_rmse_mean:.5f}/{u_rmse_std:.5f}")
    print(f"u_mpl  - mean/std     : {u_mpl_mean:.5f}/{u_mpl_std:.5f}")
    print(f"theta_rmse - mean/std : {theta_rmse_mean:.5f}/{theta_rmse_std:.5f}")
    print(f"theta_entropy         : {theta_entr:.5f}")

    stats_dict = {
        "u_rmse_mean": u_rmse_mean,
        "u_mpl_mean": u_mpl_mean,
        "theta_rmse_mean": theta_rmse_mean,
        "theta_entr": theta_entr
    }
    return stats_dict

# =========================================================
# Evaluation against gold standard samples
# =========================================================
def gold_standard_evaluation(
    us_gold_samples: Array,
    theta_gold_samples: Array,
    us_pred_samples: Array,
    theta_pred_samples: Array,
    kde_bw: float = 1.0
) -> dict:
    """
    Evaluate approximate posterior against gold standard MCMC.

    Inputs
    ----------
    us_gold_samples : (n_samples, n_points)
    theta_gold_samples : (n_samples, n_params)
    us_pred_samples : (n_samples, n_points)
    theta_pred_samples : (n_samples, n_params)
    kde_bw : float

    Returns
    -------
    stats_dict : dict
    """
    assert len(us_gold_samples.shape) == len(us_pred_samples.shape)
    assert len(theta_gold_samples.shape) == len(theta_pred_samples.shape)

    u_nipg = NIPG(us_gold_samples, us_pred_samples)
    theta_nipg = NIPG(theta_gold_samples, theta_pred_samples)
    theta_jsd = jensen_shannon_divergence(theta_gold_samples, theta_pred_samples, kde_bw)

    print("Results evaluation against gold standard")
    print(f"u_nipg     : {u_nipg:.2f}")
    print(f"theta_nipg : {theta_nipg:.2f}")
    print(f"theta_jsd  : {theta_jsd:.2f}")

    stats_dict = {
        "u_nipg": u_nipg,
        "theta_nipg": theta_nipg,
        "theta_jsd": theta_jsd
    }
    return stats_dict

# =========================================================
# Data loading and saving utilities
# =========================================================
def load_test_data(data_save_dir):
    
    Xs = jnp.load(f'{data_save_dir}/Xs.npy')
    us = jnp.load(f'{data_save_dir}/us.npy')

    return TestData(Xs, us)

def load_train_data(data_save_dir, Nu=50, noise_frac=None):
    
    Xs = jnp.load(f'{data_save_dir}/Xs.npy')
    us = jnp.load(f'{data_save_dir}/us.npy')

    Xf = jnp.load(f'{data_save_dir}/Xf.npy')
    yf = jnp.load(f'{data_save_dir}/yf.npy')

    Xu = jnp.load(f'{data_save_dir}/Xu.npy')
    
    if Xu.shape[0] < Nu:
        print(f'Xu.shape: {Xu.shape}, requested Nu: {Nu}')
        
    if noise_frac is None:
        yu = jnp.load(f'{data_save_dir}/yu.npy')
    else:
        yu_nf = jnp.load(f'{data_save_dir}/yu.npy')
        
        yu_noise = jnp.load(f'{data_save_dir}/yu_noise.npy')
        us_std = jnp.load(f'{data_save_dir}/us_std.npy')
        yu_noise_std = noise_frac*us_std

        yu = yu_nf + (yu_noise*yu_noise_std)

    train_data = TrainData(Xu[:Nu], yu[:Nu], Xf, yf)

    return train_data
    
def load_posterior_samples(dir_: str) -> PosteriorSamples:
    """
    Load posterior samples from disk.

    Inputs
    ----------
    dir_ : str

    Returns
    -------
    PosteriorSamples
    """
    us_samples = jnp.load(f"{dir_}/us_samples.npy")
    theta_samples = jnp.load(f"{dir_}/theta_samples.npy")
    return PosteriorSamples(us_samples, theta_samples)

def save_dict_to_txt(dict_: dict, dir_: str) -> None:
    """
    Save a dictionary to a text file.

    Inputs
    ----------
    dict_ : dict
    dir_ : str
    """
    with open(dir_, "w") as file:
        file.write(str(dict_))


@dataclass
class SaveDirs:
    base: str
    plots: str
    stats: str
    nn_params: str

    @classmethod
    def create(cls, base_dir: str):
        """
        Create directory structure and return SaveDirs object.
        """
        plots_dir = os.path.join(base_dir, "plots")
        stats_dir = os.path.join(base_dir, "summaryStats")
        nn_params_dir = os.path.join(base_dir, "neuralNetworkParameters")

        for d in [base_dir, plots_dir, stats_dir, nn_params_dir]:
            os.makedirs(d, exist_ok=True)

        return cls(
            base=base_dir,
            plots=plots_dir,
            stats=stats_dir,
            nn_params=nn_params_dir
        )

# =========================================================
# Plotting utilities
# =========================================================
THETA_SAMPLES_FIGSIZE_3 = (11, 3)
THETA_SAMPLES_FIGSIZE_2 = (4.5, 3)
US_PLOT_FIGSIZE = (8, 6)

def plot_theta_samples(theta_samples: Array, theta_true: Array = None, save_name: str = None) -> None:
    """
    Plot pairwise scatter of theta parameters.

    Inputs
    ----------
    theta_samples : (n_samples, n_theta)
    theta_true : (n_theta,) or None
    save_name : str or None
    """
    if theta_samples.shape[-1] ==2:
        figsize = THETA_SAMPLES_FIGSIZE_2
        pairs = [(0,1)]
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(theta_samples[:, 0], theta_samples[:, 1], 'bo', alpha=0.2)
        if theta_true is not None:
            ax.plot(theta_true[0], theta_true[1], 'ro')
        ax.set_xlabel(f"theta_{1}")
        ax.set_ylabel(f"theta_{2}")
    else:
        figsize = THETA_SAMPLES_FIGSIZE_3
        pairs = [(0, 1), (0, 2), (1, 2)]
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    
        for idx, (ii, jj) in enumerate(pairs):
            axes[idx].plot(theta_samples[:, ii], theta_samples[:, jj], 'bo', alpha=0.2)
            if theta_true is not None:
                axes[idx].plot(theta_true[ii], theta_true[jj], 'ro')
            axes[idx].set_xlabel(f"theta_{ii+1}")
            axes[idx].set_ylabel(f"theta_{jj+1}")
    
    if save_name is not None:
        plt.savefig(save_name)

def make_scatter_heatmap(ax: plt.Axes, X: Array, output: Array, title: str = None, vmin: float = None, vmax: float = None) -> None:
    """
    Scatter plot colored by output values with colorbar.

    Inputs
    ----------
    ax : matplotlib.axes.Axes
    X : (n_plot_points, 2)
    output : (n_plot_points,)
    title : str or None
    vmin : float
    vmax : float
    """
    plot_obj = ax.scatter(X[:, 0], X[:, 1], c=output, cmap="jet")
    add_colorbar(plot_obj, vmin=vmin, vmax=vmax)
    if title is not None:
        ax.set_title(title)

def add_colorbar(mappable, vmin: float = None, vmax: float = None):
    """
    Adds a colorbar to the axis of `mappable`, optionally setting vmin/vmax
    so multiple plots can share the same color scale.

    Inputs
    ----------
    mappable : matplotlib artist (scatter, imshow, etc.)
    vmin : float
    vmax : float

    Returns
    -------
    cbar : matplotlib.colorbar.Colorbar
    """
    ax = mappable.axes
    fig = ax.figure

    # Ensure vmin/vmax are numeric
    if vmin is None:
        vmin = np.min(mappable.get_array())
    if vmax is None:
        vmax = np.max(mappable.get_array())

    # Apply vmin/vmax to mappable
    mappable.set_clim(vmin, vmax)

    # Create the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(ax)
    return cbar

def plot_us_posterior(Xs_plot: Array, us_true: Array, us_pred_samples: Array, save_name: str = None) -> None:
    """
    Plot posterior prediction fields with mean, std, and loss.

    Inputs
    ----------
    Xs_plot : (n_points, dim)
    us_true : (n_points,)
    us_pred_samples : (n_samples, n_points)
    save_name : str or None
    """
    us_pred_mean = us_pred_samples.mean(0)
    us_pred_std = us_pred_samples.std(0)
    mean_loss_field = jnp.abs(us_true - us_pred_mean)

    # Create 2x2 subplot grid with shared x/y axes
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=US_PLOT_FIGSIZE, sharex=True, sharey=True)

    # Shared color scale for first row
    vmin = float(min(us_true.min(), us_pred_mean.min()))
    vmax = float(max(us_true.max(), us_pred_mean.max()))

    # First row: true and mean prediction
    make_scatter_heatmap(axs[0, 0], Xs_plot, us_true, "True solution", vmin=vmin, vmax=vmax)
    make_scatter_heatmap(axs[0, 1], Xs_plot, us_pred_mean, "Mean prediction", vmin=vmin, vmax=vmax)

    # Second row: loss and std
    make_scatter_heatmap(axs[1, 0], Xs_plot, mean_loss_field, f"Losses (mean:{mean_loss_field.mean():.2e})")
    make_scatter_heatmap(axs[1, 1], Xs_plot, us_pred_std, f"Prediction std")

    if save_name is not None:
        plt.savefig(save_name)