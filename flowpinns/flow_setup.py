# =========================================================
# Imports
# =========================================================
import jax.random as jr
from .inverse_autoregressive_flow import IAF

# =========================================================
# Flow Sampler
# =========================================================
class FlowSampler:
    """
    Wrapper for prior and flow-based posterior sampling.

    Supports:
        - Gaussian prior sampling
        - Posterior sampling via an inverse autoregressive flow (IAF)
    """
    def __init__(self, flow_model, prior_mean, prior_std, theta_dim):
        """
        Initialise the sampler.

        Inputs
        ----------
        flow_model : callable Flax/JAX model
            Flow model supporting .apply(params, x).

        prior_mean : jnp.ndarray, shape (theta_dim,)
            Mean vector of the Gaussian prior for PDE parameters.

        prior_std : jnp.ndarray, shape (theta_dim,)
            Standard deviation vector of the Gaussian prior.

        theta_dim : int
            Dimension of the PDE parameter vector theta.
        """
        self.flow_model = flow_model
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.dim = theta_dim

    def prior_sample(self, key, n_samples):
        """
        Draw samples from the Gaussian prior.

        Inputs
        ----------
        key : jax.random.PRNGKey
            Random key for reproducible sampling.

        n_samples : int
            Number of samples to draw.

        Returns
        ----------
        prior_samples : jnp.ndarray, shape (n_samples, theta_dim)
            Samples from N(prior_mean, prior_std^2).
        """
        # Sample from standard normal and scale/shift to prior
        base_samples = jr.normal(key, (n_samples, self.dim))
        return base_samples * self.prior_std + self.prior_mean

    def posterior_sample(self, key, n_samples, flow_params):
        """
        Draw samples from the flow-based posterior.

        Inputs
        ----------
        key : jax.random.PRNGKey
            Random key for reproducible sampling.

        n_samples : int
            Number of samples to draw.

        flow_params : dict
            Parameters of the trained flow model.

        Returns
        ----------
        flow_samples : jnp.ndarray, shape (n_samples, theta_dim)
            Samples transformed by the inverse autoregressive flow.
        """
        # First, draw samples from the Gaussian prior
        prior_samples = self.prior_sample(key, n_samples)

        # Transform prior samples through the flow model to obtain posterior samples
        flow_samples = self.flow_model.apply(flow_params, prior_samples)
        
        return flow_samples

# =========================================================
# Flow Initialisation
# =========================================================
def initialise_flow(rng_key, settings_dict, n_prior_samples=1000):
    """
    Initialise a flow model and corresponding FlowSampler.

    Inputs
    ----------
    rng_key : jax.random.PRNGKey
        Random key for initialisation.

    settings_dict : dict
        Dictionary containing flow and prior settings:
            - flow_depth : int, hidden depth of each MLP in the IAF
            - flow_width : int, hidden width of each MLP
            - n_flows : int, number of flow transformations
            - D_theta : int, dimension of PDE parameter vector
            - theta_transform : callable, e.g., softplus for positive scale
            - prior_mean : jnp.ndarray, shape (D_theta,)
            - prior_std : jnp.ndarray, shape (D_theta,)

    n_prior_samples : int, default=1000
        Number of prior samples to draw for initialisation.

    Returns
    ----------
    flow_sampler : FlowSampler
        Object for drawing prior and posterior samples.

    flow_params_init : dict
        Initial parameters of the flow model.
    """
    # Initialise the IAF model
    maf_model = IAF(
        depth=settings_dict['flow_depth'],
        width=settings_dict['flow_width'],
        num_flows=settings_dict['n_flows'],
        D_theta=settings_dict['D_theta'],
        theta_transform=settings_dict['theta_transform']
    )

    # Initialise FlowSampler wrapper
    flow_sampler = FlowSampler(
        flow_model=maf_model,
        prior_mean=settings_dict['prior_mean'],
        prior_std=settings_dict['prior_std'],
        theta_dim=settings_dict['D_theta']
    )

    # Split RNG for reproducibility
    use_key, rng_key = jr.split(rng_key, 2)
    
    # Draw prior samples for initialisation
    prior_samples = flow_sampler.prior_sample(use_key, n_prior_samples)

    # Split RNG and initialise flow parameters
    use_key, rng_key = jr.split(rng_key, 2)
    flow_params_init = maf_model.init(use_key, prior_samples)

    # Sanity check: sample from posterior with initial flow parameters
    use_key, rng_key = jr.split(rng_key, 2)
    flow_samples, _ = flow_sampler.posterior_sample(use_key, n_prior_samples, flow_params_init)
    assert flow_samples.shape == prior_samples.shape  # Shape: (n_prior_samples, D_theta)

    return flow_sampler, flow_params_init