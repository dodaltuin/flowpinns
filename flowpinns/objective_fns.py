# =========================================================
# Imports
# =========================================================
from jax import vmap
from jax.scipy.stats.norm import logpdf as norm_logpdf
from typing import Callable, Dict, Tuple
from jax import Array

FloatArray = Array

# =========================================================
# ELBO Objective Function
# =========================================================
def get_elbo_objective_fn(
    settings_dict: Dict,
    flow_sampler,
    upred_vmap: Callable,
    fpred_vmap: Callable,
    fixed_pinn_params: Dict | None = None,
    fixed_yf_noise_std: Array | None = None,
) -> Tuple[Callable, Callable]:
    """
    Construct the ELBO objective function and unnormalised posterior
    for variational inference of PDE parameters using a normalising flow.

    This is used in Stage One and Stage Three training of the VIF pipeline.

    Inputs
    ----------
    settings_dict : dict
        Configuration dictionary containing:
            - noise_transform : Callable mapping likelihood parameters → noise stds
            - theta_transform : Callable mapping log-theta → theta space
            - prior_mean : Array, shape (D_theta,)
            - prior_std : Array, shape (D_theta,)
            - n_theta_samples_elbo : int, number of flow samples per ELBO evaluation

    flow_sampler : object
        FlowSampler providing method:
        `posterior_sample(key, n_samples, flow_params)`

    upred_vmap : Callable
        Vectorised PINN prediction function for observed data `u`.
        Signature: `(pinn_params, Xu, log_theta) → yu_pred`
        Output shape: (n_theta_samples, n_points_u)

    fpred_vmap : Callable
        Vectorised PINN prediction function for PDE residual / RHS data `f`.
        Output shape: (n_theta_samples, n_points_f)

    fixed_pinn_params : dict, optional
        If provided, overrides `params['pinn_params']` for fixed PINN parameters.

    fixed_yf_noise_std : Array, optional
        If provided, overrides the learned PDE residual noise in the likelihood.

    Returns
    -------
    elbo_obj_fn : Callable
        Function computing the negative ELBO (for minimisation):
        `elbo_obj_fn(params, key, data_batch) → (loss, (theta_mean, theta_std))`

    unnormalised_posterior : Callable
        Function computing the unnormalised log posterior for a given `log_theta`.
        Signature: `(log_theta, params, train_data) → log_posterior`
    """

    # Extract settings
    noise_transform = settings_dict["noise_transform"]
    theta_transform = settings_dict["theta_transform"]
    prior_mean = settings_dict["prior_mean"]
    prior_std = settings_dict["prior_std"]
    n_theta_samples = settings_dict["n_theta_samples_elbo"]

    # ---------------------------------------------------------
    # Unnormalised posterior evaluation
    # ---------------------------------------------------------
    def unnormalised_posterior(log_theta: Array, params: Dict, train_data) -> FloatArray:
        """
        Compute unnormalised log posterior for a given log-theta sample.

        Inputs
        ----------
        log_theta : Array, shape (D_theta,)
            Logarithm of PDE parameter vector.

        params : dict
            Dictionary of model parameters including PINN and likelihood params.

        train_data : object
            Batch of training data containing Xu, yu, Xf, yf.

        Returns
        -------
        log_post : float
            Scalar unnormalised log posterior.
        """

        # Resolve PINN parameters (fixed or from `params`)
        pinn_params = (
            params["pinn_params"] if fixed_pinn_params is None else fixed_pinn_params
        )

        # Compute observation and PDE residual noise
        yu_noise_std, yf_noise_std_learned = noise_transform(
            params["likelihood_params"]
        )
        yf_noise_std = (
            yf_noise_std_learned
            if fixed_yf_noise_std is None
            else fixed_yf_noise_std
        )

        # Compute PINN predictions
        yu_pred = upred_vmap(pinn_params, train_data.Xu, log_theta)  # shape: (n_theta_samples, n_points_u)
        yf_pred = fpred_vmap(pinn_params, train_data.Xf, log_theta)  # shape: (n_theta_samples, n_points_f)

        # Log-likelihood terms
        yu_log_likelihood = norm_logpdf(train_data.yu, yu_pred, scale=yu_noise_std).sum()
        yf_log_likelihood = norm_logpdf(train_data.yf, yf_pred, scale=yf_noise_std).sum()
        log_likelihood = yu_log_likelihood + yf_log_likelihood

        # Gaussian prior on log-theta
        log_prior = norm_logpdf(log_theta, loc=prior_mean, scale=prior_std).sum()

        return log_likelihood + log_prior

    # Vectorise posterior evaluation across multiple theta samples
    posterior_vmap = vmap(unnormalised_posterior, in_axes=(0, None, None))

    # ---------------------------------------------------------
    # ELBO objective function
    # ---------------------------------------------------------
    def elbo_obj_fn(params: Dict, key, data_batch) -> Tuple[Array, Tuple[Array, Array]]:
        """
        Monte Carlo estimate of negative ELBO for variational inference.

        Inputs
        ----------
        params : dict
            Model parameters including flow, PINN, and likelihood parameters.

        key : PRNGKey
            Random key for sampling from the flow.

        data_batch : object
            Training batch containing Xu, yu, Xf, yf.

        Returns
        -------
        loss : Array
            Negative ELBO (for minimisation).

        aux : tuple
            Tuple `(theta_mean, theta_std)` of sampled PDE parameters for monitoring.
        """

        # Sample log-theta from variational posterior (flow)
        log_theta_samples, log_det_jacobian = flow_sampler.posterior_sample(
            key, n_theta_samples, params["flow_params"]
        )  # shape: (n_theta_samples, D_theta)

        # Evaluate unnormalised log posterior for each sample
        log_posterior = posterior_vmap(log_theta_samples, params, data_batch)

        # Monte Carlo ELBO estimate
        elbo_estimate = (log_posterior + log_det_jacobian).mean()

        # Compute mean and std of transformed theta for monitoring
        theta_samples = theta_transform(log_theta_samples)  # shape: (n_theta_samples, D_theta)
        theta_mean = theta_samples.mean(axis=0)
        theta_std = theta_samples.std(axis=0)

        return -elbo_estimate, (theta_mean, theta_std)

    return elbo_obj_fn, unnormalised_posterior

# =========================================================
# PINN objective function
# =========================================================
def pinn_obj_fn(
    pinn_params: Dict,
    log_theta_samples: Array,  # shape: (n_theta_samples, D_theta)
    Xf_batch: Array,           # shape: (batch_size, D_in)
    yf_batch: Array,           # shape: (batch_size,)
    pred_fn_vmap2: Callable,
) -> Tuple[FloatArray, None]:
    """
    Compute the PINN loss over a batch of collocation points and sampled PDE parameters.

    Inputs
    ----------
    pinn_params : dict
        Parameters of the PINN model.

    log_theta_samples : Array, shape (n_theta_samples, D_theta)
        Sampled PDE parameters in log-space.

    Xf_batch : Array, shape (batch_size, D_in)
        Batch of collocation input points for PDE residual.

    yf_batch : Array, shape (batch_size,)
        Ground-truth PDE residual values.

    pred_fn_vmap2 : Callable
        Vectorised PINN prediction function over theta samples and collocation batch.

    Returns
    -------
    loss : float
        Mean squared error over theta samples and collocation points.

    aux : None
        Placeholder for compatibility with `elbo_obj_fn`.
    """

    # Predictions over theta samples and collocation points
    # Output shape: (n_theta_samples, batch_size)
    yf_pred = pred_fn_vmap2(pinn_params, Xf_batch, log_theta_samples)

    # Align ground truth with prediction dimensions
    # shape: (1, batch_size)
    yf_target = yf_batch[None, :]

    # Mean squared error across both axes
    loss = ((yf_target - yf_pred) ** 2).mean()

    # Return None as auxiliary output for ELBO compatibility
    return loss, None