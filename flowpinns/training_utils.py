# =========================================================
# Imports
# =========================================================
import jax
import math
import jax.random as jr
import optax
from typing import Callable, Any, Tuple
from jax import Array

# =========================================================
# Learner class
# =========================================================
class Learner:
    """
    Generic training loop manager for PINN/flow models.

    Handles stepwise training, logging, and optimiser updates.
    """
    def __init__(
        self,
        params,
        train_step_fn: Callable,
        opt_state,
        rng,
        log_fn: Callable = None,
        log_fraction: float = 0.1
    ):
        """
        Initialise learner.

        Parameters
        ----------
        params : dict
            Model parameters.
        train_step_fn : Callable
            Function performing a single training step:
            `train_step_fn(opt_state, params, rng) → (new_opt_state, new_params, loss, aux)`
        opt_state : Optax state
            Initial optimiser state.
        rng : PRNGKey
            JAX random key.
        log_fn : Callable, optional
            Logging function:
            `log_fn(step, opt_state, params, loss, aux)`
        log_fraction : float, optional
            Fraction of steps for logging (default 0.1).
        """
        self.params = params
        self.train_step_fn = train_step_fn 
        self.opt_state = opt_state
        self.rng = rng
        self.log_fn = log_fn
        self.log_fraction = log_fraction
        self.loss = 0.
        self.n_steps = 0

    def train(self, n_steps: int):
        """
        Execute training loop.

        Parameters
        ----------
        n_steps : int
            Total number of training steps.
        """
        log_every = max(1, math.floor(n_steps * self.log_fraction))

        for step in range(n_steps):
            self.n_steps += 1
            self.rng, step_rng = jax.random.split(self.rng, 2)

            # Run one training step
            self.opt_state, self.params, self.loss, sample_stats = self.train_step_fn(
                self.opt_state, self.params, step_rng
            )

            # Logging at intervals
            if self.log_fn and (step % log_every == 0 or step == n_steps - 1):
                self.log_fn(self.n_steps, self.opt_state, self.params, self.loss, sample_stats)


# =========================================================
# Initialisation function for Learner
# =========================================================
def init_learner(
    stage: str,
    params: dict,
    loss_fn: Callable,
    learning_rate: float,
    rng: jr.PRNGKey,
    noise_transform: Callable,
    train_batch_loader: Callable,
    validation_batch: Tuple[Array, Array] = None,
    flow_sampler: Callable = None,
    log_fraction: float = 0.1
):
    """
    Initialise a Learner object for a given training stage.

    Handles stage-specific data handling:
        - Stage 1/3/4: Flow + PINN or Flow-only
        - Stage 2: PINN-only (collocation)

    Parameters
    ----------
    stage : str
        "stage1", "stage2", "stage3", or "stage4".
    params : dict
        Initial model parameters.
    loss_fn : Callable
        Loss/objective function.
    learning_rate : float
        Learning rate for Adam optimiser.
    rng : PRNGKey
        Random key for initialisation.
    noise_transform : Callable
        Transform function for likelihood noise parameters.
    train_batch_loader : Callable
        Function to sample a training batch. Stage 2 returns collocation points.
    validation_batch : tuple of Arrays, optional
        Validation data for logging.
    flow_sampler : Callable, optional
        Function returning PDE parameter samples (used in Stage 2).
    log_fraction : float, optional
        Fraction of steps to log.

    Returns
    -------
    learner : Learner
        Fully initialised Learner object ready for training.
    """

    # Initialise Adam optimiser
    optimiser = optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate)
    opt_state = optimiser.init(params)

    # Split RNG for learner and validation
    learner_rng, valid_rng = jr.split(rng, 2)

    # Compute loss and gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    # ---------------------------------------------------------
    # Training step function
    # ---------------------------------------------------------
    @jax.jit
    def train_step_fn(opt_state, params, rng_key):
        """
        Single training step.

        Stage-specific handling of batches and sampling:
            - Stage1/3/4: sample u/f batches
            - Stage2: sample collocation points + PDE parameters
        """
        rng_key, step_rng = jr.split(rng_key, 2)

        if stage.lower() in ["stage1", "stage3", "stage4"]:
            batch_rng, loss_rng = jr.split(step_rng, 2)
            data_batch = train_batch_loader(batch_rng)
            (loss, aux), grads = grad_fn(params, loss_rng, data_batch)

        elif stage.lower() == "stage2":
            batch_rng, p_rng = jr.split(step_rng, 2)
            Xf_batch, yf_batch = train_batch_loader(batch_rng)
            log_theta_samples = flow_sampler(p_rng)
            (loss, aux), grads = grad_fn(params, log_theta_samples, Xf_batch, yf_batch)

        else:
            raise ValueError(f"Unknown stage: {stage}")

        # Optimiser update
        updates, new_opt_state = optimiser.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_opt_state, new_params, loss, aux

    # ---------------------------------------------------------
    # Validation loss function
    # ---------------------------------------------------------
    if stage.lower() in ["stage1", "stage3", "stage4"]:
        @jax.jit
        def valid_loss_fn(params):
            return loss_fn(params, valid_rng, validation_batch)[0]
    elif stage.lower() == "stage2":
        batch_rng, p_rng = jr.split(valid_rng)
        Xf_valid, yf_valid = train_batch_loader(batch_rng)
        log_theta_samples = flow_sampler(p_rng)

        @jax.jit
        def valid_loss_fn(params):
            return loss_fn(params, log_theta_samples, Xf_valid, yf_valid)[0]
    else:
        raise ValueError(f"Unknown stage: {stage}")

    # ---------------------------------------------------------
    # Logging function
    # ---------------------------------------------------------
    def log_fn(step, opt_state, params, loss, sample_stats):
        """
        Stage-specific logging.
        """
        if stage.lower() in ["stage1", "stage3"]:
            valid_loss = valid_loss_fn(params)
            sample_mean, sample_std = sample_stats
            yu_std_i, yf_std_i = noise_transform(params['likelihood_params'])
            lr = opt_state.hyperparams['learning_rate']
            print(f"({step}), Obj/Valid: {loss:.2f}/{valid_loss:.2f}, lr: {lr:.1e}, "
                  f"sample mean/std: {sample_mean}/{sample_std}, noise_stds: {yu_std_i:.3e}/{yf_std_i:.3e}")

        elif stage.lower() == "stage2":
            valid_loss = valid_loss_fn(params)
            lr = opt_state.hyperparams['learning_rate']
            print(f"({step}), Obj/Valid: {loss:.3e}/{valid_loss:.3e}, lr: {lr:.1e}")

    # Initialise Learner object
    learner = Learner(
        params=params,
        train_step_fn=train_step_fn,
        opt_state=opt_state,
        rng=learner_rng,
        log_fn=log_fn,
        log_fraction=log_fraction
    )

    return learner