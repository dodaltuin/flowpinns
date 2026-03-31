# =========================================================
# Imports
# =========================================================
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence, Callable

# =========================================================
# Simple parameterised PINN (https://arxiv.org/abs/2408.09446)
# We use the simple approach of appending theta to x below, 
# as we found it gave essentially the same results as the
# seperate encode / decode structure proposed in the above paper
# =========================================================
class SimpleP2INN(nn.Module):
    """
    Simple fully-connected PINN/NN module with optional adaptive output scaling.

    Attributes
    ----------
    features : Sequence[int]
        List of hidden layer sizes.
    adf : Callable, optional
        Adaptive scaling function applied to output (e.g., for PDE residuals).
    mean_fn : Callable, optional
        Mean function for output offset when using adaptive scaling.
    """
    features: Sequence[int]
    adf: Callable = None
    mean_fn: Callable = lambda x: 0.

    @nn.compact
    def __call__(self, loc, theta):
        """
        Forward pass of the network.

        Parameters
        ----------
        loc : Array, shape (D_in,)
            Input coordinates (e.g., spatial locations).
        theta : Array, shape (D_theta,)
            PDE parameter vector.

        Returns
        -------
        output : Array, shape ()
            Predicted value or scaled prediction.
        """
        # Concatenate input location and PDE parameters
        z = jnp.concatenate([loc, theta])  # shape: (D_in + D_theta,)

        # Hidden layers with tanh activation (except last layer)
        for i, feat in enumerate(self.features):
            z = nn.Dense(feat)(z)
            if i != len(self.features) - 1:
                z = nn.tanh(z)

        # Apply adaptive scaling if provided
        if self.adf is None:
            return z.squeeze()
            
        return (self.mean_fn(loc) + self.adf(loc)*z).squeeze()