# =========================================================
# Imports
# =========================================================
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence, Callable, Optional

# =========================================================
# Masked dense layer for autoregressive networks
# =========================================================
class MaskedDense(nn.Module):
    """
    Dense layer with masked weights to enforce an autoregressive structure.

    Inputs
    ----------
    features : int
        Number of output features.

    mask : jnp.ndarray, shape (input_dim, output_dim)
        Binary mask to enforce autoregressive dependencies.

    use_bias : bool, default=True
        Whether to include a bias term.
    """
    features: int
    mask: jnp.ndarray
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        """
        Forward pass through the masked dense layer.

        Inputs
        ----------
        x : jnp.ndarray, shape (..., input_dim)

        Returns
        ----------
        y : jnp.ndarray, shape (..., features)
        """
        # Initialise weight matrix
        kernel = self.param(
            'kernel', 
            nn.initializers.lecun_normal(), 
            (x.shape[-1], self.features)
        )
        
        # Apply mask to enforce autoregressive structure
        masked_kernel = kernel * self.mask

        # Linear transformation
        y = jnp.dot(x, masked_kernel)
        
        if self.use_bias:
            # Add bias term
            bias = self.param('bias', nn.initializers.zeros, (self.features,))
            y += bias
            
        return y

# =========================================================
# Mask creation for autoregressive layers
# =========================================================
def create_mask(
    input_dim: int,
    output_dim: int,
    mask_type: str,
    autoregressive_dim: int
) -> jnp.ndarray:
    """
    Create masks for autoregressive connections in a neural network.

    Inputs
    ----------
    input_dim : int
        Number of inputs to the layer.

    output_dim : int
        Number of outputs from the layer.

    mask_type : str
        Type of mask: 'input', 'hidden', or 'output'.

    autoregressive_dim : int
        Dimension of autoregressive variables.

    Returns
    ----------
    mask : jnp.ndarray, shape (input_dim, output_dim)
    """
    mask = jnp.ones((input_dim, output_dim))
    
    if mask_type == 'input':
        # Input layer: allow connections only from previous variables
        for i in range(output_dim):
            for j in range(input_dim):
                if j >= autoregressive_dim:
                    # Allow full connection for non-autoregressive inputs
                    continue
                mask = mask.at[j, i].set(1 if i >= j else 0)
                
    elif mask_type == 'hidden':
        # Hidden layers: standard autoregressive mask
        for i in range(output_dim):
            for j in range(input_dim):
                mask = mask.at[j, i].set(1 if i > j else 0)
                
    elif mask_type == 'output':
        # Output layer: block autoregressive mask
        params_per_var = output_dim // autoregressive_dim
        for i in range(output_dim):
            var_idx = i // params_per_var
            for j in range(input_dim):
                mask = mask.at[j, i].set(1 if j < var_idx else 0)
                
    return mask

# =========================================================
# Masked autoregressive MLP
# =========================================================
class MaskedAutoregressiveMLP(nn.Module):
    """
    Masked autoregressive MLP with configurable architecture.

    Inputs
    ----------
    input_dim : int
        Dimension of input variables.

    hidden_dims : Sequence[int]
        Dimensions of hidden layers.

    output_params : int, default=2
        Number of output parameters per input dimension.

    use_bias : bool, default=True
        Whether to include bias terms.

    activation : Callable, default=nn.relu
        Activation function for hidden layers.
    """
    input_dim: int
    hidden_dims: Sequence[int]
    output_params: int = 2
    use_bias: bool = True
    activation: Callable = nn.relu

    def setup(self):
        """
        Initialise masked layers and masks.
        """
        # Layer sizes: input + hidden + output
        layer_sizes = [self.input_dim] + list(self.hidden_dims) + [self.input_dim * self.output_params]
        
        # Create masks for each layer
        self.masks = [
            create_mask(
                layer_sizes[i], 
                layer_sizes[i+1], 
                'input' if i == 0 else ('output' if i == len(layer_sizes)-2 else 'hidden'),
                self.input_dim
            )
            for i in range(len(layer_sizes)-1)
        ]
        
        # Initialise masked dense layers
        self.layers = [
            MaskedDense(
                features=layer_sizes[i+1],
                mask=self.masks[i],
                use_bias=self.use_bias
            )
            for i in range(len(layer_sizes)-1)
        ]

    def __call__(self, x):
        """
        Forward pass through the masked autoregressive MLP.

        Inputs
        ----------
        x : jnp.ndarray, shape (..., input_dim)

        Returns
        ----------
        x : jnp.ndarray, shape (..., input_dim, output_params)
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                # Apply activation to all but the last layer
                x = self.activation(x)
        
        # Reshape output to have separate parameter dimension
        output_shape = x.shape[:-1] + (self.input_dim, self.output_params)
        return x.reshape(output_shape)

# =========================================================
# Inverse Autoregressive Flow (IAF)
# =========================================================
class IAF(nn.Module):
    """
    Inverse Autoregressive Flow for flexible density modelling.

    Inputs
    ----------
    depth : int
        Number of hidden layers in each MLP 
    
    width : int
        Hidden layer width for each autoregressive MLP.

    num_flows : int
        Number of sequential flow transformations ($K$ in Eq.3 of the paper)

    D_theta : int
        Dimension of input latent variable z.

    theta_transform : Callable
        Transformation function for scale parameter (e.g., softplus).
    """
    depth : int
    width : int
    num_flows: int
    D_theta: int
    theta_transform: Callable
    
    def setup(self):
        """
        Initialise autoregressive flows.
        """
        # Create a list of masked autoregressive MLPs for the flow
        self.flows = [
            MaskedAutoregressiveMLP(
                self.D_theta,
                [self.width] * self.depth
            )
            for _ in range(self.num_flows)
        ]
    
    def __call__(self, z):
        """
        Forward pass through the IAF.

        Inputs
        ----------
        z : jnp.ndarray, shape (batch_size, D_theta)

        Returns
        ----------
        x : jnp.ndarray, shape (batch_size, D_theta)
        log_det_jacobian : jnp.ndarray, shape (batch_size,)
        """
        x = z
        log_det_jacobian = 0.0

        # Sequentially apply flows
        for flow in self.flows:
            # Compute autoregressive parameters: mu and alpha
            AA = flow(x)                  # shape (..., D_theta, output_params)
            mu = AA[:, :, 0]              # shape (..., D_theta)
            alpha = AA[:, :, 1]           # shape (..., D_theta)
            sigma = self.theta_transform(alpha)  # Ensure positivity

            # Update x using IAF transformation
            x = x * sigma + mu

            # Accumulate log-determinant of Jacobian
            log_det_jacobian += jnp.sum(jnp.log(sigma), axis=-1)

        return x, log_det_jacobian
            
    def inverse(self, z):
        """
        Inverse pass through the IAF (used for density evaluation).

        Inputs
        ----------
        z : jnp.ndarray, shape (batch_size, D_theta)

        Returns
        ----------
        x : jnp.ndarray, shape (batch_size, D_theta)
        """
        # Apply flows in reverse order
        for flow in reversed(self.flows):
            mu, sigma = flow(z)
            z = z * sigma + mu
        return z