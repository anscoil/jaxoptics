import equinox as eqx
import jax.numpy as jnp
from typing import Tuple, Union

class ScalarField(eqx.Module):
    """Scalar optical field.
    
    Convention: (batch_dims..., spatial_dims...)
    - 1D: (batch..., nx)
    - 2D: (batch..., nx, ny)
    
    Args:
        electric: complex array with 1-2 trailing spatial dimensions
        ds: (dx,) or (dx, dy) spatial sampling
        wavelengths: scalar or array compatible with batch dimensions
    """
    electric: jnp.ndarray
    ds: Tuple[float, ...] = eqx.field(static=True)
    wavelengths: jnp.ndarray
    
    def __init__(self, 
                 electric: jnp.ndarray,
                 ds: Union[Tuple[float], Tuple[float, float]],
                 wavelengths: Union[float, jnp.ndarray]):
        
        # Validate ds length
        n_spatial = len(ds)
        if n_spatial not in (1, 2):
            raise ValueError(f"ds must have length 1 or 2, got {n_spatial}")
        
        # Validate electric dimensions
        if electric.ndim < n_spatial:
            raise ValueError(f"electric has {electric.ndim}D but need at least {n_spatial}D")
        
        self.electric = electric
        self.ds = tuple(ds)
        
        # Batch dimensions
        batch_shape = electric.shape[:-n_spatial]
        
        # Handle wavelengths
        wl = jnp.atleast_1d(jnp.asarray(wavelengths, dtype=float))
        
        # Broadcast wavelengths to batch shape
        if wl.size == 1:
            # Scalar: broadcast to batch shape
            self.wavelengths = jnp.full(batch_shape, wl[0])
        else:
            # Check compatibility and broadcast
            wl = wl.reshape(wl.shape)  # Keep original shape
            try:
                self.wavelengths = jnp.broadcast_to(wl, batch_shape)
            except ValueError:
                raise ValueError(
                    f"wavelengths shape {wl.shape} incompatible "
                    f"with batch dimensions {batch_shape}"
                )
    
    @property
    def shape(self):
        return self.electric.shape
    
    @property
    def ndim_spatial(self):
        return len(self.ds)
    
    @property
    def batch_shape(self):
        return self.shape[:-self.ndim_spatial]
