import equinox as eqx
import jax
import jax.numpy as jnp
from ..fields import ScalarField
from ..modes import spatial_grid
from typing import Optional, Tuple, Union, Callable

def apply_phase_mask(u: jnp.ndarray, phase_mask: jnp.ndarray) -> jnp.ndarray:
    return ScalarField(u.electric * jnp.exp(1j * phase_mask), u.ds, u.wavelengths)

class Phase(eqx.Module):
    phase_mask: jnp.ndarray
    is_trainable: bool = eqx.field(static=True)
    
    def __init__(self, u: ScalarField,
                 init_fn: Optional[Callable] = None, trainable: bool = True):
        spatial_shape = u.shape[-u.ndim_spatial:]
        x, y = spatial_grid(spatial_shape, u.ds)
        if init_fn is None:
            self.phase_mask = jnp.zeros(u.electric.shape[-u.ndim_spatial:],
                                        dtype=u.electric.real.dtype)
        else:
            self.phase_mask = init_fn(x, y).astype(u.electric.real.dtype)
        self.is_trainable = trainable

    def __call__(self, u: jnp.ndarray) -> jnp.ndarray:
        if self.is_trainable:
            return apply_phase_mask(u, self.phase_mask)
        else:
            return apply_phase_mask(u, jax.lax.stop_gradient(self.phase_mask))
