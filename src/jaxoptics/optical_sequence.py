import equinox as eqx
import jax.numpy as jnp
from typing import Tuple, Union, Sequence, Callable

class OpticalSequence(eqx.Module):
    optical_components: Sequence[Callable]

    def __init__(self, *components: Callable):
        self.optical_components = components

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for component in self.optical_components:
            x = component(x)
        return x
