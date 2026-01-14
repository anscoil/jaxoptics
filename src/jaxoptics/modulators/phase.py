import equinox as eqx
import jax.numpy as jnp
from typing import Tuple, Union

class Phase(eqx.Module):
    phasemask: jnp.ndarray
