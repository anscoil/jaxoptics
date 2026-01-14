import equinox as eqx
import jax.numpy as jnp
from typing import Tuple, Union

class ASProp(eqx.Module):
    kernel: jnp.ndarray
