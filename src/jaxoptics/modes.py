import jax.numpy as jnp
from typing import Tuple, Union
from scipy.special import hermite
import math

def spatial_grid(shape: Tuple[int, ...], 
                 ds: Tuple[float, ...]) -> Tuple[jnp.ndarray, ...]:
    """Generate centered spatial grid.
    
    Returns:
        (x,) for 1D or (x, y) for 2D with meshgrid
    """
    grids = []
    for n, d in zip(shape, ds):
        axis = (jnp.arange(n) - n // 2) * d
        grids.append(axis)
    
    if len(grids) == 1:
        return (grids[0],)
    else:
        return jnp.meshgrid(*grids, indexing='ij')


class Gaussian:
    """Gaussian beam generator."""
    
    def __init__(self, w0: float, dtype=jnp.complex64):
        """
        Args:
            w0: beam waist (1/e² radius)
        """
        self.w0 = w0
        self.dtype = dtype
    
    def __call__(self, shape: Tuple[int, int], 
                 ds: Tuple[float, float]) -> jnp.ndarray:
        """Generate Gaussian field.
        
        Returns:
            (nx, ny) complex array
        """
        x, y = spatial_grid(shape, ds)
        r2 = x**2 + y**2
        return jnp.exp(-r2 / self.w0**2).astype(self.dtype)


class HermiteGaussian:
    """Hermite-Gaussian mode generator."""
    
    def __init__(self, w0: float, m: int = 0, n: int = 0, dtype=jnp.complex64):
        """
        Args:
            w0: beam waist
            m, n: mode orders (x, y)
        """
        self.w0 = w0
        self.m = m
        self.n = n
        self.dtype = dtype
        
        # Precompute Hermite polynomials
        self.Hm = hermite(m)
        self.Hn = hermite(n)
    
    def __call__(self, shape: Tuple[int, int],
                 ds: Tuple[float, float]) -> jnp.ndarray:
        """Generate HG mode.
        
        Returns:
            (nx, ny) complex array
        """
        x, y = spatial_grid(shape, ds)
        
        # Normalized coordinates
        u = jnp.sqrt(2) * x / self.w0
        v = jnp.sqrt(2) * y / self.w0
        
        # HG mode
        gaussian = jnp.exp(-(x**2 + y**2) / self.w0**2)
        hg = self.Hm(u) * self.Hn(v) * gaussian
        
        # Normalization
        norm = 1.0 / jnp.sqrt(2**(self.m + self.n) *
                              math.factorial(self.m) * 
                              math.factorial(self.n))
        
        return jnp.array(hg * norm, dtype=self.dtype)
