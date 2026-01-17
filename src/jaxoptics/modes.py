import jax.numpy as jnp
from typing import Optional, Tuple, Union, Sequence, Callable
from scipy.special import hermite
from scipy.special import factorial

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


class GridLayout:
    def __init__(self, nx: int, ny: int, px: float, py: float, 
                 offset: Tuple[float, float] = (0, 0), 
                 rotation: float = 0):
        """Centered grid layout
        nx, ny: number of points
        px, py: pitch
        rotation: rotation angle in radians
        """
        positions = []
        for i in range(nx):
            for j in range(ny):
                x = (i - (nx-1)/2) * px
                y = (j - (ny-1)/2) * py
                positions.append((x, y))
        
        self.positions = self._apply_transform(positions, offset, rotation)
    
    def _apply_transform(self, positions, offset, rotation):
        if rotation != 0:
            cos_r, sin_r = jnp.cos(rotation), jnp.sin(rotation)
            positions = [(float(x*cos_r - y*sin_r), float(x*sin_r + y*cos_r))
                        for x, y in positions]
        return [(x + offset[0], y + offset[1]) for x, y in positions]
    
    def __iter__(self):
        return iter(self.positions)
    
    def __len__(self):
        return len(self.positions)

class TriangleLayout:
    def __init__(self, np: int, px: float, py: float,
                 offset: Tuple[float, float] = (0, 0),
                 rotation: float = 0):
        """Triangle arrangement (np rows, decreasing)
        np: number of points on one edge
        px, py: pitch
        Total points: np*(np+1)/2
        """
        positions = []
        for i in range(np, 0, -1):
            for j in range(1, i+1):
                xp = ((i-1) - (np-1)/2) * px
                yp = ((j-1) - (np-1)/2) * py
                positions.append((xp, yp))
        
        self.positions = self._apply_transform(positions, offset, rotation)
    
    def _apply_transform(self, positions, offset, rotation):
        if rotation != 0:
            cos_r, sin_r = jnp.cos(rotation), jnp.sin(rotation)
            positions = [(float(x*cos_r - y*sin_r), float(x*sin_r + y*cos_r)) 
                        for x, y in positions]
        return [(x + offset[0], y + offset[1]) for x, y in positions]
    
    def __iter__(self):
        return iter(self.positions)
    
    def __len__(self):
        return len(self.positions)
    
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
                 ds: Tuple[float, float],
                 center: Tuple[float, float]=(0, 0),
                 rotation: float = 0) -> jnp.ndarray:
        """Generate Gaussian field.
        
        Returns:
            (nx, ny) complex array
        """
        x, y = spatial_grid(shape, ds)
        xc, yc = center
        x = x - xc
        y = y - yc
        r2 = x**2 + y**2
        A = jnp.sqrt(2.0 / (jnp.pi * self.w0**2))
        return A*jnp.exp(-r2 / self.w0**2).astype(self.dtype)


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
                 ds: Tuple[float, float],
                 center: Tuple[float, float]=(0, 0),
                 rotation: float = 0) -> jnp.ndarray:
        """Generate HG mode.
        
        Returns:
            (nx, ny) complex array
        """
        x, y = spatial_grid(shape, ds)
        xc, yc = center
        x = x - xc
        y = y - yc

        # Apply rotation
        if rotation != 0:
            cos_r = jnp.cos(rotation)
            sin_r = jnp.sin(rotation)
            x_rot = x * cos_r + y * sin_r
            y_rot = -x * sin_r + y * cos_r
            x, y = x_rot, y_rot
        
        # Normalized coordinates
        u = jnp.sqrt(2) * x / self.w0
        v = jnp.sqrt(2) * y / self.w0
        
        # HG mode
        gaussian = jnp.exp(-(x**2 + y**2) / self.w0**2)
        hg = self.Hm(u) * self.Hn(v) * gaussian
        
        # Normalization
        A = jnp.sqrt(2.0 / (jnp.pi * self.w0**2)) * (-1)**(2*self.m+self.n)
        norm = A / jnp.sqrt(2**(self.m + self.n) *
                              factorial(self.m) * 
                              factorial(self.n))
        
        return jnp.array(hg * norm, dtype=self.dtype)

    
def hermite_gaussian_groups(n_groups: int, w0: float):
    modes = []
    for m in range(n_groups):
        for n in range(n_groups - m):
            modes.append(HermiteGaussian(w0, m, n))
    return modes

def generate_mode_stack(modes: Sequence[Callable],
                        shape: Tuple[int, ...], ds: Tuple[float, ...],
                        centers: Optional[Sequence[Tuple[float, ...]]] = None,
                        rotation: float = 0):
    if centers is not None:
        assert len(centers) == len(modes)
    else:
        centers = [(0, 0)] * len(modes)
    return jnp.stack(list(map(lambda x: x[0](shape, ds, center=x[1], rotation=rotation),
                              zip(modes, centers))), axis=0)
