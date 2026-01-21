import jax.numpy as jnp
from typing import Optional, Tuple, Union, Sequence, Callable
from scipy.special import hermite, genlaguerre, factorial, jv, jn_zeros

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

    
class LaguerreGaussian:
    """Laguerre-Gaussian mode generator."""
    
    def __init__(self, w0: float, p: int = 0, l: int = 0, dtype=jnp.complex64):
        """
        Args:
            w0: beam waist
            p: radial mode order
            l: azimuthal mode order (orbital angular momentum)
        """
        self.w0 = w0
        self.p = p
        self.l = l
        self.dtype = dtype
        
        # Precompute generalized Laguerre polynomial L_p^|l|
        self.Lpl = genlaguerre(p, abs(l))
    
    def __call__(self, shape: Tuple[int, int],
                 ds: Tuple[float, float],
                 center: Tuple[float, float]=(0, 0),
                 rotation: float = 0) -> jnp.ndarray:
        """Generate LG mode.
        
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
        
        # Polar coordinates
        r = jnp.sqrt(x**2 + y**2)
        theta = jnp.arctan2(y, x)
        
        # Normalized radial coordinate
        rho = jnp.sqrt(2) * r / self.w0
        
        # LG mode components
        radial = (rho**abs(self.l)) * self.Lpl(rho**2)
        gaussian = jnp.exp(-rho**2 / 2)
        azimuthal = jnp.exp(1j * self.l * theta)
        
        lg = radial * gaussian * azimuthal
        
        # Normalization factor
        norm = jnp.sqrt(2 * factorial(self.p) / 
                       (jnp.pi * factorial(self.p + abs(self.l)))) / self.w0
        
        return jnp.array(lg * norm, dtype=self.dtype)

class BesselCircular:
    """Circular Bessel mode generator (Fourier-Bessel modes)."""
    
    def __init__(self, r0: float, l: int = 0, m: int = 1, 
                 orientation: str = 'cos', dtype=jnp.complex64):
        """
        Args:
            r0: characteristic radius
            l: azimuthal order
            m: radial order (starts at 1)
            orientation: 'cos' or 'sin' for l > 0
        """
        self.r0 = r0
        self.l = l
        self.m = m
        self.orientation = orientation
        self.dtype = dtype
        
        # Get m-th zero of J_l Bessel function
        self.u_lm = jn_zeros(l, m)[m-1]
    
    def __call__(self, shape: Tuple[int, int],
                 ds: Tuple[float, float],
                 center: Tuple[float, float]=(0, 0),
                 rotation: float = 0) -> jnp.ndarray:
        """Generate Bessel circular mode.
        
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
        
        # Polar coordinates
        r = jnp.sqrt(x**2 + y**2)
        theta = jnp.arctan2(y, x)
        
        # Normalized radial coordinate
        rho = self.u_lm * r / self.r0
        
        # Radial part: Bessel function
        radial = jv(self.l, rho)
        radial = jnp.where(r <= self.r0, jv(self.l, rho), 0.0)
        
        # Azimuthal part
        if self.l == 0:
            azimuthal = 1.0
        else:
            if self.orientation == 'cos':
                azimuthal = jnp.cos(self.l * theta)
            else:  # 'sin'
                azimuthal = jnp.sin(self.l * theta)
        
        mode = radial * azimuthal
        
        # Normalization
        norm = jnp.sqrt(2) / (self.r0 * abs(jv(self.l + 1, self.u_lm)))
        if self.l == 0:
            norm = norm / jnp.sqrt(2)
        
        return jnp.array(mode * norm, dtype=self.dtype)

    
def hermite_gaussian_groups(n_groups: int, w0: float):
    """Generate Hermite-Gaussian modes grouped by degenerate order N = m + n.
    
    Creates modes ordered by increasing total order N, then by m index.
    For each group N, generates modes with (m,n) where m + n = N.
    
    Args:
        n_groups: Number of mode groups (N = 0, 1, ..., n_groups-1)
        w0: Beam waist parameter
        
    Returns:
        List of HermiteGaussian instances. Number of modes = n_groups*(n_groups+1)/2
    """
    modes = []
    for m in range(n_groups):
        for n in range(n_groups - m):
            modes.append(HermiteGaussian(w0, m, n))
    return modes

def laguerre_gaussian_groups(n_groups: int, w0: float):
    """Generate LG modes grouped by degenerate order N = 2p + |l|.
    
    Args:
        n_groups: number of mode groups (N = 0, 1, ..., n_groups-1)
        w0: beam waist
        
    Returns:
        List of LaguerreGaussian modes ordered by increasing N, then by p
    """
    modes = []
    for N in range(n_groups):
        for p in range(N + 1):
            l = N - 2*p
            if l >= 0:
                modes.append(LaguerreGaussian(w0, p, l))
                if l > 0:  # Add negative l mode
                    modes.append(LaguerreGaussian(w0, p, -l))
    return modes

def bessel_circular_groups(n_groups: int, r0: float):
    """Generate Bessel circular modes grouped by cutoff frequency order.
    
    Modes are ordered by increasing cutoff (u_lm values).
    For l > 0, both 'cos' and 'sin' orientations are included.
    
    Args:
        n_groups: Number of mode groups to generate
        w0: Characteristic radius
        
    Returns:
        List of BesselCircular instances (n_groups*(n_groups+1)/2 unique modes)
    """
    modes = []
    n_modes = n_groups*(n_groups+1)/2
    for N in range(n_groups):
        for p in range(N + 1):
            l = N - 2*p
            m = p + 1
            if l >= 0:
                modes.append(BesselCircular(r0, l, m, 'cos'))
                if l > 0:
                    modes.append(BesselCircular(r0, l, m, 'sin'))
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
