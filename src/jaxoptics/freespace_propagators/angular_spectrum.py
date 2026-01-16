import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Callable
from ..fields import ScalarField

def make_AS_kernel(ns: Tuple[int, int],
                   ds: Tuple[float, float],
                   wavelengths: jnp.ndarray,
                   z: float,
                   n0: float = 1.0,
                   paraxial: bool = False,
                   filter_fn: Optional[Callable] = None) -> jnp.ndarray:
    """Build Angular Spectrum propagation kernel.
    
    Args:
        shape: (nx, ny)
        ds: (dx, dy) spatial sampling
        wavelengths: array of wavelengths (batch compatible)
        z: propagation distance
        n0: refractive index
        paraxial: use paraxial approximation
        filter_fn: optional filter (fx, fy) -> transmission
    
    Returns:
        kernel: (batch..., nx, ny) complex array
    """
    nx, ny = ns
    dx, dy = ds
    fx = jnp.fft.fftfreq(nx, dx)
    fy = jnp.fft.fftfreq(ny, dy)

    fx, fy = jnp.meshgrid(fx, fy, indexing="ij")
    
    # Handle backward propagation
    z_sign = jnp.where(z >= 0, 1.0, -1.0)
    z_abs = abs(z)
    
    # Broadcast wavelengths to batch shape
    wl = wavelengths / n0 + 0j
    # Reshape for broadcasting: (batch..., 1, 1)
    wl_shape = wl.shape + (1, 1)
    wl = wl.reshape(wl_shape)
    
    if paraxial:
        # Paraxial: H = exp(-i π λ z (fx² + fy²))
        kernel = jnp.exp(-1j * jnp.pi * wl * z_sign * z_abs * (fx**2 + fy**2))
    else:
        # Exact: H = exp(i 2π z sqrt(1/λ² - fx² - fy²))
        f_squared = 1.0 / wl**2
        sqrt_arg = f_squared - fx**2 - fy**2
        kernel = jnp.exp(1j * 2 * jnp.pi * z_sign * z_abs * jnp.sqrt(sqrt_arg))
    
    # Apply filter if provided
    if filter_fn is not None:
        filter_mask = filter_fn(fx, fy)
        kernel = kernel * filter_mask
    return kernel


class ASProp(eqx.Module):
    """Angular Spectrum propagator.
    
    Supports both cached (static kernel) and dynamic (trainable distance) modes.
    """
    kernel: jnp.ndarray
    z: jnp.ndarray
    filter_fn: Optional[Callable] = eqx.field(static=True)
    use_cache: bool = eqx.field(static=True)
    is_paraxial: bool = eqx.field(static=True)
    n0: float = eqx.field(static=True)
    
    def __init__(self,
                 u: ScalarField,
                 z: float,
                 use_cache: bool = True,
                 paraxial: bool = False,
                 filter_fn: Optional[Callable] = None,
                 n0: float = 1.0):
        """
        Args:
            u: Template field (for ds, wavelengths, shape)
            z: Propagation distance
            use_cache: Pre-compute kernel (static) vs dynamic
            paraxial: Use paraxial approximation
            filter: Optional frequency filter
            n0: Refractive index
        """
        self.z = jnp.array([z])  # Array for trainability
        self.use_cache = use_cache
        self.is_paraxial = paraxial
        self.n0 = n0
        self.filter_fn = filter_fn
        
        # Pre-compute kernel if cached
        if use_cache:
            # Deduce shape from u
            spatial_shape = u.shape[-u.ndim_spatial:]
            
            # Build kernel (numpy)
            self.kernel = make_AS_kernel(spatial_shape, u.ds, u.wavelengths,
                                         z, n0, paraxial, filter_fn)
        else:
            self.kernel = jnp.empty((0,), dtype=u.electric.dtype)
    
    def __call__(self, u: ScalarField) -> ScalarField:
        """Propagate the field u through free space.
        
        Args:
            field: Input field
        
        Returns:
            Propagated field
        """
        if self.use_cache:
            # Cached mode: z is not trainable
            z_use = jax.lax.stop_gradient(self.z[0])
            # Convert kernel to jnp (on GPU if needed)
            kernel = jax.lax.stop_gradient(self.kernel)
        else:
            # Dynamic mode: z is trainable, compute kernel on-the-fly
            z_use = self.z[0]
            spatial_shape = u.shape[-u.ndim_spatial:]
            
            # Compute kernel dynamically
            kernel = make_AS_kernel(spatial_shape, u.ds, u.wavelengths,
                                    z_use, self.n0, self.is_paraxial, self.filter_fn)
        
        # FFT propagation
        spatial_axes = tuple(range(-u.ndim_spatial, 0))
        u_fft = jnp.fft.fft2(u.electric, axes=spatial_axes)
        u_fft_prop = u_fft * kernel
        electric_prop = jnp.fft.ifft2(u_fft_prop, axes=spatial_axes)
        
        return ScalarField(electric_prop, u.ds, u.wavelengths)
