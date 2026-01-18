import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Callable
from ..fields import ScalarField

def make_RS_kernel(ns: Tuple[int, int],
                   ds: Tuple[float, float],
                   wavelengths: jnp.ndarray,
                   z: float,
                   n0: float = 1.0) -> jnp.ndarray:
    """Build Rayleigh Sommerfeld propagation kernel.
    
    Args:
        shape: (nx, ny)
        ds: (dx, dy) spatial sampling
        wavelengths: array of wavelengths (batch compatible)
        z: propagation distance
        n0: refractive index
    
    Returns:
        kernel: (batch..., nx, ny) complex array
    """
    x, y = [jnp.roll(jnp.arange(1 - nx, nx) * dx, nx) for nx, dx in zip(ns, ds)]
    x, y = jnp.meshgrid(x, y, indexing="ij")
    
    # Handle backward propagation
    z_sign = jnp.where(z >= 0, 1.0, -1.0)
    z_abs = abs(z)

    # Broadcast wavelengths to batch shape
    wl = wavelengths / n0 + 0j
    # Reshape for broadcasting: (batch..., 1, 1)
    wl_shape = wl.shape + (1, 1)
    wl = wl.reshape(wl_shape)

    dx, dy = ds
    nrm_f = dx * dy / (2 * jnp.pi)
    k = 2 * jnp.pi / wl
    r = jnp.sqrt(x**2 + y**2 + z_abs**2)
    kernel = nrm_f * (jnp.exp(1j*z_sign*k*r) / r) * (z_abs/r) * (1/r - 1j*z_sign*k)

    return jnp.fft.fft2(kernel, axes=(-2, -1))

class RSProp(eqx.Module):
    """Rayleigh Sommerfeld propagator.
    
    Supports both cached (static kernel) and dynamic (trainable distance) modes.
    """
    kernel: jnp.ndarray
    z: jnp.ndarray
    is_trainable: bool = eqx.field(static=True)
    n0: float = eqx.field(static=True)
    
    def __init__(self,
                 u: ScalarField,
                 z: float,
                 trainable: bool = False,
                 n0: float = 1.0):
        """
        Args:
            u: Template field (for ds, wavelengths, shape)
            z: Propagation distance
            trainable: Pre-compute kernel if not trainable vs dynamic if trainable
            n0: Refractive index
        """
        self.z = jnp.array([z])  # Array for trainability
        self.is_trainable = trainable
        self.n0 = n0
        
        # Pre-compute kernel if cached
        if not(trainable):
            # Deduce shape from u
            spatial_shape = u.shape[-u.ndim_spatial:]
            
            # Build kernel (numpy)
            self.kernel = make_RS_kernel(spatial_shape, u.ds, u.wavelengths, z, n0)
        else:
            self.kernel = jnp.empty((0,), dtype=u.electric.dtype)
    
    def __call__(self, u: ScalarField) -> ScalarField:
        """Propagate the field u through free space.
        
        Args:
            u: Input field
        
        Returns:
            Propagated field
        """
        spatial_shape = u.shape[-u.ndim_spatial:]
        if self.is_trainable:
            # Dynamic mode: z is trainable, compute kernel on-the-fly
            z_use = self.z[0]
            
            # Compute kernel dynamically
            kernel = make_RS_kernel(spatial_shape, u.ds, u.wavelengths, z_use, self.n0)
        else:
            # Cached mode: z is not trainable
            z_use = jax.lax.stop_gradient(self.z[0])
            # Convert kernel to jnp (on GPU if needed)
            kernel = jax.lax.stop_gradient(self.kernel)
        
        nx, ny = spatial_shape
        padded_shape = (2*nx - 1, 2*ny - 1)

        # Construct pad_width: no padding for batch dims, pad spatial dims
        pad_width = [(0, 0)] * (u.electric.ndim - u.ndim_spatial)  # batch dims
        pad_width.extend([(0, ps - s) for s, ps in zip(spatial_shape, padded_shape)])

        # Pad with zeros
        u_padded = jnp.pad(u.electric, pad_width, mode='constant', constant_values=0)

        # FFT on spatial axes
        spatial_axes = tuple(range(-u.ndim_spatial, 0))
        u_fft = jnp.fft.fft2(u_padded, axes=spatial_axes)

        # Multiply by kernel
        u_fft_prop = u_fft * kernel

        # IFFT
        u_prop_padded = jnp.fft.ifft2(u_fft_prop, axes=spatial_axes)

        # Extract top-left corner (original size)
        slices = [slice(None)] * (u.electric.ndim - u.ndim_spatial)  # keep batch dims
        slices.extend([slice(0, s) for s in spatial_shape])  # extract original spatial
        electric_prop = u_prop_padded[tuple(slices)]
        
        return ScalarField(electric_prop, u.ds, u.wavelengths)
