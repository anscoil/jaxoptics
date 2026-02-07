import equinox as eqx
import jax
import jax.numpy as jnp
from scipy.special import jacobi
from ..fields import ScalarField
from ..modes import spatial_grid
from typing import Optional, Tuple, Union, Callable

def apply_phase_mask(u: jnp.ndarray, phase_mask: jnp.ndarray) -> jnp.ndarray:
    return ScalarField(u.electric * jnp.exp(1j * phase_mask), u.ds, u.wavelengths)

class Phase(eqx.Module):
    """Phase element applying a spatial phase mask.
    
    Attributes:
        phase_mask: Phase values in radians (nx, ny)
        is_trainable: If True, gradients flow through phase_mask
    
    Args:
        u: Input field to initialize shape
        init_fn: Optional initialization function(x, y) -> phase. 
                 Defaults to zeros.
        trainable: Enable gradient computation for phase_mask
    """
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

    def phase(self):
        return self.phase_mask

    def __call__(self, u: jnp.ndarray) -> jnp.ndarray:
        if self.is_trainable:
            return apply_phase_mask(u, self.phase_mask)
        else:
            return apply_phase_mask(u, jax.lax.stop_gradient(self.phase_mask))


def compute_phase(coeffs, basis):
    return jnp.einsum('i,i...->...', coeffs, basis)
    
class PhaseProjection(eqx.Module):
    """Phase element with basis projection (Zernike, wavelets, etc.).
    
    Attributes:
        coefficients: Expansion coefficients on basis functions
        basis_matrix: Precomputed basis modes (n_basis, nx, ny)
        is_trainable: If True, gradients flow through coefficients
    
    Args:
        u: Input field to initialize shape
        basis_matrix: Precomputed basis (n_basis, nx, ny)
        trainable: Enable gradient computation for coefficients
    """
    coefficients: jnp.ndarray
    basis_matrix: jnp.ndarray # (n_basis, nx, ny)
    is_trainable: bool = eqx.field(static=True)
    
    def __init__(self, u: ScalarField, basis_matrix: jnp.ndarray, trainable=True):
        """
        Args:
            u: field to initialize shape
            basis_matrix: precomputed basis (n_basis, nx, ny)
        """
        self.coefficients = jnp.zeros(len(basis_matrix), dtype=u.electric.real.dtype)
        self.basis_matrix = basis_matrix
        self.is_trainable = trainable
    
    def phase(self):
        # Reconstruct phase from coefficients
        basis = jax.lax.stop_gradient(self.basis_matrix)
        return compute_phase(self.coefficients, basis)

    def __call__(self, u: ScalarField) -> ScalarField:
        phase_mask = self.phase()
        # phase_mask = jnp.sum(self.coefficients[:, None, None] * basis, axis=0)
        
        if self.is_trainable:
            return apply_phase_mask(u, phase_mask)
        else:
            return apply_phase_mask(u, jax.lax.stop_gradient(phase_mask))

def zernike_radial(n: int, m: int, rho: jnp.ndarray) -> jnp.ndarray:
    """Radial part using Jacobi polynomials."""
    if (n - m) % 2 != 0:
        return jnp.zeros_like(rho)
    
    s = (n - m) // 2
    P = jacobi(s, 0, m)
    result = (-1)**s * rho**m * P(2*rho**2 - 1)
    return jnp.asarray(result)  # Ensure JAX array

def compute_zernike_basis(shape: tuple[int, int], 
                          ds: tuple[float, float],
                          max_order: int,
                          radius: float,
                          exclude: list[int] = None) -> jnp.ndarray:
    """Compute Zernike polynomial basis up to specified order.
    
    Args:
        shape: Spatial shape (nx, ny)
        ds: Pixel spacing (dx, dy)
        max_order: Maximum radial order n
        radius: Aperture radius for Zernike normalization
        exclude: List of Noll indices to exclude (e.g., [1] for piston, 
                 [1, 2, 3] for piston + tilts)
    
    Returns:
        basis_matrix: (n_modes, nx, ny) array of Zernike modes
        
    Note:
        Uses Noll indexing convention (1-indexed):
        1: piston, 2: tip, 3: tilt, 4: defocus, etc.
    """
    x, y = spatial_grid(shape, ds)
    
    # Normalize to unit circle
    r = jnp.sqrt(x**2 + y**2)
    rho = r / radius
    theta = jnp.arctan2(y, x)
    
    # Generate modes up to max_order
    modes = []
    j = 1  # Noll index
    
    for n in range(max_order + 1):
        for m in range(-n, n + 1, 2):
            if exclude is not None and j in exclude:
                j += 1
                continue
                
            # Radial polynomial R_n^|m|
            R = zernike_radial(n, abs(m), rho)
            
            # Azimuthal part
            if m >= 0:
                Z = R * jnp.cos(m * theta)
            else:
                Z = R * jnp.sin(abs(m) * theta)
            
            # Normalization
            norm = jnp.sqrt(2 * (n + 1) / (1 + (m == 0)))
            
            # Mask outside aperture
            Z = jnp.where(rho <= 1.0, Z * norm, 0.0)
            
            modes.append(Z)
            j += 1
    
    return jnp.array(modes)
