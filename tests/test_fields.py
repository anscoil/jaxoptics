import pytest
import jax.numpy as jnp
from jaxoptics import ScalarField

def test_scalar_field_2d_single():
    field = ScalarField(
        jnp.ones((512, 512), dtype=jnp.complex64),
        ds=(10e-6, 10e-6),
        wavelengths=1.064e-6
    )
    assert field.shape == (512, 512)
    assert field.batch_shape == ()
    assert field.ndim_spatial == 2

def test_scalar_field_2d_batch():
    field = ScalarField(
        jnp.ones((10, 512, 512)),
        ds=(10e-6, 10e-6),
        wavelengths=jnp.linspace(1.0, 1.1, 10)*1e-6
    )
    assert field.batch_shape == (10,)
    assert field.wavelengths.shape == (10,)

def test_wavelength_broadcast():
    field = ScalarField(
        jnp.ones((5, 3, 512, 512)),
        ds=(10e-6, 10e-6),
        wavelengths=jnp.ones((5, 1))*1e-6
    )
    assert field.wavelengths.shape == (5, 3)

