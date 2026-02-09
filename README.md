# JAXOptics

A minimal Python/JAX implementation of scalar optical field propagation with automatic differentiation. Built with [Equinox](https://github.com/patrick-kidger/equinox) for composable differentiable optical systems. Created to explore the JAX ecosystem as an alternative to the full [FluxOptics.jl](https://github.com/anscoil/FluxOptics.jl) framework.

## Purpose

This project reimplements a **subset** of FluxOptics.jl functionality using JAX and Equinox, to compare performance and developer experience between:
- **JAX/Equinox** (Python): JIT compilation via XLA, PyTree-based composability, GPU support
- **Julia/Zygote** (FluxOptics.jl): Native GPU via CUDA.jl, composable AD, multiple dispatch

JAXOptics focuses on **simple tensor-based operations** (FFT, phase masks, elementwise operations) where both frameworks are expected to perform similarly. For production work and comprehensive features (3D propagation, arbitrary geometries, extensive mode libraries), use [FluxOptics.jl](https://github.com/anscoil/FluxOptics.jl).

## Implemented Features

**Propagation Methods:**
- Angular Spectrum (AS) propagation
- Rayleigh-Sommerfeld (RS) propagation
- Trainable propagation distances

**Optical Elements:**
- Phase masks

**Mode Generation:**
- Laguerre-Gaussian modes
- Hermite-Gaussian modes
- Gaussian beams

**Utilities:**
- Power normalization
- Field visualization
- Composable optical sequences

## Installation

```bash
pip install jax jaxlib equinox optax matplotlib
```

For GPU support, install JAX with CUDA:
```bash
pip install jax[cuda12]  # Adjust CUDA version as needed
```

Clone and install:
```bash
git clone https://github.com/anscoil/jaxoptics
cd jaxoptics
pip install -e .
```

## Quick Example

```python
import jax.numpy as jnp
from jaxoptics import *

# Create field
ns = (512, 512)
ds = (1.0, 1.0)  # μm
wavelength = 0.532  # μm

modes = generate_mode_stack(laguerre_gaussian_groups(5, 25.0), ns, ds)
field = ScalarField(modes, ds, wavelength)

# Propagate with phase mask
propagator = ASProp(field, z=1000.0)
phase_mask = Phase(field, trainable=True)

system = OpticalSequence(propagator, phase_mask, propagator)
output = system(field)
```

See `examples/` for complete optimization workflows.

## Performance Notes

For simple tensor-based operations (FFT, phase masks, elementwise operations), JAX and FluxOptics.jl perform comparably—both leverage optimized GPU kernels and benefit from the inherently parallel nature of these operations.

Performance differences may emerge for:
- More complex optical systems (custom propagators, arbitrary geometries)
- Large-scale problems requiring sophisticated memory management

## Contributing

This project serves primarily as a comparison reference. If you're interested in developing JAXOptics further or using it for your research, please contact me.

## Citation

If you use this code, please cite:

```bibtex
@software{jaxoptics2026,
  author = {Barré, Nicolas},
  title = {JAXOptics: Differentiable Optical Propagation in JAX},
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.XXXXXXX},
  url = {https://github.com/anscoil/jaxoptics}
}
```

*(DOI will be assigned upon first Zenodo release)*

## License

MIT License. See FluxOptics.jl for the reference implementation and comprehensive documentation.

## Related Projects

- **[FluxOptics.jl](https://github.com/anscoil/FluxOptics.jl)** - Full-featured Julia framework (primary project)
- **[TorchOptics](https://github.com/MatthewFilipovich/torchoptics)** - PyTorch-based optical propagation
