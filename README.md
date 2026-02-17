# JAXOptics
[![DOI](https://zenodo.org/badge/1134334955.svg)](https://doi.org/10.5281/zenodo.18547486)

A minimal Python implementation of scalar optical field propagation with automatic differentiation using [JAX](https://github.com/google/jax) and [Equinox](https://github.com/patrick-kidger/equinox). Created to explore the JAX ecosystem as an alternative to the full [FluxOptics.jl](https://github.com/anscoil/FluxOptics.jl) framework.

## 🎯 Purpose

This project reimplements a **subset** of [FluxOptics.jl](https://github.com/anscoil/FluxOptics.jl) functionality in Python using JAX and Equinox, to compare performance and developer experience between the two ecosystems. It also serves as a **well-optimized Python baseline** for benchmarking against other Python optical simulation libraries.

## ⚡ Performance

JAXOptics significantly outperforms [TorchOptics](https://github.com/MatthewFilipovich/torchoptics)
on Angular Spectrum propagation benchmarks (512×512, NVIDIA RTX 4070): **two orders of magnitude
faster** on isolated propagation, and **13× faster** on a simple 3-plane monomode
[beam splitter design](https://anscoil.github.io/FluxOptics.jl/stable/api/#Typical-Workflow:-Beam-Splitter).

JAXOptics is also within **20% of FluxOptics.jl** on propagation-only benchmarks, and may match
or slightly exceed it on full optimization loops thanks to XLA whole-graph compilation.

→ [Full benchmark details](https://github.com/anscoil/FluxOptics.jl/tree/main/benchmarks)

## 🔧 Implemented Features

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

## 📦 Installation

```bash
pip install jax jaxlib equinox optax matplotlib
```

For GPU support:
```bash
pip install jax[cuda12]  # Adjust CUDA version as needed
```

Clone and install:
```bash
git clone https://github.com/anscoil/jaxoptics
cd jaxoptics
pip install -e .
```

## 🚀 Quick Example

```python
import jax.numpy as jnp
from jaxoptics import *

ns = (512, 512)
ds = (1.0, 1.0)  # μm
wavelength = 0.532  # μm

modes = generate_mode_stack(laguerre_gaussian_groups(5, 25.0), ns, ds)
field = ScalarField(modes, ds, wavelength)

propagator = ASProp(field, z=1000.0)
phase_mask = Phase(field, trainable=True)
system = OpticalSequence(propagator, phase_mask, propagator)
output = system(field)
```

See [`examples/`](examples) for complete optimization workflows.

## ⚠️ Current Limitations

JAXOptics currently focuses on **scalar fields with phase-only elements**, covering the most common use cases in beam shaping and diffractive optics. More advanced components are available in [FluxOptics.jl](https://github.com/anscoil/FluxOptics.jl).

## 🤝 Contributing

JAXOptics was developed primarily as a benchmark reference, but the JAX/Equinox stack turned
out to be a genuinely productive environment for optical simulations. Extending it toward a
more complete feature set while keeping the performance advantages over existing Python libraries
would be a natural next step. If you are interested in collaborating or have a use case in mind,
feel free to reach out.

## 📝 Citation

```bibtex
@software{jaxoptics2026,
  author = {Barré, Nicolas},
  title = {JAXOptics: Differentiable Optical Propagation in JAX},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.18547486},
  url = {https://github.com/anscoil/jaxoptics}
}
```

## 📄 License

[MIT License](LICENSE).

## 🔗 Related Projects

- **[FluxOptics.jl](https://github.com/anscoil/FluxOptics.jl)** — Full-featured Julia framework (primary project)
- **[TorchOptics](https://github.com/MatthewFilipovich/torchoptics)** — PyTorch-based optical propagation
