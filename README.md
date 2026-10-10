# feast

FEAST contour-integral eigensolver for selected eigenvalues and eigenspaces
of matrix pencils. Import name: `feast`. Distribution name: `feast-eig`.
Documentation: https://jayggg.github.io/feast/

The package separates three concepts:

- `Contour`: quadrature nodes, weights, membership tests, and rational filters.
- `Resolvent`: shifted solves and reduced matrices supplied by a backend.
- `feast()`: Hermitian, right-only, or biorthogonal iteration returning a
  `FeastResult` with vectors, eigenvalues, and diagnostics.

NumPy/SciPy matrix backends are included. NGSolve backends support finite
element pencils, callback resolvents, and polynomial eigenproblems.

## Installation

From this checkout, using Python 3.10 or later:

```sh
python -m pip install -e .
```

Use your existing NGSolve Python environment, or install the optional extra
with `python -m pip install -e '.[ngsolve]'`. The PyPI distribution named
`feast` is unrelated to this project.

## Quick start

```python
import numpy as np
from feast import Contour, feast
from feast.backends.numpy_backend import MatrixResolvent, NumpySpan

A = np.diag([-1.0, 1.0, -1j, 1j, 3.0, 4.0])
R = MatrixResolvent(A)
C = Contour.circle(center=3.5, radius=0.8, npts=8)
Y = NumpySpan.random(A.shape[0], 4, seed=10)

result = feast(R, C, Y, mode="biorth", seed=10)
print(np.sort_complex(result.eigenvalues))
```

See [getting started](docs/getting_started.rst), the runnable
[demos](demos/),
[NGSolve usage](docs/ngsolve.rst), and
[backend contracts](docs/architecture.rst).
Read the [current limitations](docs/limitations.rst) before relying on
convergence flags, contour membership, or returned span dimensions.

## Build the documentation

```sh
python -m pip install -e '.[docs]'
python -m sphinx -b html -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html`, or run
`sphinx-autobuild docs docs/_build/html --watch src --watch demos` for a
live-reloading preview at http://127.0.0.1:8000. Sphinx generates the API reference from
NumPy-style docstrings using autodoc and Napoleon. Rerun the build after
edits; no separate API-generation command is needed. See
[documentation maintenance](docs/building.rst) for details.

## Development

```sh
python -m pip install -e '.[dev]'
python -m pytest
python -m ruff check src tests demos
```

NGSolve tests run when NGSolve is available and otherwise skip. Preserve
any `PYTHONPATH` entries required by a local NGSolve installation when
adding the source tree, for example `PYTHONPATH="src:$PYTHONPATH"`.
