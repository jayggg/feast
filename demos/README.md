
# Demos

Small, complete scripts showing typical use of `feast`. They are also
rendered in the documentation (`docs/demos.rst`) and run by the test
suite (`tests/test_demos.py`).

| Demo | Shows | Needs |
| --- | --- | --- |
| `core/01_quickstart.py` | Eigenvalues of a small matrix inside a circle | numpy, scipy |
| `core/02_modes.py` | `hermitian`, `right` and `biorth` modes on one sparse pencil | numpy, scipy |
| `core/03_generalized_pencil.py` | `A x = λ B x` with a mass matrix; B-orthonormality, residuals | numpy, scipy |
| `core/04_contours.py` | Circle, ellipse and a custom rectangle; filter values | numpy, scipy |
| `core/05_diagnostics.py` | Logging levels, `result.history`, checking results, nonconvergence | numpy, scipy |
| `core/06_projector.py` | `SpectralProjectorApprox` applied directly; `P = V f(Λ) V⁻¹`; `apply` vs `apply_adj` on a non-normal matrix | numpy, scipy |
| `core/07_symmetry.py` | `reduce_sym=True` halves the factorizations | numpy, scipy |
| `core/08_zoom.py` | Splitting one contour into one per eigenvalue group | numpy, scipy |
| `core/09_polynomial.py` | Quadratic eigenproblem with `MatrixPolyResolvent` | numpy, scipy |
| `ngsolve/01_dirichlet_square.py` | Finite element pencil, degenerate eigenvalues | ngsolve |

Run one from the repository root, e.g. `python demos/core/01_quickstart.py`,
or all of them with `python -m pytest tests/test_demos.py`.

## Rules for adding a demo

- **One self-contained file.** No imports from other demos, no shared
  helper module. Repeating a few lines of setup is fine.
- **Small and fast.** Roughly under 100 lines and a few seconds at
  default settings.
- **Checks itself.** End with `assert`s against known answers (exact
  eigenvalues, a residual bound).
- **No output files.** Do not write `.npz`, `.vtk`, logs or figures
  into the repository.
- **Seeded.** Pass explicit seeds to spans and to `feast()`.
- **Header docstring.** Say what the demo shows, what it needs, and how
  to run it.
- **Named for reading order.** `NN_topic.py` inside `core/` or the
  backend's folder.
- **Listed in the docs.** Add a `literalinclude` to `docs/demos.rst` and
  a row to the table above. `tests/test_demos.py` fails if the docs
  entry is missing.

Backend demos show only what is specific to that backend. If everything
after building the resolvent and span is the same as in a core demo, say
so and refer to it instead of repeating it.

Research runs, convergence studies and project-specific reproductions do not
belong here.
