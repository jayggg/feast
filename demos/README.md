# Demos

Small, complete scripts showing typical use of `feast`. They are also
rendered in the documentation (`docs/demos.rst`) and run by the test
suite (`tests/test_demos.py`).

| Demo | Shows | Needs |
| --- | --- | --- |
| `core/01_quickstart.py` | Eigenvalues of a small matrix inside a circle | numpy, scipy |
| `core/02_modes.py` | `hermitian`, `right` and `biorth` modes on one sparse pencil | numpy, scipy |
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

Research runs, convergence studies and paper reproductions do not
belong here.
