"""Dirichlet Laplacian on the unit square with NGSolve.

-Δu = λu with u = 0 on the boundary has eigenvalues (l² + m²)π².
The circle below encloses 2π² and the degenerate pair 5π², 5π², and
excludes 8π². The pencil is Hermitian, so ``mode="hermitian"`` applies.

Only the setup differs from the matrix demos: assemble forms on a
complex finite element space, wrap the matrices in NGPencilResolvent
and start from an NGSpan. The feast() call is the same.

Requires: ngsolve (pip install 'feast-eig[ngsolve]').
Run:      python demos/ngsolve/01_dirichlet_square.py
"""

import ngsolve as ng
import numpy as np
from netgen.geom2d import unit_square

from feast import Contour, feast
from feast.backends.ngsolve import NGPencilResolvent, NGSpan
from feast.backends.ngsolve.post import gridfun

mesh = ng.Mesh(unit_square.GenerateMesh(maxh=0.15))
fes = ng.H1(mesh, order=3, dirichlet="left|right|top|bottom", complex=True)
u, v = fes.TnT()
a = ng.BilinearForm(ng.grad(u) * ng.grad(v) * ng.dx).Assemble()
b = ng.BilinearForm(u * v * ng.dx).Assemble()

R = NGPencilResolvent(fes, a.mat, b.mat)
C = Contour.circle(center=40.0, radius=35.0, npts=8)
Y = NGSpan.random(fes, 5, seed=1)

result = feast(R, C, Y, mode="hermitian", seed=1)

lam = np.sort(result.eigenvalues.real)
exact = np.array([2, 5, 5]) * np.pi**2
print(result)
print("computed:", lam)
print("exact:   ", exact)

# A grid function for drawing, e.g. ng.Draw(mode) in a notebook.
mode = gridfun(result.right, i=0)

assert result.converged
assert np.allclose(lam, exact, rtol=2e-3)  # discretization error
