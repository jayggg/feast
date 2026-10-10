NGSolve usage and API
===========================

A finite element pencil
-----------------------------

Assemble the matrices on a complex finite element space. The resolvent
restricts shifted inverses to the space's free degrees of freedom.

.. code-block:: python

   import ngsolve as ng
   from netgen.geom2d import unit_square
   from feast import Contour, feast
   from feast.backends.ngsolve import NGPencilResolvent, NGSpan
   from feast.backends.ngsolve.post import gridfun

   mesh = ng.Mesh(unit_square.GenerateMesh(maxh=0.15))
   fes = ng.H1(
       mesh, order=3, dirichlet="left|right|top|bottom", complex=True,
   )
   u, v = fes.TnT()
   a = ng.BilinearForm(ng.grad(u) * ng.grad(v) * ng.dx).Assemble()
   b = ng.BilinearForm(u * v * ng.dx).Assemble()
   R = NGPencilResolvent(fes, a.mat, b.mat)
   C = Contour.circle(center=20.0, radius=10.0, npts=8)
   Y = NGSpan.random(fes, 3, seed=10)
   result = feast(R, C, Y, mode="biorth", seed=1)
   mode = gridfun(result.right, i=0)
   print(result.eigenvalues)

This contour targets the ground state near ``2π²``. To visualize the
result in an appropriate GUI or notebook environment, use ``ng.Draw(mode)``
or ``feast.backends.ngsolve.post.draw(result.right)``.

Pass ``inverse="umfpack"`` or another available inverse type to the
resolvent to select a solver. Left filtering requires the inverse's
conjugate-transpose operation to work correctly. Symmetry reduction
requires a real pencil as well as a symmetric contour; Hermitian
matrices with complex entries do not automatically satisfy this condition.

Polynomial eigenproblems
------------------------------

For ``A(λ) = A0 + λ A1 + ... + λᵈ Ad``, pass the assembled coefficient
matrices to ``NGPolyResolvent``. Its initial spans live on
``R.product_fes``, whose vectors have ``d * fes.ndof`` entries.

Continuing from the finite element assembly above:

.. code-block:: python

   from feast.backends.ngsolve import NGPolyResolvent

   negm = ng.BilinearForm(-u * v * ng.dx).Assemble()
   zero = negm.mat.CreateMatrix()
   zero.AsVector().data = 0.0 * negm.mat.AsVector()
   Rpoly = NGPolyResolvent([a.mat, zero, negm.mat], fes)
   Cpoly = Contour.circle(center=4.4, radius=1.5, npts=8)
   result = feast(
       Rpoly, Cpoly, Rpoly.random_span(3, seed=10),
       mode="biorth", seed=1,
   )
   physical_right = Rpoly.first(result.right)
   physical_left = Rpoly.last(result.left)

This solves ``K x = λ² M x`` for the positive ground-state root near
``sqrt(2) * π``. The last block of a left companion eigenvector is a
left null vector of the evaluated polynomial. The companion pencil is
generally non-Hermitian even when its coefficients are Hermitian.

Backend reference
-----------------------

.. automodule:: feast.backends.ngsolve.span
   :members:
   :undoc-members:

.. automodule:: feast.backends.ngsolve.resolvent
   :members:
   :undoc-members:

.. automodule:: feast.backends.ngsolve.poly
   :members:
   :undoc-members:

.. automodule:: feast.backends.ngsolve.post
   :members:
