Backend contracts and design
==================================

The three main interfaces
-------------------------------

``Contour`` owns quadrature nodes, weights, a membership test, and the
scale used for relative convergence measurements. It is independent of
the discretization.

``Span`` represents columns of vectors. Its required operations are
``dim``, ``copy``, ``rotate``, ``scale``, ``remove``, ``append_random``, and
``qr``.
Mutating methods operate in place. ``rotate(V)`` replaces Y with YV,
including when V has fewer columns. Symmetry reduction also requires
an in-place ``conjugate`` method.

``qr()`` replaces Y by orthonormal columns Q and returns a small factor R
such that Y = Q R up to roundoff. R need not be triangular: a backend may
pivot the columns and absorb that permutation into R. The shared
``orthonormalize`` operation takes an SVD of R and retains singular values
greater than ``rank_tol`` times the largest. This avoids forming YᴴY for
rank detection. Zero and roundoff-level directions may be removed by QR.

The driver uses the coefficient Euclidean inner product for this step,
including for generalized pencils with non-Hermitian B. ``NGSpan.qr``
additionally accepts ``ipmat`` for applications needing a Hermitian
positive definite custom metric. It operates on NGSolve vectors, using
pivoted reorthogonalization and native normalization. Dependent columns
are rejected before normalization; the tests verify the native
``Orthogonalize`` custom-mass and ``AppendOrthogonalize`` Euclidean paths.

``Resolvent`` supplies shifted solves, reduced matrices, zero storage,
and accumulation. With K(z) = zB - A, the two solve contracts are::

    apply(z, Y)     = K(z)⁻¹ B Y
    apply_adj(z, Y) = K(z)⁻ᴴ Bᴴ Y

Both return new spans without changing the input. The second operation
filters generalized left eigenvectors satisfying
``Aᴴ y = conj(λ) Bᴴ y``. For general B it differs from the Euclidean
adjoint of ``apply``, which would put Bᴴ on the other side of the inverse.

Reduced matrices follow the column convention::

    rayleigh(Yl, Yr) = (Ylᴴ A Yr, Ylᴴ B Yr)
    gram(Yl, Yr)     = Ylᴴ B Yr

``new_like(Y)`` must allocate zero storage; ``axpy(Y, a, X)`` updates
Y by adding aX. A residual implementation is needed only when residual
reporting is requested. Report its norm and normalization in backend
documentation.

Iteration and storage
---------------------------

The driver composes a projector from a contour and resolvent. Each
iteration filters the spans, solves a reduced eigenproblem, and updates
the spans through their public operations. Biorthogonal kernel cleaning
analyzes a small Gram matrix before applying rotations and cuts.

Hermitian and right-only modes perform QR/SVD rank reduction before the
reduced solve. Refreshes and restarts work on copies of search spans,
preserving the last computed Ritz pairs for return if a later solve fails.
Returned pairs and convergence comparisons use only in-contour Ritz
values on every iteration. Convergence requires successive nonempty
in-contour sets of equal size, with at least ``expected_dim`` values when
requested. Periodic pruning controls the working span independently;
exterior directions can remain available for later iterations. If no
reduced solve succeeds or the last successful solve has no in-contour
values, the returned eigenvalue array and spans are empty and the result
is nonconverged.

This separation keeps decisions about rank and convergence independent
of vector storage. It also allows direct projector use without running
FEAST and lets a resolvent reuse cached factorizations across runs.
Construct a new resolvent after changing matrices or refining a mesh;
there is no public cache invalidation method.

Distributed implementations would need consistent reduced matrices on
every rank and deterministic random vectors for a given partition and
generator state. MPI scheduling and injection of custom projectors into
the driver are not currently implemented.

Finite element postprocessing
-----------------------------------

``NGSpan`` holds algebraic vectors and enforces homogeneous constrained
degrees of freedom. ``feast.backends.ngsolve.post`` provides grid
functions and drawing. Mesh refinement, interpolation between spaces,
physical normalization, and local error indicators belong to application
code. There is no built-in mesh-transfer or adaptive loop.

Eigenspace error estimates
--------------------------------

``estimate_eigenspace_error`` combines linear source error
representations at the contour nodes, forms their Gram matrix G and a
positive definite basis Gram matrix M, and solves ``G x = λ M x``.
The largest eigenvalue gives the squared estimator and its eigenvector
identifies the worst direction in the computed eigenspace.

The source estimator must supply representations, not scalar norms.
Its objects must support ``copy`` and ``rotate``. Reliability requires
the source-error assumptions described in :mod:`feast.estimator`;
this function does not establish them for an arbitrary backend.
``ExactSourceError`` is a reference implementation using a trusted
resolvent. A production DPG source estimator is not included.
