The FEAST algorithm as implemented
=========================================

This page states, step by step, what :func:`~feast.driver.feast`, an
implementation of the FEAST contour-integral eigensolver [Pol2009]_,
does for each value of ``mode``, including every branch the code can take.
It describes the implementation, not an idealized method. Steps carry
labels such as **[H4]**; the same labels appear as comments in
``src/feast/driver.py`` at the code that carries them out, and the test
``tests/test_algorithm_doc.py`` fails if a label is added to or removed
from one place but not the other.

.. contents:: On this page
   :local:
   :depth: 1

Notation
--------------

The pencil is :math:`A x = \lambda B x` with :math:`A, B \in
\mathbb{C}^{n\times n}`, supplied by a backend
:class:`~feast.projector.Resolvent`. Write

.. math::

   K(z) = zB - A .

A :class:`~feast.contour.Contour` :math:`\Gamma` supplies nodes
:math:`z_k` and weights :math:`w_k`, :math:`k = 1, \dots, N`, approximating
:math:`\frac{1}{2\pi i}\oint_\Gamma g(z)\,dz \approx \sum_k w_k g(z_k)`,
a membership test ``within`` for the region enclosed by :math:`\Gamma`,
and a length ``scale`` :math:`s_\Gamma` used to make eigenvalue distances
relative. The **right and left filters** are

.. math::

   P = \sum_{k=1}^N w_k\, K(z_k)^{-1} B, \qquad
   \tilde P = \sum_{k=1}^N \overline{w_k}\, K(z_k)^{-H} B^H .

For a diagonalizable pencil with eigenvalues :math:`\lambda_j` and right
eigenvector matrix :math:`V`, :math:`P = V f(\Lambda) V^{-1}` with the
rational filter :math:`f(\lambda) = \sum_k w_k/(z_k-\lambda)`
(:meth:`~feast.contour.Contour.filter`): about 1 for eigenvalues well
inside :math:`\Gamma`, about 1/2 on it, small outside. :math:`\tilde P`
does the same for generalized left eigenvectors,
:math:`A^H y = \bar\lambda B^H y`. For general :math:`B` it is *not* the
Euclidean adjoint of :math:`P`.

A **span** :math:`Y` is a set of :math:`m = \dim Y` vectors, written as
the columns of an :math:`n\times m` matrix; the driver touches it only
through the :class:`~feast.span.Span` protocol. Orthonormalization uses
the Euclidean inner product of coefficient vectors, never :math:`B`.
The relative **Hausdorff distance** between finite sets
:math:`\Lambda, M \subset \mathbb{C}` is

.. math::

   d(\Lambda, M) = \frac{1}{s_\Gamma}\max\Bigl(
     \max_{\mu\in M}\min_{\lambda\in\Lambda}|\lambda-\mu|,\;
     \max_{\lambda\in\Lambda}\min_{\mu\in M}|\lambda-\mu| \Bigr),

and :math:`d = \infty` if either set is empty. The convergence test
([U7]) uses it between successive iterates. It is also the measure in
which [GGO2020]_ (Theorem 5.7) bounds the distance between the cluster
computed by filtered subspace iteration with a discretized resolvent
and the exact eigenvalue cluster of a selfadjoint operator.

Parameters of :func:`~feast.driver.feast` appear in code font
(``rank_tol``); fields of the returned
:class:`~feast.driver.FeastResult` are written ``result.right``, and
fields of its :class:`~feast.driver.FeastHistory` ``history.ritz_values``
and so on.

Overview
--------------

All three modes share one loop. Each iteration filters the working span,
makes a **rank decision** that may shrink it, solves a small
Rayleigh–Ritz eigenproblem, and then updates and tests the result
(Procedure U). The modes differ only in the rank decision and the
reduced eigenproblem:

==============  ===========================  ==========================  ==============================
mode            rank decision                reduced eigenproblem        rank record in ``history``
==============  ===========================  ==========================  ==============================
``hermitian``   QR + SVD of the filtered     Hermitian definite,         ``basis_singular_values``,
                basis, ``rank_tol`` [H4]     ``eigh`` [H7]               ``retained_ranks``
``right``       same as hermitian [R4]       general, ``eig`` [R7]       same
``biorth``      SVD of the left–right        two-sided ``eig``, with     ``gram_singular_values``,
                B-Gram matrix,               reduced B ≈ I [B7]          ``retained_ranks``
                ``eta_tol`` [B4]
==============  ===========================  ==========================  ==============================

Two words recur below. To **abandon the attempt** means to leave the
iteration loop of the current attempt; the next attempt (restart) begins
if ``nrestarts`` allows, otherwise the run ends without converging. A
**refresh** appends random vectors to the working span within the same
attempt, at most ``nrefresh`` times per attempt. Neither discards the
last saved result (step [U2]), which is what is returned if nothing
later succeeds.

Setup and finish (all modes)
----------------------------------

**[S1] Validate.** Reject an unknown ``mode``, ``reduce_sym=True``
outside Hermitian mode, ``rank_tol`` outside :math:`(0,1)`, and negative
``count_probes``. Treat ``nrestarts`` :math:`\le 0` as 1.

**[S2] Random stream and filter.** Create the random generator from
``seed``. It supplies all refresh, restart and default left vectors.
Build the filter: with ``reduce_sym=True``,
:class:`~feast.projector.SymmetryReducedProjector` (one factorization
per conjugate node pair, using
:math:`K(\bar z)^{-1} B y = \overline{K(z)^{-1} B \bar y}` for a real
pencil); otherwise :class:`~feast.projector.SpectralProjectorApprox`.

**[S3] Left span** (biorth only). If ``yl`` is given it must have
:math:`\dim \tilde Y = \dim Y`. If not, :math:`\tilde Y` is
:math:`\dim Y` random vectors. In the other modes ``yl`` is ignored.

**[S4] Initialize.** Empty history; saved result
:math:`(\Lambda_s, X_s, \tilde X_s)` empty; not converged; iteration
counter 0.

*The mode's algorithm (H, R or B below) runs here.*

**[E1] Empty result.** If no reduced solve ever succeeded, the saved
spans are empty spans of the right storage type. A warning is logged
whenever the run did not converge.

**[E2] Completeness check.** If ``count_probes`` :math:`> 0`, run
:func:`~feast.count.estimate_eigenvalue_count` with the same filter and
an independent random stream, giving the *unexplained filter weight*
:math:`\sum f(\lambda)` over eigenvalues not returned (see
:mod:`feast.count`). It is stored in ``result.count_estimate``, and a
warning is logged when it suggests missing eigenvalues.

**[E3] Return** :class:`~feast.driver.FeastResult` with
``eigenvalues`` :math:`=\Lambda_s`, ``right`` :math:`= X_s`, ``left``
:math:`=\tilde X_s` (biorth) or ``None``, ``converged``,
``iterations`` (total over all attempts, failed iterations included),
``history`` and ``count_estimate``.

Algorithm H: ``mode="hermitian"``
----------------------------------------

For pencils with :math:`A` Hermitian and :math:`B` Hermitian positive
definite on the filtered subspace.

**[H1] Attempt loop.** For attempt :math:`r = 0, 1, \dots,`
``nrestarts`` :math:`-1`: if :math:`r>0`, append
:math:`\max(\dim Y, 1)` random vectors to :math:`Y`. Set the refresh
budget to ``nrefresh`` and the previous in-contour set
:math:`\Lambda_{\rm prev} = \emptyset`.

**[H2] Iteration loop.** For ``it`` :math:`= 1,\dots,` ``niterations``,
count one iteration and do [H3]–[H9].

**[H3] Filter.** :math:`Q = P\,Y` (:math:`N` shifted solves per column,
or about :math:`N/2` with ``reduce_sym``). :math:`\dim Q = \dim Y = m`.

**[H4] Rank decision: QR + SVD of the filtered basis**
(:func:`~feast.subspace.orthonormalize`).

* Factor :math:`Q = Q_0 R` with :math:`Q_0` orthonormal (``Span.qr``).
  A backend may already drop zero or roundoff-level columns here
  (``NGSpan`` does), so :math:`R` is :math:`m_0\times m` with
  :math:`m_0 \le m`.
* Compute the SVD :math:`R = U \Sigma V^H`,
  :math:`\sigma_1 \ge \dots \ge \sigma_{m_0}`. These are the singular
  values of :math:`Q` itself, obtained without forming :math:`Q^HQ`.
* Keep the :math:`r` directions with
  :math:`\sigma_i > \texttt{rank_tol}\cdot\sigma_1` and set
  :math:`Q \leftarrow Q_0 U_{:,1:r}`, which is orthonormal. If
  :math:`r = 0`, :math:`Q` becomes empty.
* Record ``history.basis_singular_values`` :math:`\leftarrow
  (\sigma_1,\dots,\sigma_{m_0})`, all of them, before truncation, and
  ``history.retained_ranks`` :math:`\leftarrow (m, r)`.
* If the QR or SVD raises, log "rank reduction failed" and abandon the
  attempt. Nothing is recorded.

See `Rank decisions`_ for what these numbers mean.

**[H5] Empty span.** If :math:`\dim Q = 0` (log "no usable
directions"): if refreshes remain and ``it`` :math:`<`
``niterations``, replace the span by :math:`c` random vectors, where
:math:`c =` ``expected_dim`` if given, else the dimension :math:`m` of
the span before filtering (at least 1); use one refresh, set
:math:`\Lambda_{\rm prev}=\emptyset`, and go to the next iteration.
Otherwise abandon the attempt.

**[H6] Reduced matrices.** :math:`\hat A = Q^H A Q`,
:math:`\hat B = Q^H B Q` (``Resolvent.rayleigh``), both
:math:`r\times r`.

**[H7] Reduced eigenproblem.** Solve :math:`\hat A W = \hat B W \Lambda`
with ``scipy.linalg.eigh``, which requires :math:`\hat A` Hermitian and
:math:`\hat B` Hermitian positive definite, and returns
:math:`W^H \hat B W = I`. If it raises (for example :math:`\hat B` not
positive definite) or any Ritz value is not finite, log "small
eigensolve failed" and abandon the attempt.

**[H8] Ritz vectors.** :math:`Y \leftarrow Q W`. All :math:`r` Ritz
pairs, inside and outside :math:`\Gamma`, form the new working span.
They are :math:`B`-orthonormal: :math:`Y^H B Y = I`.

**[H9] Update and test:** Procedure U.

Algorithm R: ``mode="right"``
------------------------------------

For non-Hermitian pencils when only right eigenvectors are wanted.
Identical to Algorithm H except at [R7] and [R8].

**[R1]**–**[R6]** As [H1]–[H6]: attempt loop, iteration loop, filter
:math:`Q = PY` (``reduce_sym`` is not allowed), the QR + SVD rank
decision with ``rank_tol`` (same records in ``history``), empty-span
handling, and :math:`\hat A = Q^HAQ`, :math:`\hat B = Q^HBQ`.

**[R7] Reduced eigenproblem.** Solve :math:`\hat A W = \hat B W\Lambda`
with ``scipy.linalg.eig``. Singular :math:`\hat B` gives infinite Ritz
values; these, or a raised error, abandon the attempt as in [H7].

**[R8] Ritz vectors.** :math:`Y \leftarrow QW`. ``eig`` scales each
column of :math:`W` to unit Euclidean norm, and :math:`Q` is
orthonormal, so each Ritz vector has unit Euclidean coefficient norm.
No :math:`B`-normalization is applied.

**[R9] Update and test:** Procedure U, with no left span.

Algorithm B: ``mode="biorth"``
-------------------------------------

For non-Hermitian pencils, iterating a right span :math:`Y` and a left
span :math:`\tilde Y` together. Left Ritz vectors give two-sided
Rayleigh quotients and are returned in ``result.left``.

**[B1] Attempt loop.** As [H1]; restarts append the same number of
random vectors to :math:`Y` and to :math:`\tilde Y`.

**[B2] Iteration loop.** As [H2], doing [B3]–[B9].

**[B3] Filter both spans.** :math:`Q = P\,Y`,
:math:`\tilde Q = \tilde P\,\tilde Y`. :math:`2N` shifted solves per
column pair; ``reduce_sym`` is not available.

**[B4] Rank decision: SVD of the B-Gram matrix**
(:func:`~feast.subspace.analyze_gram`,
:func:`~feast.subspace.apply_cleaning`). Done only if
:math:`\dim Q = m > 0`.

* Form the :math:`m\times m` matrix :math:`G = \tilde Q^H B Q`
  (``Resolvent.gram``) and its SVD :math:`G = U S V^H`,
  :math:`s_1\ge\dots\ge s_m`.
* If the SVD fails, or :math:`s_1 = 0`, or :math:`s_1` is not finite,
  log "invalid B-Gram matrix" and abandon the attempt. Nothing is
  recorded.
* Otherwise keep the :math:`r` directions with
  :math:`s_i > \texttt{eta_tol}\cdot s_1` and cut the rest.
  Record ``history.gram_singular_values`` :math:`\leftarrow
  (s_1,\dots,s_m)` and ``history.retained_ranks`` :math:`\leftarrow
  (m, r)`. When something is cut, the log shows "Kernel cleaning" with
  the singular values and the cut indices.
* Rotate :math:`\tilde Q\leftarrow \tilde Q U`, :math:`Q\leftarrow QV`,
  which makes :math:`\tilde Q^H B Q = S` exactly diagonal; scale kept
  column :math:`i` of both by :math:`s_i^{-1/2}`; remove the cut
  columns. Now :math:`\tilde Q^H B Q \approx I_r`.

The spans are *not* orthonormalized individually and ``rank_tol`` is
not used. See `Rank decisions`_.

**[B5] Empty span.** As [H5], refreshing both spans with the same
number of random vectors.

**[B6] Reduced matrices.** :math:`\hat A = \tilde Q^H A Q`,
:math:`\hat B = \tilde Q^H B Q \approx I_r`.

**[B7] Reduced eigenproblem.** Solve with ``scipy.linalg.eig`` for right
and left eigenvectors:
:math:`\hat A W = \hat B W\Lambda`,
:math:`\tilde W^H \hat A = \Lambda \tilde W^H \hat B`.
Errors and non-finite values abandon the attempt as in [H7].

**[B8] Ritz vectors.** :math:`Y\leftarrow QW`,
:math:`\tilde Y \leftarrow \tilde Q \tilde W`. Columns of :math:`W` and
:math:`\tilde W` have unit Euclidean norm, so for simple Ritz values
:math:`\tilde Y^H B Y` is approximately diagonal with entries
:math:`\tilde w_k^H w_k` of modulus at most 1. It is not rescaled to
the identity.

**[B9] Update and test:** Procedure U, with the left span carried along.

Procedure U: update and test
----------------------------------

Runs after every successful reduced solve, with :math:`r` Ritz values
:math:`\Lambda = (\lambda_1,\dots,\lambda_r)`, the working span
:math:`Y` and, in biorth mode, :math:`\tilde Y`.

**[U1] Record.** Append all :math:`r` Ritz values to
``history.ritz_values``. If ``report_residuals``, append
``Resolvent.residual``\ :math:`(\Lambda, Y, \tilde Y)` to
``history.residuals``. Residuals never influence any decision.

**[U2] Select and save.** Let :math:`I = \{k : \lambda_k` inside
:math:`\Gamma\}` (``Contour.within``). Save
:math:`\Lambda_s = \Lambda_I`, :math:`X_s = Y_{:,I}` and
:math:`\tilde X_s = \tilde Y_{:,I}`. This saved triple is what [E3]
returns unless a later solve replaces it.

**[U3] Distance.** :math:`d = d(\Lambda_s, \Lambda_{\rm prev})`;
append :math:`d` to ``history.hausdorff``. It is :math:`\infty` after
any reset of :math:`\Lambda_{\rm prev}` (first iteration, refresh,
restart) and whenever :math:`\Lambda_s` is empty.

**[U4] Prune.** Let *enough* mean ``expected_dim`` is unset or
:math:`|\Lambda_s| \ge` ``expected_dim``. If ``check_contour``
:math:`> 0`, ``it`` is a multiple of it, some Ritz value is outside
:math:`\Gamma`, :math:`\Lambda_s \ne \emptyset`, and *enough* holds,
replace the working spans by the saved in-contour columns. Otherwise
exterior Ritz vectors stay in the working span, where they can help
the next filter step. This step only changes the working span; [U2]
already excluded exterior values from the result.

**[U5] Replenish.** If ``expected_dim`` is set and
:math:`\dim Y <` ``expected_dim``: if refreshes remain and ``it``
:math:`<` ``niterations``, append ``expected_dim``
:math:`- \dim Y` random vectors (to both spans in biorth mode), use one
refresh, set :math:`\Lambda_{\rm prev}=\emptyset`, and go to the next
iteration, skipping [U6]–[U7]. Otherwise log "requested span dimension
was not recovered" and abandon the attempt.

**[U6] Advance.** :math:`\Lambda_{\rm prev} \leftarrow \Lambda_s`. If
:math:`\dim Y = 0`, log "no candidates left" and abandon the attempt.

**[U7] Convergence test.** Declare convergence and end the run if
*enough* holds, :math:`|\Lambda_s|` equals the size of the previous
in-contour set, and :math:`d <` ``stop_tol``. Because a reset makes
:math:`d=\infty`, at least two successive successful solves without a
reset are needed. Convergence is a statement about eigenvalue sets
only; see :doc:`limitations`.

Rank decisions
--------------------

Each iteration may shrink the working span at [H4], [R4] or [B4], and
later at [U4]. Knowing which decision removed what is the main tool for
diagnosing a hard problem.

What the filtered basis looks like
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Expand a starting vector in eigenvectors. Filtering multiplies the
component along eigenvalue :math:`\lambda_j` by :math:`f(\lambda_j)`.
After an iteration or two, :math:`Q = PY` therefore has, roughly (and
more so the better conditioned the eigenvectors), :math:`k` directions
of size about 1, one per eigenvalue inside :math:`\Gamma`,
then directions of size about :math:`|f(\lambda)|` for eigenvalues near
but outside :math:`\Gamma`, then directions that are only roundoff. The
singular values of :math:`Q` show this staircase. A starting span larger
than :math:`k` (recommended) always produces the lower steps.

Basis rank, ``rank_tol`` ([H4], [R4])
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``rank_tol`` (default :math:`10^{-12}`) is a guard against *numerical
linear dependence*, not the boundary between wanted and unwanted
eigenvalues. It removes directions whose singular value is at roundoff
level relative to :math:`\sigma_1`, which would otherwise make
:math:`\hat B` singular. Directions with small but genuine filter weight
are kept. They produce Ritz values outside :math:`\Gamma`, which [U2]
excludes from the result and [U4] periodically drops from the working
span. The separation between wanted and unwanted is made by
contour membership of Ritz values, not by ``rank_tol``.

For example, a 30×30 non-Hermitian pencil with 3 eigenvalues inside a
16-node circle, started from 6 vectors, has
``history.retained_ranks == [(6, 6), (6, 6), (6, 6)]``: nothing is cut,
and the 3 exterior Ritz values are dropped by contour selection.

Reading the records: ``history.basis_singular_values[j]`` holds all
:math:`m_0` singular values for the *j*-th rank decision and
``history.retained_ranks[j] = (m, r)``. If
:math:`m_0 < m`, the backend's QR already discarded :math:`m-m_0`
roundoff-level columns. The cut lies between :math:`\sigma_r` and
:math:`\sigma_{r+1}`.

Gram rank, ``eta_tol`` ([B4])
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In biorth mode the reduced pencil is :math:`(\tilde Q^H A Q,
\tilde Q^H B Q)`. Even when :math:`Q` and :math:`\tilde Q` each have
full rank, :math:`G = \tilde Q^H B Q` can be singular or nearly so:

* more search vectors than the eigenspace dimension, so that some
  filtered directions are roundoff in one or both spans;
* directions in which the left and right spans are
  :math:`B`-orthogonal, for instance components at
  :math:`\lambda = \infty` (in the kernel of :math:`B`), which the
  filter suppresses but roundoff can leave behind when :math:`B` is
  singular, as for polynomial problems with singular :math:`A_d`;
* near-defective eigenvalues, where left and right eigenvectors are
  nearly :math:`B`-orthogonal.

Keeping such directions would make :math:`\hat B` singular and create
spurious or infinite Ritz values. [B4] removes the directions with
:math:`s_i \le \texttt{eta_tol}\cdot s_1` and rescales the rest so
:math:`\hat B \approx I`. The kept directions are scaled by
:math:`s_i^{-1/2}`, so kept directions with small :math:`s_i` are
amplified. A pairing of two directions that are each of relative size
:math:`\varepsilon` produces :math:`s_i \approx \varepsilon^2`, which is
roughly why the default ``eta_tol`` (:math:`10^{-8}`) is much larger
than the default ``rank_tol`` (:math:`10^{-12}`).

Reading the records: ``history.gram_singular_values[j]`` holds all
:math:`m` singular values of :math:`G` and
``history.retained_ranks[j] = (m, r)``.

Contour pruning ([U4])
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every ``check_contour`` iterations, Ritz vectors whose values lie
outside :math:`\Gamma` are dropped from the working span, unless that
would leave it empty or with fewer than ``expected_dim`` in-contour
values. Pruning is not recorded in ``retained_ranks``; it shows up as a
smaller :math:`m` in the next entry, and in the log as "Removing
out-of-contour Ritz values".

Aligning the history lists
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The lists in :class:`~feast.driver.FeastHistory` are appended by
different steps, so equal indices need not mean the same iteration:

==============================  ================================================
list                            appended at
==============================  ================================================
``retained_ranks``              every successful rank decision ([H4], [R4], [B4])
``basis_singular_values``       same, hermitian and right modes
``gram_singular_values``        same, biorth mode
``ritz_values``, ``hausdorff``  every successful reduced solve ([U1], [U3])
``residuals``                   same, if ``report_residuals``
==============================  ================================================

A rank decision is followed by no Ritz entry when it leaves an empty
span ([H5]) or when the reduced solve then fails ([H7], [R7], [B7]).
Failed rank decisions append nothing. Logging at ``INFO`` level prints
each iteration's records as they are made.

Diagnosing with the records
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **A clear gap at the cut** in the recorded singular values is
  reassuring. A smooth decay through the threshold means the cut is
  arbitrary: the retained rank and the Ritz values can change from one
  iteration to the next, and ``history.hausdorff`` may stall above
  ``stop_tol``. More quadrature nodes sharpen the filter and widen the
  gap.
* **Retained rank below the number of eigenvalues you expect inside**
  means wanted directions were cut or never reached. Compare with
  ``result.count_estimate`` (``count_probes``), use a larger starting
  span, or check whether eigenvalues lie close to :math:`\Gamma`, where
  :math:`f \approx 1/2`.
* **Repeated biorth cuts with** :math:`s_i` **just below** ``eta_tol``
  point to near-defective eigenvalues or to components at
  :math:`\lambda=\infty`. Inspect the cut values in
  ``history.gram_singular_values`` before changing ``eta_tol``.
* **Hermitian mode abandoning attempts at [H7]** means
  :math:`\hat B` was not positive definite: the pencil is not
  Hermitian definite on the filtered space. Use ``mode="right"`` or
  ``"biorth"``.
* **Eigenvalues converge but eigenvectors do not.** [U7] tests
  eigenvalues only, and eigenvalues (especially two-sided ones in biorth
  mode) can converge much faster than eigenvectors when the filter
  separates inside from outside poorly. Use ``report_residuals`` or the
  backend's residual method to check eigenvectors.

Polynomial eigenproblems
------------------------------

No separate algorithm is needed: the driver runs Algorithms H, R and B
unchanged. Only the resolvent differs. For

.. math::

   P(\lambda) x = 0, \qquad P(z) = \sum_{i=0}^d z^i A_i, \quad d \ge 2,

:class:`~feast.backends.numpy_backend.MatrixPolyResolvent` and
``NGPolyResolvent`` present the first companion pencil
:math:`(\mathcal A, \mathcal B)` of size :math:`dn` to the driver, with
:math:`A_0, \dots, A_{d-1}` in the last block row of :math:`\mathcal A`
and :math:`\mathcal B = \operatorname{diag}(I, \dots, I, -A_d)`. Every
step above holds with :math:`(A, B)` replaced by
:math:`(\mathcal A, \mathcal B)`. The method is that of [GPV2022]_.

**Shifted solves without the large matrix.** In [H3], [R3] and [B3],
:math:`(z\mathcal B-\mathcal A)^{-1}\mathcal B Y` is computed from
:math:`\mathcal B Y` by the block recurrences of [GPV2022]_ (Theorem 4).
With :math:`\mathcal B Y` written in blocks :math:`Y_0,\dots,Y_{d-1}`,

.. math::

   X_0 &= P(z)^{-1}\Bigl(-Y_{d-1} - A_d Y_{d-1}
         + \sum_{i=1}^{d} A_i \sum_{j=0}^{i-1} z^{i-1-j} Y_j\Bigr),
   \\
   X_i &= z X_{i-1} - Y_{i-1}, \qquad i = 1,\dots,d-1,

and the left solves use the analogous recurrences for
:math:`(z\mathcal B - \mathcal A)^{-H}`. Only the :math:`n\times n`
matrices :math:`P(z_k)` are factorized, one per node, and cached.

**Extracting eigenvectors.** A companion eigenvector has blocks
:math:`x, \lambda x, \dots, \lambda^{d-1} x`. The first block of each
right Ritz vector is a right eigenvector of :math:`P` (``R.first``);
the last block of each left Ritz vector is a left eigenvector
(``R.last``) ([GPV2022]_, Theorem 2).

**Rank decisions for polynomials.**

* :math:`\mathcal B` is singular exactly when :math:`A_d` is, and its
  kernel is the eigenspace of :math:`\lambda = \infty`. In biorth mode
  the Gram cleaning [B4] removes such directions. It is the counterpart
  of steps 16–22 of Algorithm 1 in [GPV2022]_, with unitary SVD
  rotations in place of a biorthogonalizing diagonalization and the
  relative threshold ``eta_tol`` in place of :math:`d_\ell \approx 0`.
* The companion pencil is generally not Hermitian even when every
  :math:`A_i` is, so use ``mode="biorth"`` or ``"right"``.
* The blocks of a companion eigenvector differ in size by powers of
  :math:`|\lambda|`. The Euclidean QR of [R4] and the singular values
  in the history see the whole stacked vector, which is dominated by its
  last blocks when :math:`|\lambda| > 1` and by its first block when
  :math:`|\lambda| < 1`. The backends' ``residual`` methods, in
  contrast, evaluate :math:`\|P(\lambda)x\|` on the first block only.

**Correspondence with the paper.** In Algorithm 1 of [GPV2022]_, step 2
(factorize :math:`P(z_k)`) happens lazily inside the resolvent; steps
4–15 are [B3]; 16–22 are [B4]; 23–25 are [B6]–[B8]; 26 is [U4]; 27 is
[U3] and [U7]; 28 is the extraction above.
