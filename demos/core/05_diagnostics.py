"""Diagnostics: logging levels, iteration history and checking results.

FEAST reports through Python's logging module, under the logger named
"feast". Configure it like any other logger:

* level INFO shows the parameters and every iteration (as in the other
  demos);

* level WARNING shows only problems, such as nonconvergence;

* logging.getLogger("feast").setLevel(...) sets the minimum severity
  for messages from FEAST and its backend modules, without changing
  logger levels for other libraries. Output handlers may apply their
  own severity thresholds.

The same information is kept in result.history for use in code:
ritz_values, hausdorff (relative change of the in-contour eigenvalue
set) and, with report_residuals=True, residuals. A deliberately cheap
contour (npts=4) makes the iteration take several steps.

Rank decisions are recorded too: basis_singular_values (hermitian and
right modes) or gram_singular_values (biorth mode), with
retained_ranks. A clear gap in the singular values shows how many
directions the contour really captured.

Convergence means the eigenvalue set stopped changing. It does not
prove accuracy or that every enclosed eigenvalue was found, so check
residuals and contour membership yourself, or use the completeness
check (which is stochastic, with count_probes > 0), as shown below.

The result of the completeness check, run.count_estimate.unexplained,
is the "unexplained weight": the sum of the contour's filter values
f(lambda) = C.filter(lambda) over all eigenvalues the run did not
return. f is about 1 well inside the contour and near 0 far outside, so
the unexplained weight is near 0 when nothing was missed and about 1
per missed eigenvalue. The "±" after it is one standard error, the
noise level of the random estimate. FEAST warns when the value exceeds
both 1/4 and three standard errors. See feast.count for details.

Three runs show this: one that converges, the same run stopped early
(not converged, but nothing missing), and one whose span is too small
for the enclosed eigenvalues (an eigenvalue is missed and flagged).

Requires: numpy, scipy.
Run:      python demos/core/05_diagnostics.py
"""

import logging
import sys

import numpy as np
import scipy.sparse as sp

from feast import Contour, feast
from feast.backends.numpy_backend import MatrixResolvent, NumpySpan

logging.basicConfig(
    level=logging.INFO,
    format="  %(message)s",  # indent to sit in the blocks
    stream=sys.stdout)
logging.getLogger("feast").setLevel(logging.WARNING)  # FEAST: problems only

n = 200
h = 1.0 / (n + 1)
A = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n)) / h**2
k = np.arange(1, n + 1)
exact = 4 / h**2 * np.sin(k * np.pi * h / 2)**2

C = Contour.circle(center=50.0, radius=45.0, npts=4)
R = MatrixResolvent(A)

# The filter weights the completeness check is measured in.
weights = C.filter(exact[:5]).real
print("-" * 70)
print("Setup: filter weights f(lambda) of the five smallest eigenvalues")
print(f"  eigenvalues: {np.round(exact[:5], 2)}")
print(f"  weights:     {np.round(weights, 3)}  (first three inside)")

RANK_TOL = 1e-12  # the default, passed explicitly to show it


def report(run, details=True):
    """Print the diagnostics of one FEAST run; details=False prints only
    the summary and the completeness check."""
    hist = run.history
    print(f"  converged:   {run.converged} after {run.iterations} iterations")
    print(f"  eigenvalues: {np.sort(run.eigenvalues.real)}")
    if details:
        print_history_and_ranks(hist)
    est = run.count_estimate
    print(f"  Completeness check ({len(est.samples)} probes)")
    print(f"    unexplained weight: {est.unexplained:.2g} "
          f"± {est.unexplained_stderr:.2g}")
    print(f"    missing suspected:  {est.suggests_missing()}")


def print_history_and_ranks(hist):

    # History entries cover all Ritz values, including those outside the
    # contour (never accurate); restrict to the inside ones with C.within.
    print("  History")
    print("    iteration  inside/all  rel. Hausdorff  max inside residual")
    for i, (ritz, dist, res) in enumerate(zip(hist.ritz_values, hist.hausdorff,
                                              hist.residuals),
                                          start=1):
        inside = C.within(ritz)
        count = f"{inside.sum()}/{len(ritz)}"
        print(f"    {i:9d}  {count:>10}  {dist:14.2e}"
              f"  {res[inside].max():19.2e}")

    # Singular values relative to the largest, cut at rank_tol.
    print(f"  Rank decisions (rank_tol = {RANK_TOL:.0e})")
    for sv, (filtered, kept) in zip(hist.basis_singular_values,
                                    hist.retained_ranks):
        values = np.array2string(
            sv / sv[0], formatter={"float_kind": lambda x: f"{x:.1e}"})
        print(f"    {values}  kept {kept} of {filtered}")


options = dict(mode="hermitian",
               seed=4,
               rank_tol=RANK_TOL,
               report_residuals=True,
               count_probes=16)

# 1. A converged run. Five vectors, three enclosed eigenvalues: note the
#    gap after the third singular value. The small unexplained weight
#    (about 0.02) is the weight of the eigenvalues outside the contour,
#    which leak through this cheap 4-node filter: exactly
#    sum(C.filter(exact[3:])) = 0.033.
print("-" * 70)
print("Run 1: converged")
result = feast(R, C, NumpySpan.random(n, 5, seed=4), **options)
report(result)

lam = np.sort(result.eigenvalues.real)
residuals = R.residual(result.eigenvalues, result.right)
assert result.converged
assert np.allclose(lam, exact[:3], rtol=1e-9)
assert C.within(result.eigenvalues).all()
assert np.all(residuals < 1e-6 * np.abs(result.eigenvalues))
assert len(result.history.hausdorff) == result.iterations
assert not result.count_estimate.suggests_missing()

# 2. The same run stopped after 2 iterations. A warning is logged (it
#    appears at the top of the block), and the result holds
#    the last computed in-contour pairs with converged=False. All three
#    eigenvalues are already present, only not converged, so the
#    completeness check finds nothing missing: nonconvergence and
#    missing eigenvalues are different failures.
print("-" * 70)
print("Run 2: limited to 2 iterations")
short = feast(R, C, NumpySpan.random(n, 5, seed=4), niterations=2, **options)
report(short)

assert not short.converged
assert short.iterations == 2
assert len(short.eigenvalues) == 3
assert not short.count_estimate.suggests_missing()

# 3. A missed eigenvalue: two vectors cannot hold three eigenvectors.
#    FEAST never converges (50 iterations, so only the summary is shown).
#    The completeness check runs after the iteration stops. Its
#    unexplained weight (about 0.55) is mostly the missed eigenvalue
#    9.87, whose weight is 0.61.
print("-" * 70)
print("Run 3: span of 2 vectors for 3 enclosed eigenvalues")
small = feast(R, C, NumpySpan.random(n, 2, seed=4), **options)
report(small, details=False)
print("-" * 70)

assert not small.converged
assert len(small.eigenvalues) == 2
assert small.count_estimate.suggests_missing()
