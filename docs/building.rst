Building and maintaining documentation
============================================

The API reference is generated from the Python docstrings using Sphinx
``autodoc``. The ``napoleon`` extension parses their NumPy-style
``Parameters``, ``Returns``, ``Attributes``, and ``Notes`` sections.
The narrative pages and API directives are reStructuredText files in
``docs/``.

Install and build
-----------------------

From the repository root, in your chosen Python environment::

    python -m pip install -e '.[docs]'
    python -m sphinx -b html -W --keep-going docs docs/_build/html

Open ``docs/_build/html/index.html`` in a browser. On macOS::

    open docs/_build/html/index.html

Rerun the Sphinx command after editing docstrings or documentation pages;
the API reference is regenerated automatically. A separate
``sphinx-apidoc`` step is unnecessary because ``api.rst`` and
``ngsolve.rst`` already select the modules to document.

NGSolve and Netgen imports are mocked during documentation builds so
the reference can be generated without those optional native libraries.
Documentation builds do not execute code examples or test backend
behavior. Validate those separately in an NGSolve environment::

    python -m pytest

If NGSolve depends on an existing ``PYTHONPATH``, preserve it when adding
the source tree, for example ``PYTHONPATH="src:$PYTHONPATH"`` on a POSIX
shell. Replacing it can hide the installation or mix incompatible native
libraries.

Automation
----------------

The same install and build commands can run in CI. ``-W`` makes Sphinx
warnings fail the build, and ``--keep-going`` collects all warnings in
one run. Publish the contents of ``docs/_build/html`` using your chosen
static hosting service. No hosting or CI provider is configured here.

Keep the Python source, ``docs/*.rst``, and ``docs/conf.py`` in version
control. Generated HTML, bytecode, caches, and package metadata are
excluded by ``.gitignore``.

Writing docstrings
------------------------

Document behavior, mathematical conventions, assumptions, and mutation
semantics. Use the current public API in examples and keep architecture
rationale in :doc:`architecture`. An example function docstring is::

    """Compute a reduced Gram matrix.

    Parameters
    ----------
    left, right : Span
        Spans stored as columns; neither input is modified.

    Returns
    -------
    numpy.ndarray
        Matrix with shape ``(left.dim, right.dim)``.
    """

Use double backticks for inline code, Sphinx roles for API cross-references,
and ``.. math::`` for displayed mathematics. Indented ASCII matrices need
a literal-block introducer (``::``) to preserve their layout.

See the official `autodoc documentation
<https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_
and `NumPy-style docstring support
<https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`_.
