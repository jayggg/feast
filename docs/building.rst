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

This is exactly the command CI runs, so a build that succeeds locally
without warnings will also succeed online. Open the result in a
browser. On macOS::

    open docs/_build/html/index.html

Rerun the Sphinx command after editing docstrings, documentation pages,
or demos; the API reference is regenerated automatically. A separate
``sphinx-apidoc`` step is unnecessary because ``api.rst`` and
``ngsolve.rst`` already select the modules to document. If a page looks
stale, delete ``docs/_build`` and rebuild.

Live preview
------------------

While writing, ``sphinx-autobuild`` (installed by the ``docs`` extra)
serves the documentation locally and rebuilds and reloads the browser
on every save::

    sphinx-autobuild docs docs/_build/html --watch src --watch demos

Then open http://127.0.0.1:8000. The ``--watch`` options also rebuild
when docstrings or demo scripts change. Stop the server with Ctrl-C.

Demos in the documentation
--------------------------------

The scripts in ``demos/`` are shown in :doc:`demos` and in
:doc:`getting_started` with ``literalinclude`` directives, so the
documentation always shows the current files. Sphinx does not run them.
They are run by ``tests/test_demos.py``, which also fails if a demo is
not referenced in ``docs/demos.rst``. A ``literalinclude`` pointing at a
missing file produces a warning, which ``-W`` turns into an error.

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

GitHub Actions runs two workflows from ``.github/workflows``:

* ``ci.yml`` on every push and pull request: ``ruff check``; the test
  suite without NGSolve on Python 3.10 and 3.13 (this proves the core
  does not need it); and the full suite, including NGSolve tests and
  demos, with NGSolve installed from PyPI.
* ``docs.yml`` builds the documentation with the command above on every
  push and pull request. On pushes to ``main`` it publishes the HTML to
  GitHub Pages at https://jayggg.github.io/feast/.

Publishing requires a one-time repository setting: Settings → Pages →
Build and deployment → Source: GitHub Actions.

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
