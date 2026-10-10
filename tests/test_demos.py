"""Run every demo script and check that the docs include each one.

Demos end with assertions, so running them doubles as a smoke test of
the public API. Demos under ``demos/ngsolve`` carry the ``ngsolve``
marker and are skipped when NGSolve is not installed.
"""

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEMOS = sorted((ROOT / "demos").glob("*/[0-9]*.py"))


def _param(path):
    rel = path.relative_to(ROOT / "demos").as_posix()
    marks = [pytest.mark.ngsolve] if rel.startswith("ngsolve/") else []
    return pytest.param(path, id=rel, marks=marks)


def test_demos_found():
    assert DEMOS, "no demo scripts found"


@pytest.mark.parametrize("path", [_param(p) for p in DEMOS])
def test_demo_runs(path, monkeypatch):
    if path.parent.name == "ngsolve":
        pytest.importorskip("ngsolve")
    monkeypatch.chdir(path.parent)
    runpy.run_path(str(path), run_name="__main__")


def test_every_demo_is_in_docs():
    docs = (ROOT / "docs" / "demos.rst").read_text()
    missing = [
        p.relative_to(ROOT).as_posix()
        for p in DEMOS
        if f"../{p.relative_to(ROOT).as_posix()}" not in docs
    ]
    assert not missing, f"add these to docs/demos.rst: {missing}"
