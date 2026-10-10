import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "feast"
author = "Jay Gopalakrishnan"
release = importlib.import_module("feast").__version__
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]
autodoc_mock_imports = ["ngsolve", "netgen"]
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_use_ivar = True
exclude_patterns = ["_build"]
html_theme = "alabaster"
