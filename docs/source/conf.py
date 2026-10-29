import os
import posixpath
import sys
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
from sphinx.ext.autodoc import Documenter, FunctionDocumenter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Pangolin"
copyright = "2022-2026, Justin Domke"
author = "Justin Domke"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    # "sphinx.ext.intersphinx",
    # "sphinx_autodoc_typehints",
    # "myst_parser",
    # "autodoc2",
]

templates_path = ["_templates"]
exclude_patterns = []

doctest_global_setup = """
import numpy as np
import jax
import jax.numpy as jnp
from pangolin import ir, functional
from pangolin import interface as pi
from pangolin.interface import Interface
from pangolin.functional import *
"""


# Configure autodoc
autodoc_packages = [
    {
        "path": "../../pangolin",  # Point this to your code folder
    },
]

# very hard to get sphinx to document PyTree arguments
# the current set of incantations is:
# 1) Manually document any PyTree arguments
# 2) Turn off napoleon_preprocess_types
# 3) do not use autodoc_typehints extension (but do use autodoc)
# 4) Turn on from __future__ import annotations


# ALL napoleon options
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True
napoleon_type_aliases = {
    "PyTree": "PyTree",
    "PyTree[Node]": "PyTree[Node]",
    "PyTree[InfixRV]": "PyTree[InfixRV]",
}

# autodoc options
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_inherit_docstrings = False
# autodoc_preserve_defaults = True
# autodoc_typehints_format = "short"

# typehints_use_rtype = False

default_role = "any"

autodoc_type_aliases = {
    "RVLike": "RVLike",
    "ArrayLike": "ArrayLike",
}

viewcode_follow_imported_members = False


# The combinators use overloads to type their decorator and direct-call forms.
# Those overloads help static type checkers, but showing every one obscures the
# much smaller runtime API. This opt-in documenter renders one unannotated
# runtime signature; its directives use reader-facing types written in the docstring.
class CleanFunctionDocumenter(FunctionDocumenter):
    objtype = "cleanfunction"
    directivetype = "function"
    priority = FunctionDocumenter.priority + 1

    def format_signature(self, **kwargs):
        return Documenter.format_signature(self, **kwargs)


def configure_sidebar_navigation(app, pagename, templatename, context, doctree):
    """Keep API navigation concise without changing links to individual functions or guides."""
    toctree = context["toctree"]

    def navigation_tree(**kwargs):
        tree = BeautifulSoup(toctree(**kwargs), "html.parser")
        for link in tree.find_all("a", href=True):
            path = urlsplit(link["href"]).path
            target = (
                posixpath.normpath(posixpath.join(posixpath.dirname(pagename), path))
                if path
                else pagename
            )
            if target.startswith(("generated/", "api/")) or target.removesuffix(".html") in {
                "interface.examples",
                "interface.broadcasting",
                "interface.indexing",
            }:
                link.find_parent("li").decompose()
        for subtree in reversed(tree.find_all("ul")):
            if not subtree.find("li", recursive=False):
                subtree.decompose()
        return str(tree)

    context["toctree"] = navigation_tree


def setup(app):
    app.add_autodocumenter(CleanFunctionDocumenter)
    # Install the filter before the theme renders and caches its navigation.
    app.connect("html-page-context", configure_sidebar_navigation, priority=400)

# intersphinx_mapping = {
#     "python": ("https://docs.python.org/3", None),
#     "numpy": ("https://numpy.org/doc/stable/", None),
# }


# Options for HTML output
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_theme_options = {"max_navbar_depth": 2}
html_static_path = ["_static"]
html_css_files = ["layout.css"]
html_logo = "_static/pangolin-logo-small.png"
html_sidebars = {"_modules/**": []}

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}
