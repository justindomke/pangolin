import os
import sys

sys.path.insert(0, os.path.abspath("../../pangolin/"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Pangolin"
copyright = "2025, Justin Domke"
author = "Justin Domke"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    # "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    # "myst_parser",
    # "autodoc2",
]

templates_path = ["_templates"]
exclude_patterns = []


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
    "Shape": "Shape",
    "PyTree": "PyTree",
    "PyTree[Node]": "PyTree[Node]",
}

# autodoc options
# autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_inherit_docstrings = False
# autodoc_preserve_defaults = True
# autodoc_typehints_format = "short"

autodoc_typehints = "none"
typehints_use_rtype = False

default_role = "any"

autodoc_type_aliases = {
    "Shape": "`Shape`",
    "RVLike": "RVLike",
    "ArrayLike": "ArrayLike",
    "PyTree": "PyTree",
}

# intersphinx_mapping = {
#     "python": ("https://docs.python.org/3", None),
#     "numpy": ("https://numpy.org/doc/stable/", None),
# }


# Options for HTML output
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_logo = "_static/pangolin-logo-small.png"

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}
