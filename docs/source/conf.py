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

napoleon_numpy_docstring = True
napoleon_google_docstring = False  # Set to True if you mix styles
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_rtype = False

autodoc_typehints = "description"
autodoc_member_order = "bysource"
default_role = "any"

autodoc_type_aliases = {
    "RVLike": "`RVLike`",
    "ArrayLike": "ArrayLike",
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

html_logo = "_static/pangolin-logo-small.png"

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}
