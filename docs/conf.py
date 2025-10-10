# Configuration file for the Sphinx documentation builder.

project = 'call-analytics-system'
copyright = '2025, Mujtaba Khan'
author = 'Mujtaba Khan'
release = '1.0.0'


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


html_static_path = ['_static']


extensions = [
    "myst_parser",                # Markdown in Sphinx
    "sphinx.ext.autodoc",         # API from docstrings
    "sphinx.ext.napoleon",        # Google/NumPy docstrings
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.mermaid",      # Mermaid diagrams
    "sphinx.ext.githubpages",
]

html_theme = "sphinx_rtd_theme"

autodoc_mock_imports = ["whisper", "torch", "ffmpeg"]
