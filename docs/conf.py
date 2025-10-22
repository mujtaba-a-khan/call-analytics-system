import sys
from pathlib import Path

# Configuration file for the Sphinx documentation builder.

DOCS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_DIR.parent
SOURCE_DIR = PROJECT_ROOT / "src"

# Ensure modules from the project can be imported when building API docs
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SOURCE_DIR))

project = 'call-analytics-system'
copyright = '2025, Mujtaba Khan'
author = 'Mujtaba Khan'
release = '1.0.0'


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


html_static_path = ['_static']

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}


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
html_css_files = ["custom.css"]

autodoc_mock_imports = ["whisper", "torch", "ffmpeg"]

myst_enable_extensions = [
    "linkify",             # Auto-detect bare links
    "tasklist",            # GitHub-style task lists
]

myst_heading_anchors = 4


def _inject_dynamic_anchors(_app, docname, source):
    """Adjust problematic markdown without editing original files."""
    text = source[0]

    if docname == "ai-coding":
        target_heading = "## Codex-Assisted Development Workflow"
        marker = "(ai-assisted-development-workflow)="
        if target_heading in text and marker not in text:
            source[0] = text.replace(
                target_heading,
                f"{marker}\n\n{target_heading}",
                1,
            )
            text = source[0]

    if docname == "refactoring":
        marker = "(references)="
        if marker not in text:
            source[0] = text.rstrip() + f"\n\n{marker}\n"
            text = source[0]

    if docname == "git-journal":
        trimmed = text.rstrip()
        if trimmed.endswith("\n---") or trimmed.endswith("\n---\r"):
            source[0] = trimmed + "\n\n.. raw:: html\n\n   <!-- spacer -->\n"

    if docname == "ide":
        # Fence declared as json but contains Markdown; downgrade to text.
        snippet = "```json\n## My Development Workflow"
        if snippet in text:
            source[0] = text.replace(snippet, "```text\n## My Development Workflow", 1)


def setup(app):
    app.connect("source-read", _inject_dynamic_anchors)
