"""
Requirements tracker FastAPI app used for professional requirement management evidence.

This module exposes the FastAPI app and helper utilities so other parts of the
project (tests, seed scripts, build tooling) can import them easily.
"""

from .app import app  # noqa: F401
