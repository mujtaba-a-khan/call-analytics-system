"""
UI Package for Call Analytics System

This package contains the Streamlit-based user interface.
"""

__version__ = '1.0.0'

def run_app():
    """Launch the Streamlit application"""
    from .app import main
    return main()

__all__ = ['run_app']