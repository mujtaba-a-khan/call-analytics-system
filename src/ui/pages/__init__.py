"""
UI Pages Package for Call Analytics System

This package exports all page modules for the Streamlit application.
Each page represents a major feature area of the application.
"""

# Import page modules
from .dashboard import DashboardPage, render_dashboard_page
from .analysis import AnalysisPage, render_analysis_page
from .qa_interface import QAInterface, render_qa_interface
from .upload import UploadPage, render_upload_page

# Define package exports
__all__ = [
    # Page classes
    'DashboardPage',
    'AnalysisPage',
    'QAInterface',
    'UploadPage',
    
    # Render functions
    'render_dashboard_page',
    'render_analysis_page',
    'render_qa_interface',
    'render_upload_page'
]

# Package version
__version__ = '1.0.0'