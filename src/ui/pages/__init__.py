"""
UI Pages Package for Call Analytics System

This package exports all page modules for the Streamlit application.
Each page represents a major feature area of the application.
"""

# Package version
__version__ = '1.0.0'

# Define available exports
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

def __getattr__(name):
    """
    Lazy loading of page modules.
    Pages are imported only when accessed to reduce memory overhead.
    """
    # Dashboard page
    if name in ['DashboardPage', 'render_dashboard_page']:
        from . import dashboard
        return getattr(dashboard, name)

    # Analysis page
    elif name in ['AnalysisPage', 'render_analysis_page']:
        from . import analysis
        return getattr(analysis, name)

    # QA interface
    elif name in ['QAInterface', 'render_qa_interface']:
        from . import qa_interface
        return getattr(qa_interface, name)

    # Upload page
    elif name in ['UploadPage', 'render_upload_page']:
        from . import upload
        return getattr(upload, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
