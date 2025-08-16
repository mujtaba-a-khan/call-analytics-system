"""
Call Analytics System - Main Package

A comprehensive system for analyzing call center data with speech-to-text,
semantic search, and advanced analytics capabilities.
"""

# Import core modules
from .core import (
    CallRecord,
    AudioProcessor,
    CSVProcessor,
    StorageManager,
    LabelingEngine
)

# Import analysis modules
from .analysis import (
    MetricsCalculator,
    SemanticSearchEngine,
    QueryInterpreter,
    AdvancedFilters
)

# Import ML modules
from .ml import (
    WhisperSTT,
    EmbeddingManager,
    get_ml_capabilities
)

# Import vector database modules
from .vectordb import (
    ChromaDBClient,
    VectorIndexer,
    VectorRetriever
)

# Import UI modules
from .ui import run_app

# Import utilities
from .utils import (
    setup_logging,
    get_logger,
    validate_phone_number,
    format_duration
)

# Package metadata
__title__ = 'Call Analytics System'
__version__ = '1.0.0'
__author__ = 'Mujtaba Khan'
__license__ = 'MIT'
__copyright__ = 'Copyright 2025'

# Define main exports
__all__ = [
    # Core
    'CallRecord',
    'AudioProcessor',
    'CSVProcessor',
    'StorageManager',
    'LabelingEngine',
    
    # Analysis
    'MetricsCalculator',
    'SemanticSearchEngine',
    'QueryInterpreter',
    'AdvancedFilters',
    
    # ML
    'WhisperSTT',
    'EmbeddingManager',
    'get_ml_capabilities',
    
    # Vector DB
    'ChromaDBClient',
    'VectorIndexer',
    'VectorRetriever',
    
    # UI
    'run_app',
    
    # Utils
    'setup_logging',
    'get_logger',
    'validate_phone_number',
    'format_duration',
    
    # Metadata
    '__version__',
    '__author__'
]

def get_system_info():
    """
    Get comprehensive system information and capabilities.
    
    Returns:
        Dictionary containing system information and status
    """
    from datetime import datetime
    import platform
    
    system_info = {
        'version': __version__,
        'title': __title__,
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'timestamp': datetime.now().isoformat(),
        'capabilities': get_ml_capabilities()
    }
    
    return system_info