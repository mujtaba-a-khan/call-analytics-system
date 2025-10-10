"""
Call Analytics System - Main Package

A comprehensive system for analyzing call center data with speech-to-text,
semantic search, and advanced analytics capabilities.
"""

import os

# Disable Hugging Face tokenizer parallelism to avoid fork-related warnings.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Package metadata
__title__ = "Call Analytics System"
__version__ = "1.0.0"
__author__ = "Mujtaba Khan"
__license__ = "MIT"
__copyright__ = "Copyright 2025"

# Lazy imports to prevent circular dependencies and reduce memory overhead


def get_system_info():
    """
    Get comprehensive system information and capabilities.

    Returns:
        Dict[str, Any]: Dictionary containing system information and status
    """
    import platform
    import sys
    from datetime import datetime

    system_info = {
        "version": __version__,
        "title": __title__,
        "python_version": platform.python_version(),
        "python_version_info": sys.version_info[:3],
        "platform": platform.platform(),
        "timestamp": datetime.now().isoformat(),
        "python_3_13_compatible": sys.version_info >= (3, 13),
    }

    # Try to get ML capabilities if available
    try:
        from .ml import get_ml_capabilities

        system_info["capabilities"] = get_ml_capabilities()
    except ImportError:
        system_info["capabilities"] = {"ml_available": False}

    return system_info


def run_app():
    """
    Launch the Streamlit application.

    This is the main entry point for the UI.
    """
    try:
        from .ui import run_app as _run_app

        return _run_app()
    except ImportError as e:
        import logging

        logging.error(f"Failed to import UI module: {e}")
        raise RuntimeError(
            "UI modules not available. Please ensure Streamlit is installed: "
            "pip install streamlit>=1.35.0"
        ) from e


def setup_logging(level="INFO", log_file=None):
    """
    Initialize logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file

    Returns:
        logging.Logger: Configured logger instance
    """
    try:
        from .utils.logger import setup_logging as _setup_logging

        return _setup_logging(level=level, log_file=log_file)
    except ImportError:
        # Fallback to basic logging if utils not available
        import logging

        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        return logging.getLogger(__name__)


# Define what's available for import
__all__ = [
    "__version__",
    "__author__",
    "__title__",
    "__license__",
    "get_system_info",
    "run_app",
    "setup_logging",
]


# Only import heavy modules when explicitly requested
def __getattr__(name):
    """
    Lazy loading of heavy modules to prevent circular imports and reduce startup time.

    This is called when an attribute is accessed but not found in the module.
    """
    # Core module imports
    if name == "CallRecord":
        from .core.data_schema import CallRecord

        return CallRecord
    elif name == "AudioProcessor":
        from .core.audio_processor import AudioProcessor

        return AudioProcessor
    elif name == "CSVProcessor":
        from .core.csv_processor import CSVProcessor

        return CSVProcessor
    elif name == "StorageManager":
        from .core.storage_manager import StorageManager

        return StorageManager
    elif name == "LabelingEngine":
        from .core.labeling_engine import LabelingEngine

        return LabelingEngine

    # Analysis module imports
    elif name == "MetricsCalculator":
        from .analysis.aggregations import MetricsCalculator

        return MetricsCalculator
    elif name == "SemanticSearchEngine":
        from .analysis.semantic_search import SemanticSearchEngine

        return SemanticSearchEngine
    elif name == "QueryInterpreter":
        from .analysis.query_interpreter import QueryInterpreter

        return QueryInterpreter
    elif name == "AdvancedFilters":
        from .analysis.filters import AdvancedFilters

        return AdvancedFilters

    # ML module imports
    elif name == "WhisperSTT":
        from .ml.whisper_stt import WhisperSTT

        return WhisperSTT
    elif name == "EmbeddingManager":
        from .ml.embeddings import EmbeddingManager

        return EmbeddingManager
    elif name == "get_ml_capabilities":
        from .ml import get_ml_capabilities

        return get_ml_capabilities

    # Vector DB imports
    elif name == "ChromaDBClient":
        from .vectordb.chroma_client import ChromaDBClient

        return ChromaDBClient
    elif name == "VectorIndexer":
        from .vectordb.indexer import VectorIndexer

        return VectorIndexer
    elif name == "VectorRetriever":
        from .vectordb.retriever import VectorRetriever

        return VectorRetriever

    # Utility imports
    elif name == "get_logger":
        from .utils.logger import get_logger

        return get_logger
    elif name == "validate_phone_number":
        from .utils.validators import validate_phone_number

        return validate_phone_number
    elif name == "format_duration":
        from .utils.formatters import format_duration

        return format_duration

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
