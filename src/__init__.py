"""
Call Analytics System - Main Package

Provides high-level convenience functions and lazy exports for the rest of
the project while keeping import time low.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, NamedTuple

# Disable Hugging Face tokenizer parallelism to avoid fork-related warnings.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Package metadata
__title__ = "Call Analytics System"
__version__ = "1.0.0"
__author__ = "Mujtaba Khan"
__license__ = "MIT"
__copyright__ = "Copyright 2025"


def get_system_info() -> dict[str, Any]:
    """
    Gather system information and optional ML capabilities.
    """
    import platform
    import sys
    from datetime import datetime

    system_info: dict[str, Any] = {
        "version": __version__,
        "title": __title__,
        "python_version": platform.python_version(),
        "python_version_info": sys.version_info[:3],
        "platform": platform.platform(),
        "timestamp": datetime.now().isoformat(),
        "python_3_13_compatible": sys.version_info >= (3, 13),
    }

    try:
        from .ml import get_ml_capabilities

        system_info["capabilities"] = get_ml_capabilities()
    except ImportError:
        system_info["capabilities"] = {"ml_available": False}

    return system_info


def run_app() -> Any:
    """
    Launch the Streamlit application.
    """
    try:
        from .ui import run_app as _run_app

        return _run_app()
    except ImportError as exc:
        logging.error("Failed to import UI module: %s", exc)
        raise RuntimeError(
            "UI modules not available. Please ensure Streamlit is installed: "
            "pip install streamlit>=1.35.0"
        ) from exc


def setup_logging(
    level: str = "INFO",
    log_dir: str | Path | None = None,
    *,
    console_output: bool = True,
    file_output: bool = True,
    structured_logs: bool = False,
) -> logging.Logger:
    """
    Configure the project-wide logging defaults and return a logger instance.
    """
    resolved_dir = Path(log_dir) if log_dir is not None else None

    try:
        from .utils.logger import setup_logging as _setup_logging

        _setup_logging(
            log_level=level,
            log_dir=resolved_dir,
            console_output=console_output,
            file_output=file_output,
            structured_logs=structured_logs,
        )
    except ImportError:
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


class _LazyExport(NamedTuple):
    module: str
    attr: str


# Lazy export module paths that are reused multiple times.
CHROMA_CLIENT_MODULE = "vectordb.chroma_client"
VALIDATORS_MODULE = "utils.validators"


_LAZY_EXPORTS: dict[str, _LazyExport] = {
    # Core modules
    "CallRecord": _LazyExport("core.data_schema", "CallRecord"),
    "AudioProcessor": _LazyExport("core.audio_processor", "AudioProcessor"),
    "CSVProcessor": _LazyExport("core.csv_processor", "CSVProcessor"),
    "StorageManager": _LazyExport("core.storage_manager", "StorageManager"),
    "LabelingEngine": _LazyExport("core.labeling_engine", "LabelingEngine"),
    # Analysis
    "MetricsCalculator": _LazyExport("analysis.aggregations", "MetricsCalculator"),
    "SemanticSearchEngine": _LazyExport("analysis.semantic_search", "SemanticSearchEngine"),
    "QueryInterpreter": _LazyExport("analysis.query_interpreter", "QueryInterpreter"),
    "AdvancedFilters": _LazyExport("analysis.filters", "AdvancedFilters"),
    # ML
    "WhisperSTT": _LazyExport("ml.whisper_stt", "WhisperSTT"),
    "EmbeddingManager": _LazyExport("ml.embeddings", "EmbeddingManager"),
    "get_ml_capabilities": _LazyExport("ml", "get_ml_capabilities"),
    # Vector DB (with backward compatible aliases)
    "ChromaClient": _LazyExport(CHROMA_CLIENT_MODULE, "ChromaClient"),
    "ChromaDBClient": _LazyExport(CHROMA_CLIENT_MODULE, "ChromaClient"),
    "VectorDBError": _LazyExport(CHROMA_CLIENT_MODULE, "VectorDBError"),
    "DocumentIndexer": _LazyExport("vectordb.indexer", "DocumentIndexer"),
    "VectorIndexer": _LazyExport("vectordb.indexer", "DocumentIndexer"),
    "DocumentRetriever": _LazyExport("vectordb.retriever", "DocumentRetriever"),
    "VectorRetriever": _LazyExport("vectordb.retriever", "DocumentRetriever"),
    # Utilities
    "get_logger": _LazyExport("utils.logger", "get_logger"),
    "validate_phone": _LazyExport(VALIDATORS_MODULE, "validate_phone"),
    "validate_phone_number": _LazyExport(VALIDATORS_MODULE, "validate_phone"),
    "validate_email": _LazyExport(VALIDATORS_MODULE, "validate_email"),
    "format_duration": _LazyExport("utils.formatters", "format_duration"),
}


def __getattr__(name: str) -> Any:
    """
    Lazily resolve heavyweight modules to avoid eager imports.
    """
    try:
        module_name, attr = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc

    module = __import__(f"{__name__}.{module_name}", fromlist=[attr])
    return getattr(module, attr)


def __dir__() -> list[str]:
    """Ensure lazy exports show up in interactive introspection tools."""
    return sorted(set(__all__) | set(_LAZY_EXPORTS.keys()) | set(globals()))
