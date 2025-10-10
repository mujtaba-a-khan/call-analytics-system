"""
Core Package for Call Analytics System

This package contains the core business logic and data processing modules
including audio processing, data schema, storage management, and labeling.
"""

# Package version
__version__ = "1.0.0"

# Define available exports - these will be loaded on demand
__all__ = [
    # Data schema
    "CallRecord",
    "AudioFile",
    "TranscriptionResult",
    "AnalyticsMetrics",
    "ValidationError",
    # Audio processing
    "AudioProcessor",
    "AudioFormat",
    "AudioMetadata",
    # CSV processing
    "CSVProcessor",
    "CSVExporter",
    # Labeling
    "LabelingEngine",
    "CallLabeler",
    "LabelingRule",
    "RuleCondition",
    # Storage
    "StorageManager",
    "DataStore",
    "StorageBackend",
]


def __getattr__(name):
    """
    Lazy loading implementation for core modules.
    Only imports modules when they are actually accessed.
    """
    schema_exports = {
        "CallRecord",
        "AudioFile",
        "TranscriptionResult",
        "AnalyticsMetrics",
        "ValidationError",
    }
    audio_exports = {"AudioProcessor", "AudioFormat", "AudioMetadata"}
    csv_exports = {"CSVProcessor", "CSVExporter"}
    labeling_exports = {"LabelingEngine", "CallLabeler", "LabelingRule", "RuleCondition"}
    storage_exports = {"StorageManager", "DataStore", "StorageBackend"}

    # Data schema imports
    if name in schema_exports:
        from . import data_schema

        return getattr(data_schema, name)

    # Audio processor imports
    if name in audio_exports:
        from . import audio_processor

        return getattr(audio_processor, name)

    # CSV processor imports
    if name in csv_exports:
        from . import csv_processor

        return getattr(csv_processor, name)

    # Labeling engine imports
    if name in labeling_exports:
        from . import labeling_engine

        return getattr(labeling_engine, name)

    # Storage manager imports
    if name in storage_exports:
        from . import storage_manager

        return getattr(storage_manager, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
