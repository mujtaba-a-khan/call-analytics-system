"""
Core Package for Call Analytics System

This package contains the core business logic and data processing modules
including audio processing, data schema, storage management, and labeling.
"""

# Import data schema
from .data_schema import (
    CallRecord,
    AudioFile,
    TranscriptionResult,
    AnalyticsMetrics,
    ValidationError
)

# Import audio processor
from .audio_processor import (
    AudioProcessor,
    AudioFormat,
    AudioMetadata
)

# Import CSV processor
from .csv_processor import (
    CSVProcessor,
    CSVExporter
)

# Import labeling engine
from .labeling_engine import (
    LabelingEngine,
    CallLabeler,
    LabelingRule,
    RuleCondition
)

# Import storage manager
from .storage_manager import (
    StorageManager,
    DataStore,
    StorageBackend
)

# Define package exports
__all__ = [
    # Data schema
    'CallRecord',
    'AudioFile',
    'TranscriptionResult',
    'AnalyticsMetrics',
    'ValidationError',
    
    # Audio processing
    'AudioProcessor',
    'AudioFormat',
    'AudioMetadata',
    
    # CSV processing
    'CSVProcessor',
    'CSVExporter',
    
    # Labeling
    'LabelingEngine',
    'CallLabeler',
    'LabelingRule',
    'RuleCondition',
    
    # Storage
    'StorageManager',
    'DataStore',
    'StorageBackend'
]

# Package version
__version__ = '1.0.0'