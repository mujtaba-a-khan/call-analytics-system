"""
Core Package for Call Analytics System

This package contains the core business logic and data processing modules
including audio processing, data schema, storage management, and labeling.
"""

# Package version
__version__ = '1.0.0'

# Define available exports - these will be loaded on demand
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


def __getattr__(name):
    """
    Lazy loading implementation for core modules.
    Only imports modules when they are actually accessed.
    """
    # Data schema imports
    if name in ['CallRecord', 'AudioFile', 'TranscriptionResult', 'AnalyticsMetrics', 'ValidationError']:
        from . import data_schema
        return getattr(data_schema, name)
    
    # Audio processor imports
    elif name in ['AudioProcessor', 'AudioFormat', 'AudioMetadata']:
        from . import audio_processor
        return getattr(audio_processor, name)
    
    # CSV processor imports
    elif name in ['CSVProcessor', 'CSVExporter']:
        from . import csv_processor
        return getattr(csv_processor, name)
    
    # Labeling engine imports
    elif name in ['LabelingEngine', 'CallLabeler', 'LabelingRule', 'RuleCondition']:
        from . import labeling_engine
        return getattr(labeling_engine, name)
    
    # Storage manager imports
    elif name in ['StorageManager', 'DataStore', 'StorageBackend']:
        from . import storage_manager
        return getattr(storage_manager, name)
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")