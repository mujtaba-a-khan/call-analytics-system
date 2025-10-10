"""
Utilities Package for Call Analytics System

This package provides utility functions for validation, formatting,
logging, and common operations used throughout the application.
"""

# Package version
__version__ = '1.0.0'

# Define available exports
__all__ = [
    # Logging utilities
    'setup_logging',
    'get_logger',
    'log_execution_time',
    'log_api_call',
    'LogContext',

    # Validation utilities
    'validate_phone_number',
    'validate_email',
    'validate_duration',
    'validate_date_range',
    'validate_csv_structure',
    'validate_audio_file',
    'ValidationError',

    # Formatting utilities
    'format_phone_number',
    'normalize_phone_number',
    'format_duration',
    'format_bytes',
    'format_percentage',
    'format_currency',
    'parse_datetime_flexible',
    'truncate_text'
]


def __getattr__(name):
    """
    Lazy loading implementation for utility modules.
    Imports are deferred until actually needed.
    """
    # Logger utilities
    if name in ['setup_logging', 'get_logger', 'log_execution_time', 'log_api_call',
                'LogContext', 'log_with_context', 'log_dataframe_info', 'log_config',
                'create_audit_logger', 'StructuredFormatter', 'ColoredFormatter']:
        from . import logger
        return getattr(logger, name)

    # Validator utilities
    elif name in ['validate_phone_number', 'validate_email', 'validate_duration',
                  'validate_date_range', 'validate_csv_structure', 'validate_audio_file',
                  'ValidationError']:
        from . import validators
        return getattr(validators, name)

    # Formatter utilities
    elif name in ['format_phone_number', 'normalize_phone_number', 'format_duration',
                  'format_bytes', 'format_percentage', 'format_currency',
                  'parse_datetime_flexible', 'truncate_text']:
        from . import formatters
        return getattr(formatters, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
