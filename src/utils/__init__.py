"""
Utilities Package for Call Analytics System

This package provides utility functions for validation, formatting,
logging, and common operations used throughout the application.
"""

# Import logging utilities
from .logger import (
    setup_logging,
    get_logger,
    log_execution_time,
    log_api_call,
    LogContext,
    log_with_context,
    log_dataframe_info,
    log_config,
    create_audit_logger,
    StructuredFormatter,
    ColoredFormatter
)

# Import validation utilities
from .validators import (
    validate_phone_number,
    validate_email,
    validate_duration,
    validate_date_range,
    validate_csv_structure,
    validate_audio_file,
    ValidationError
)

# Import formatting utilities
from .formatters import (
    format_phone_number,
    normalize_phone_number,
    format_duration,
    format_bytes,
    format_percentage,
    format_currency,
    parse_datetime_flexible,
    truncate_text
)

# Define package exports
__all__ = [
    # Logging
    'setup_logging',
    'get_logger',
    'log_execution_time',
    'log_api_call',
    'LogContext',
    'log_with_context',
    'log_dataframe_info',
    'log_config',
    'create_audit_logger',
    'StructuredFormatter',
    'ColoredFormatter',
    
    # Validation
    'validate_phone_number',
    'validate_email',
    'validate_duration',
    'validate_date_range',
    'validate_csv_structure',
    'validate_audio_file',
    'ValidationError',
    
    # Formatting
    'format_phone_number',
    'normalize_phone_number',
    'format_duration',
    'format_bytes',
    'format_percentage',
    'format_currency',
    'parse_datetime_flexible',
    'truncate_text'
]

# Package version
__version__ = '1.0.0'