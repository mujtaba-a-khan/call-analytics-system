"""
Utilities Package for Call Analytics System

This package exposes logging, validation, and formatting helpers used
throughout the application. Heavy modules are imported lazily to keep
startup time low and avoid optional dependency issues.
"""

from __future__ import annotations

from typing import Any

__version__ = "1.0.0"

_LOGGER_EXPORTS = {
    "setup_logging": "setup_logging",
    "get_logger": "get_logger",
    "log_execution_time": "log_execution_time",
    "log_api_call": "log_api_call",
    "log_with_context": "log_with_context",
    "log_dataframe_info": "log_dataframe_info",
    "log_config": "log_config",
    "create_audit_logger": "create_audit_logger",
    "LogContext": "LogContext",
    "StructuredFormatter": "StructuredFormatter",
    "ColoredFormatter": "ColoredFormatter",
}

_VALIDATOR_EXPORTS = {
    "validate_phone": "validate_phone",
    "validate_phone_number": "validate_phone",  # backwards compatibility alias
    "validate_email": "validate_email",
    "validate_date": "validate_date",
    "validate_number": "validate_number",
    "validate_text": "validate_text",
    "validate_url": "validate_url",
    "validate_dataframe": "validate_dataframe",
    "validate_json": "validate_json",
    "validate_config": "validate_config",
    "validate_time_range": "validate_time_range",
}

_FORMATTER_EXPORTS = {
    "format_phone_number": "format_phone_number",
    "normalize_phone_number": "normalize_phone_number",
    "format_duration": "format_duration",
    "format_bytes": "format_bytes",
    "format_percentage": "format_percentage",
    "format_currency": "format_currency",
    "parse_datetime_flexible": "parse_datetime_flexible",
    "truncate_text": "truncate_text",
}

__all__ = [
    *(_LOGGER_EXPORTS.keys()),
    *(_VALIDATOR_EXPORTS.keys()),
    *(_FORMATTER_EXPORTS.keys()),
]


def __getattr__(name: str) -> Any:
    """
    Lazily resolve utilities to avoid importing optional dependencies
    until they are actually needed.
    """
    if name in _LOGGER_EXPORTS:
        from . import logger

        return getattr(logger, _LOGGER_EXPORTS[name])

    if name in _VALIDATOR_EXPORTS:
        from . import validators

        return getattr(validators, _VALIDATOR_EXPORTS[name])

    if name in _FORMATTER_EXPORTS:
        from . import formatters

        return getattr(formatters, _FORMATTER_EXPORTS[name])

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Ensure dynamic attributes appear in dir() output."""
    return sorted(set(globals()) | set(__all__))
