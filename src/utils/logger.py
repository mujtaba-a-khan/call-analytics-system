"""
Logging Configuration Module for Call Analytics System

This module provides centralized logging configuration with support for
multiple handlers, log rotation, structured logging, and performance monitoring.
"""

import json
import logging
import logging.handlers
import sys
import time
import traceback
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured JSON logs for better parsing
    and analysis in production environments.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as structured JSON.

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        log_obj = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add extra fields if present
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
            ]:
                log_obj[key] = value

        return json.dumps(log_obj)


class ColoredFormatter(logging.Formatter):
    """
    Formatter that adds color to console output for better readability
    during development.
    """

    # Color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors for console output.

        Args:
            record: Log record to format

        Returns:
            Colored log string
        """
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"

        # Format the message
        result = super().format(record)

        # Reset level name for other handlers
        record.levelname = levelname

        return result


class PerformanceFilter(logging.Filter):
    """
    Filter that adds performance metrics to log records.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add performance metrics to log record.

        Args:
            record: Log record to filter

        Returns:
            True to keep the record
        """
        # Add memory usage
        try:
            import psutil

            process = psutil.Process()
            record.memory_mb = process.memory_info().rss / 1024 / 1024
            record.cpu_percent = process.cpu_percent()
        except ImportError:
            pass

        return True


def setup_logging(
    log_level: str = "INFO",
    log_dir: Path | None = None,
    console_output: bool = True,
    file_output: bool = True,
    structured_logs: bool = False,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5,
) -> None:
    """
    Configure application-wide logging with multiple handlers and formatters.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (creates if not exists)
        console_output: Whether to output to console
        file_output: Whether to output to files
        structured_logs: Whether to use structured JSON logging
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
    """
    # Create log directory if specified
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
    else:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatters
    if structured_logs:
        file_formatter = StructuredFormatter()
    else:
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_formatter = ColoredFormatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%H:%M:%S"
    )

    # Add performance filter
    perf_filter = PerformanceFilter()

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)

    # File handlers
    if file_output:
        # Main application log
        app_log_file = log_dir / "application.log"
        app_handler = logging.handlers.RotatingFileHandler(
            app_log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        app_handler.setFormatter(file_formatter)
        app_handler.addFilter(perf_filter)
        root_logger.addHandler(app_handler)

        # Error log
        error_log_file = log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        error_handler.setFormatter(file_formatter)
        error_handler.setLevel(logging.ERROR)
        root_logger.addHandler(error_handler)

        # Performance log (for metrics and timing)
        perf_log_file = log_dir / "performance.log"
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        perf_handler.setFormatter(StructuredFormatter())
        perf_handler.setLevel(logging.DEBUG)
        perf_handler.addFilter(lambda r: hasattr(r, "performance"))
        root_logger.addHandler(perf_handler)

    # Configure specific loggers
    configure_module_loggers()

    # Log startup message
    root_logger.info(
        f"Logging initialized - Level: {log_level}, "
        f"Console: {console_output}, File: {file_output}, "
        f"Structured: {structured_logs}"
    )


def configure_module_loggers() -> None:
    """
    Configure logging levels for specific modules to reduce noise
    and focus on relevant information.
    """
    # Reduce verbosity of third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Set specific levels for application modules
    logging.getLogger("src.core").setLevel(logging.INFO)
    logging.getLogger("src.ml").setLevel(logging.INFO)
    logging.getLogger("src.analysis").setLevel(logging.DEBUG)
    logging.getLogger("src.ui").setLevel(logging.INFO)
    logging.getLogger("src.vectordb").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_execution_time(func):
    """
    Decorator to log function execution time.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            logger.debug(
                f"{func.__name__} executed in {execution_time:.3f} seconds",
                extra={"performance": True, "execution_time": execution_time},
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"{func.__name__} failed after {execution_time:.3f} seconds: {e}",
                exc_info=True,
                extra={"performance": True, "execution_time": execution_time},
            )
            raise

    return wrapper


def log_api_call(service: str, endpoint: str, method: str = "GET"):
    """
    Decorator to log API calls with details.

    Args:
        service: Name of the service being called
        endpoint: API endpoint
        method: HTTP method

    Returns:
        Decorator function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            start_time = time.time()

            logger.info(
                f"API call to {service}: {method} {endpoint}",
                extra={"api_service": service, "api_endpoint": endpoint, "api_method": method},
            )

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                logger.info(
                    f"API call successful: {service} - {execution_time:.3f}s",
                    extra={
                        "api_service": service,
                        "api_endpoint": endpoint,
                        "api_method": method,
                        "api_duration": execution_time,
                        "api_success": True,
                    },
                )

                return result

            except Exception as e:
                execution_time = time.time() - start_time

                logger.error(
                    f"API call failed: {service} - {e}",
                    exc_info=True,
                    extra={
                        "api_service": service,
                        "api_endpoint": endpoint,
                        "api_method": method,
                        "api_duration": execution_time,
                        "api_success": False,
                    },
                )
                raise

        return wrapper

    return decorator


class LogContext:
    """
    Context manager for adding contextual information to logs within a scope.
    """

    _context_var: ContextVar[dict[str, Any] | None] = ContextVar("log_context", default=None)

    def __init__(self, **context: Any):
        """
        Initialize log context.

        Args:
            **context: Key-value pairs to add to log records
        """
        self.context = context
        self.token: Token | None = None

    def __enter__(self):
        """Enter the context and add contextual information."""
        current_context = self._context_var.get()
        merged_context = dict(current_context) if current_context is not None else {}
        merged_context.update(self.context)
        self.token = self._context_var.set(merged_context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and remove contextual information."""
        if self.token is not None:
            self._context_var.reset(self.token)


def log_with_context(**context):
    """
    Decorator to add context to all logs within a function.

    Args:
        **context: Context key-value pairs

    Returns:
        Decorator function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with LogContext(**context):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Utility functions for common logging patterns


def log_dataframe_info(df, logger: logging.Logger, name: str = "DataFrame") -> None:
    """
    Log information about a pandas DataFrame.

    Args:
        df: Pandas DataFrame
        logger: Logger instance
        name: Name for the DataFrame in logs
    """
    logger.info(
        f"{name} info - Shape: {df.shape}, "
        f"Columns: {list(df.columns)}, "
        f"Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
    )


def log_config(config: dict[str, Any], logger: logging.Logger) -> None:
    """
    Log configuration dictionary in a readable format.

    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    logger.info("Configuration loaded:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"    {sub_key}: {sub_value}")
        else:
            logger.info(f"  {key}: {value}")


def create_audit_logger(name: str = "audit") -> logging.Logger:
    """
    Create a specialized logger for audit trails.

    Args:
        name: Logger name

    Returns:
        Configured audit logger
    """
    audit_logger = logging.getLogger(name)
    audit_logger.setLevel(logging.INFO)

    # Create audit log handler
    audit_handler = logging.handlers.RotatingFileHandler(
        "logs/audit.log", maxBytes=10485760, backupCount=10
    )

    # Use structured format for audit logs
    audit_handler.setFormatter(StructuredFormatter())
    audit_logger.addHandler(audit_handler)

    return audit_logger
