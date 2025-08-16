"""
Formatting Utilities for Call Analytics System

This module provides functions for formatting and normalizing various
data types including phone numbers, dates, durations, and currencies.
"""

import re
import logging
from typing import Optional, Union, Any
from datetime import datetime, date, timedelta
import phonenumbers
from dateutil import parser as date_parser
import pytz

# Configure module logger
logger = logging.getLogger(__name__)


def format_phone_number(phone: str, country_code: str = 'US') -> str:
    """
    Format phone number to international format.
    
    Args:
        phone: Phone number string
        country_code: ISO country code for parsing
        
    Returns:
        Formatted phone number string
    """
    try:
        parsed = phonenumbers.parse(phone, country_code)
        return phonenumbers.format_number(
            parsed,
            phonenumbers.PhoneNumberFormat.INTERNATIONAL
        )
    except phonenumbers.NumberParseException:
        # Fallback to basic formatting
        digits = re.sub(r'\D', '', phone)
        if len(digits) == 10:  # US number without country code
            return f"+1 ({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        elif len(digits) == 11 and digits[0] == '1':  # US number with country code
            return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        return phone


def normalize_phone_number(phone: str, country_code: str = 'US') -> str:
    """
    Normalize phone number to E.164 format for storage.
    
    Args:
        phone: Phone number string
        country_code: ISO country code for parsing
        
    Returns:
        Normalized phone number in E.164 format
    """
    try:
        parsed = phonenumbers.parse(phone, country_code)
        return phonenumbers.format_number(
            parsed,
            phonenumbers.PhoneNumberFormat.E164
        )
    except phonenumbers.NumberParseException:
        # Fallback to basic normalization
        digits = re.sub(r'\D', '', phone)
        if len(digits) == 10:  # Add US country code
            return f"+1{digits}"
        elif len(digits) == 11 and digits[0] == '1':
            return f"+{digits}"
        return f"+{digits}"  # Assume already has country code


def format_duration(seconds: Union[int, float], format_type: str = 'human') -> str:
    """
    Format duration in seconds to human-readable format.
    
    Args:
        seconds: Duration in seconds
        format_type: Format type ('human', 'short', 'clock')
        
    Returns:
        Formatted duration string
    """
    if seconds is None or seconds < 0:
        return "0s"
    
    seconds = int(seconds)
    
    if format_type == 'clock':
        # Format as HH:MM:SS
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    elif format_type == 'short':
        # Format as abbreviated
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}m {secs}s" if secs > 0 else f"{minutes}m"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m" if minutes > 0 else f"{hours}h"
    
    else:  # human format
        # Format as full words
        if seconds < 60:
            return f"{seconds} second{'s' if seconds != 1 else ''}"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            result = f"{minutes} minute{'s' if minutes != 1 else ''}"
            if secs > 0:
                result += f" {secs} second{'s' if secs != 1 else ''}"
            return result
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            result = f"{hours} hour{'s' if hours != 1 else ''}"
            if minutes > 0:
                result += f" {minutes} minute{'s' if minutes != 1 else ''}"
            return result


def format_bytes(bytes_value: int, precision: int = 2) -> str:
    """
    Format bytes to human-readable format.
    
    Args:
        bytes_value: Number of bytes
        precision: Decimal precision
        
    Returns:
        Formatted bytes string
    """
    if bytes_value < 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    unit_index = 0
    
    while bytes_value >= 1024 and unit_index < len(units) - 1:
        bytes_value /= 1024.0
        unit_index += 1
    
    if unit_index == 0:  # Bytes
        return f"{int(bytes_value)} {units[unit_index]}"
    else:
        return f"{bytes_value:.{precision}f} {units[unit_index]}"


def format_percentage(value: float, precision: int = 1, include_sign: bool = False) -> str:
    """
    Format decimal value as percentage.
    
    Args:
        value: Decimal value (0.5 = 50%)
        precision: Decimal precision
        include_sign: Whether to include + sign for positive values
        
    Returns:
        Formatted percentage string
    """
    percentage = value * 100
    
    if include_sign and percentage > 0:
        return f"+{percentage:.{precision}f}%"
    else:
        return f"{percentage:.{precision}f}%"


def format_currency(amount: float, currency: str = 'USD', precision: int = 2) -> str:
    """
    Format amount as currency string.
    
    Args:
        amount: Monetary amount
        currency: Currency code
        precision: Decimal precision
        
    Returns:
        Formatted currency string
    """
    currency_symbols = {
        'USD': '$',
        'EUR': '€',
        'GBP': '£',
        'JPY': '¥',
        'CNY': '¥'
    }
    
    symbol = currency_symbols.get(currency, currency + ' ')
    
    # Format with thousands separator
    if currency == 'JPY':  # No decimal places for Yen
        return f"{symbol}{amount:,.0f}"
    else:
        return f"{symbol}{amount:,.{precision}f}"


def parse_datetime_flexible(
    datetime_str: str,
    date_format: Optional[str] = None,
    timezone: Optional[str] = None
) -> Optional[datetime]:
    """
    Parse datetime string flexibly, handling various formats.
    
    Args:
        datetime_str: Datetime string to parse
        date_format: Specific format string (uses dateutil if None)
        timezone: Timezone name (e.g., 'US/Eastern')
        
    Returns:
        Parsed datetime object or None if parsing fails
    """
    if not datetime_str:
        return None
    
    try:
        # Try specific format first if provided
        if date_format:
            dt = datetime.strptime(datetime_str, date_format)
        else:
            # Use dateutil for flexible parsing
            dt = date_parser.parse(datetime_str)
        
        # Add timezone if specified
        if timezone:
            tz = pytz.timezone(timezone)
            if dt.tzinfo is None:
                dt = tz.localize(dt)
            else:
                dt = dt.astimezone(tz)
        
        return dt
        
    except (ValueError, TypeError) as e:
        logger.debug(f"Failed to parse datetime '{datetime_str}': {e}")
        
        # Try common formats as fallback
        common_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d %H:%M:%S',
            '%m/%d/%Y %H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y%m%d',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S.%f'
        ]
        
        for fmt in common_formats:
            try:
                return datetime.strptime(datetime_str, fmt)
            except ValueError:
                continue
        
        return None


def truncate_text(text: str, max_length: int = 100, suffix: str = '...') -> str:
    """
    Truncate text to maximum length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated text
    """
    if not text:
        return ''
    
    if len(text) <= max_length:
        return text
    
    # Truncate and add suffix
    truncate_at = max_length - len(suffix)
    
    # Try to truncate at word boundary
    truncated = text[:truncate_at]
    last_space = truncated.rfind(' ')
    
    if last_space > truncate_at * 0.8:  # If we found a space reasonably close
        truncated = truncated[:last_space]
    
    return truncated + suffix


def format_list(items: list, max_items: int = 5, separator: str = ', ') -> str:
    """
    Format list of items with truncation.
    
    Args:
        items: List of items to format
        max_items: Maximum items to show
        separator: Item separator
        
    Returns:
        Formatted list string
    """
    if not items:
        return ''
    
    if len(items) <= max_items:
        return separator.join(str(item) for item in items)
    
    shown_items = items[:max_items]
    remaining = len(items) - max_items
    
    result = separator.join(str(item) for item in shown_items)
    result += f" (and {remaining} more)"
    
    return result


def format_time_ago(dt: datetime, reference: Optional[datetime] = None) -> str:
    """
    Format datetime as relative time (e.g., "2 hours ago").
    
    Args:
        dt: Datetime to format
        reference: Reference datetime (now if None)
        
    Returns:
        Relative time string
    """
    if reference is None:
        reference = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
    
    delta = reference - dt
    
    if delta.days > 365:
        years = delta.days // 365
        return f"{years} year{'s' if years != 1 else ''} ago"
    elif delta.days > 30:
        months = delta.days // 30
        return f"{months} month{'s' if months != 1 else ''} ago"
    elif delta.days > 0:
        return f"{delta.days} day{'s' if delta.days != 1 else ''} ago"
    elif delta.seconds > 3600:
        hours = delta.seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif delta.seconds > 60:
        minutes = delta.seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    else:
        return "just now"


def sanitize_filename(filename: str, replacement: str = '_') -> str:
    """
    Sanitize filename by removing invalid characters.
    
    Args:
        filename: Original filename
        replacement: Character to replace invalid chars with
        
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, replacement, filename)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    # Limit length
    max_length = 255
    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        name = name[:max_length - len(ext) - 1]
        sanitized = name + ext
    
    return sanitized or 'unnamed'