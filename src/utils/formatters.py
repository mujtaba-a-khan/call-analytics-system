"""
Formatting Utilities for Call Analytics System.

Provides helpers for formatting phone numbers, durations, dates,
and other commonly displayed values.
"""

import logging
import os
import re
from collections.abc import Iterable
from datetime import datetime
from typing import Any

import phonenumbers
import pytz
from dateutil import parser as date_parser

logger = logging.getLogger(__name__)


def format_phone_number(phone: str, country_code: str = 'US') -> str:
    """Format a phone number using international notation."""
    try:
        parsed = phonenumbers.parse(phone, country_code)
        return phonenumbers.format_number(
            parsed,
            phonenumbers.PhoneNumberFormat.INTERNATIONAL,
        )
    except phonenumbers.NumberParseException:
        digits = re.sub(r'\D', '', phone)
        if len(digits) == 10:
            return f"+1 ({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        if len(digits) == 11 and digits.startswith('1'):
            return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        return phone


def normalize_phone_number(phone: str, country_code: str = 'US') -> str:
    """Normalize a phone number to E.164 format."""
    try:
        parsed = phonenumbers.parse(phone, country_code)
        return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    except phonenumbers.NumberParseException:
        digits = re.sub(r'\D', '', phone)
        if len(digits) == 10:
            return f"+1{digits}"
        if digits.startswith('+'):
            return digits
        return f"+{digits}"


def format_duration(seconds: int | float | None, format_type: str = 'human') -> str:
    """Format a duration (in seconds) into a human readable string."""
    if seconds is None or seconds < 0:
        return "0s"

    seconds = int(seconds)

    if format_type == 'clock':
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    if format_type == 'short':
        if seconds < 60:
            return f"{seconds}s"
        if seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}m {secs}s" if secs else f"{minutes}m"
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"

    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    if seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        result = f"{minutes} minute{'s' if minutes != 1 else ''}"
        if secs:
            result += f" {secs} second{'s' if secs != 1 else ''}"
        return result

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    result = f"{hours} hour{'s' if hours != 1 else ''}"
    if minutes:
        result += f" {minutes} minute{'s' if minutes != 1 else ''}"
    return result


def format_bytes(count: int, precision: int = 2) -> str:
    """Convert a raw byte count into a human readable string."""
    if count < 0:
        return "0 B"

    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    index = 0
    value = float(count)

    while value >= 1024 and index < len(units) - 1:
        value /= 1024
        index += 1

    if index == 0:
        return f"{int(value)} {units[index]}"
    return f"{value:.{precision}f} {units[index]}"


def format_percentage(value: float, precision: int = 1, include_sign: bool = False) -> str:
    """Format a decimal value as a percentage string."""
    percentage = value * 100
    prefix = "+" if include_sign and percentage > 0 else ""
    return f"{prefix}{percentage:.{precision}f}%"


def format_currency(amount: float, currency: str = 'USD', precision: int = 2) -> str:
    """Format a monetary amount with the appropriate symbol."""
    currency_symbols = {
        'USD': '$',
        'EUR': '€',
        'GBP': '£',
        'JPY': '¥',
        'CNY': '¥',
    }

    symbol = currency_symbols.get(currency, f"{currency} ")
    if currency == 'JPY':
        return f"{symbol}{amount:,.0f}"
    return f"{symbol}{amount:,.{precision}f}"


def parse_datetime_flexible(
    datetime_str: str,
    date_format: str | None = None,
    timezone: str | None = None,
) -> datetime | None:
    """Parse a datetime string using flexible formats."""
    if not datetime_str:
        return None

    try:
        if date_format:
            parsed = datetime.strptime(datetime_str, date_format)
        else:
            parsed = date_parser.parse(datetime_str)

        if timezone:
            tz = pytz.timezone(timezone)
            parsed = tz.localize(parsed) if parsed.tzinfo is None else parsed.astimezone(tz)

        return parsed

    except (ValueError, TypeError) as exc:
        logger.debug("Failed to parse datetime '%s': %s", datetime_str, exc)

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
        '%Y-%m-%dT%H:%M:%S.%f',
    ]

    for fmt in common_formats:
        try:
            parsed = datetime.strptime(datetime_str, fmt)
            if timezone:
                tz = pytz.timezone(timezone)
                parsed = tz.localize(parsed)
            return parsed
        except ValueError:
            continue

    return None


def truncate_text(text: str, max_length: int = 100, suffix: str = '...') -> str:
    """Trim text to a maximum length, adding a suffix if truncated."""
    if not text or len(text) <= max_length:
        return text or ''

    truncate_at = max_length - len(suffix)
    truncated = text[:truncate_at]
    last_space = truncated.rfind(' ')
    if last_space > truncate_at * 0.8:
        truncated = truncated[:last_space]
    return truncated + suffix


def format_list(items: Iterable[Any], max_items: int = 5, separator: str = ', ') -> str:
    """Format a list of items, truncating when too long."""
    items_list = list(items)
    if not items_list:
        return ''

    if len(items_list) <= max_items:
        return separator.join(str(item) for item in items_list)

    shown_items = items_list[:max_items]
    remaining = len(items_list) - max_items
    return f"{separator.join(str(item) for item in shown_items)} (and {remaining} more)"


def format_time_ago(dt: datetime, reference: datetime | None = None) -> str:
    """Return a human readable relative time string."""
    reference = reference or (datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now())
    delta = reference - dt

    if delta.days > 365:
        years = delta.days // 365
        return f"{years} year{'s' if years != 1 else ''} ago"
    if delta.days > 30:
        months = delta.days // 30
        return f"{months} month{'s' if months != 1 else ''} ago"
    if delta.days > 0:
        return f"{delta.days} day{'s' if delta.days != 1 else ''} ago"
    if delta.seconds > 3600:
        hours = delta.seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    if delta.seconds > 60:
        minutes = delta.seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    return "just now"


def sanitize_filename(filename: str, replacement: str = '_') -> str:
    """Sanitize a filename by removing unsafe characters."""
    sanitized = re.sub(r'[<>:"/\\|?*]', replacement, filename)
    sanitized = sanitized.strip('. ')

    max_length = 255
    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        sanitized = f"{name[:max_length - len(ext) - 1]}{ext}"

    return sanitized or 'unnamed'
