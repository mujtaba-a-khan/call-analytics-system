"""
Data Validation Utilities

Functions for validating and sanitizing data inputs.
"""

import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, date
import pandas as pd


def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
    
    Returns:
        True if valid email format
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email)))


def validate_phone(phone: str, region: str = 'US') -> bool:
    """
    Validate phone number format.
    
    Args:
        phone: Phone number to validate
        region: Region code for validation
    
    Returns:
        True if valid phone format
    """
    # Remove common separators
    phone = re.sub(r'[\s\-\.\(\)]', '', str(phone))
    
    if region == 'US':
        # US phone number: 10 digits, optionally starting with 1
        pattern = r'^1?\d{10}$'
        return bool(re.match(pattern, phone))
    else:
        # Generic international: 7-15 digits
        pattern = r'^\+?\d{7,15}$'
        return bool(re.match(pattern, phone))


def validate_date(date_value: Any,
                 format: str = '%Y-%m-%d',
                 min_date: Optional[date] = None,
                 max_date: Optional[date] = None) -> bool:
    """
    Validate date value and format.
    
    Args:
        date_value: Date to validate
        format: Expected date format
        min_date: Minimum allowed date
        max_date: Maximum allowed date
    
    Returns:
        True if valid date
    """
    try:
        if isinstance(date_value, str):
            parsed_date = datetime.strptime(date_value, format).date()
        elif isinstance(date_value, datetime):
            parsed_date = date_value.date()
        elif isinstance(date_value, date):
            parsed_date = date_value
        else:
            return False
        
        if min_date and parsed_date < min_date:
            return False
        
        if max_date and parsed_date > max_date:
            return False
        
        return True
        
    except (ValueError, TypeError):
        return False


def validate_number(value: Any,
                   min_value: Optional[float] = None,
                   max_value: Optional[float] = None,
                   allow_negative: bool = True) -> bool:
    """
    Validate numeric value.
    
    Args:
        value: Value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        allow_negative: Whether negative values are allowed
    
    Returns:
        True if valid number
    """
    try:
        num = float(value)
        
        if not allow_negative and num < 0:
            return False
        
        if min_value is not None and num < min_value:
            return False
        
        if max_value is not None and num > max_value:
            return False
        
        return True
        
    except (ValueError, TypeError):
        return False


def validate_text(text: Any,
                 min_length: int = 0,
                 max_length: Optional[int] = None,
                 pattern: Optional[str] = None) -> bool:
    """
    Validate text input.
    
    Args:
        text: Text to validate
        min_length: Minimum text length
        max_length: Maximum text length
        pattern: Regex pattern to match
    
    Returns:
        True if valid text
    """
    if not isinstance(text, str):
        return False
    
    if len(text) < min_length:
        return False
    
    if max_length and len(text) > max_length:
        return False
    
    if pattern and not re.match(pattern, text):
        return False
    
    return True


def validate_url(url: str) -> bool:
    """
    Validate URL format.
    
    Args:
        url: URL to validate
    
    Returns:
        True if valid URL
    """
    pattern = r'^https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)$'
    return bool(re.match(pattern, str(url)))


def validate_dataframe(df: pd.DataFrame,
                      required_columns: List[str],
                      column_types: Optional[Dict[str, type]] = None) -> Dict[str, Any]:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        column_types: Optional mapping of column names to expected types
    
    Returns:
        Validation result dictionary
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Check if DataFrame is empty
    if df.empty:
        result['valid'] = False
        result['errors'].append("DataFrame is empty")
        return result
    
    # Check required columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        result['valid'] = False
        result['errors'].append(f"Missing required columns: {missing_columns}")
    
    # Check column types if specified
    if column_types:
        for col, expected_type in column_types.items():
            if col in df.columns:
                actual_type = df[col].dtype
                
                # Map pandas types to Python types
                type_mapping = {
                    str: ['object', 'string'],
                    int: ['int64', 'int32', 'int16', 'int8'],
                    float: ['float64', 'float32', 'float16'],
                    bool: ['bool'],
                    datetime: ['datetime64[ns]']
                }
                
                valid_dtypes = type_mapping.get(expected_type, [])
                if str(actual_type) not in valid_dtypes:
                    result['warnings'].append(
                        f"Column '{col}' has type {actual_type}, expected {expected_type}"
                    )
    
    # Calculate statistics
    result['stats'] = {
        'row_count': len(df),
        'column_count': len(df.columns),
        'null_counts': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    # Check for high null percentages
    for col in df.columns:
        null_pct = (df[col].isnull().sum() / len(df)) * 100
        if null_pct > 50:
            result['warnings'].append(f"Column '{col}' has {null_pct:.1f}% null values")
    
    return result


def sanitize_filename(filename: str,
                     max_length: int = 255,
                     replacement: str = '_') -> str:
    """
    Sanitize filename for safe file system use.
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
        replacement: Character to replace invalid characters
    
    Returns:
        Sanitized filename
    """
    # Remove path separators
    filename = filename.replace('/', replacement).replace('\\', replacement)
    
    # Remove invalid characters
    invalid_chars = '<>:"|?*'
    for char in invalid_chars:
        filename = filename.replace(char, replacement)
    
    # Remove control characters
    filename = ''.join(char if ord(char) >= 32 else replacement for char in filename)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Truncate if too long
    if len(filename) > max_length:
        # Keep file extension if present
        if '.' in filename:
            name, ext = filename.rsplit('.', 1)
            max_name_length = max_length - len(ext) - 1
            filename = name[:max_name_length] + '.' + ext
        else:
            filename = filename[:max_length]
    
    # Ensure filename is not empty
    if not filename:
        filename = 'unnamed'
    
    return filename


def validate_json(json_str: str) -> Tuple[bool, Optional[str]]:
    """
    Validate JSON string.
    
    Args:
        json_str: JSON string to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    import json
    
    try:
        json.loads(json_str)
        return True, None
    except json.JSONDecodeError as e:
        return False, str(e)


def validate_config(config: Dict[str, Any],
                   schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration against a schema.
    
    Args:
        config: Configuration dictionary
        schema: Schema defining expected structure
    
    Returns:
        Validation result
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    def validate_value(value, expected_type, path=""):
        """Recursively validate configuration values"""
        if expected_type == 'string':
            if not isinstance(value, str):
                result['errors'].append(f"{path} must be a string")
                result['valid'] = False
        
        elif expected_type == 'int':
            if not isinstance(value, int):
                result['errors'].append(f"{path} must be an integer")
                result['valid'] = False
        
        elif expected_type == 'float':
            if not isinstance(value, (int, float)):
                result['errors'].append(f"{path} must be a number")
                result['valid'] = False
        
        elif expected_type == 'bool':
            if not isinstance(value, bool):
                result['errors'].append(f"{path} must be a boolean")
                result['valid'] = False
        
        elif expected_type == 'list':
            if not isinstance(value, list):
                result['errors'].append(f"{path} must be a list")
                result['valid'] = False
        
        elif isinstance(expected_type, dict):
            if not isinstance(value, dict):
                result['errors'].append(f"{path} must be a dictionary")
                result['valid'] = False
            else:
                for key, subschema in expected_type.items():
                    if key in value:
                        validate_value(value[key], subschema, f"{path}.{key}" if path else key)
                    elif subschema.get('required', False):
                        result['errors'].append(f"{path}.{key} is required")
                        result['valid'] = False
    
    validate_value(config, schema)
    
    return result


def is_valid_id(id_value: Any,
               pattern: str = r'^[a-zA-Z0-9_-]+$') -> bool:
    """
    Validate an ID value.
    
    Args:
        id_value: ID to validate
        pattern: Regex pattern for valid IDs
    
    Returns:
        True if valid ID
    """
    if not isinstance(id_value, str):
        return False
    
    if not id_value:
        return False
    
    return bool(re.match(pattern, id_value))


def validate_time_range(start: datetime,
                       end: datetime,
                       max_duration_hours: Optional[int] = None) -> bool:
    """
    Validate a time range.
    
    Args:
        start: Start time
        end: End time
        max_duration_hours: Maximum allowed duration
    
    Returns:
        True if valid time range
    """
    if start >= end:
        return False
    
    if max_duration_hours:
        duration = (end - start).total_seconds() / 3600
        if duration > max_duration_hours:
            return False
    
    return True