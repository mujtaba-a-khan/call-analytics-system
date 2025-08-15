"""
File Handling Utilities

Functions for file I/O operations, format detection,
and safe file handling.
"""

import os
import json
import yaml
import toml
import pickle
import shutil
import hashlib
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_file_write(filepath: Union[str, Path], 
                   content: Union[str, bytes],
                   mode: str = 'w') -> bool:
    """
    Safely write to a file using atomic operations.
    
    Args:
        filepath: Path to file
        content: Content to write
        mode: Write mode ('w' for text, 'wb' for binary)
    
    Returns:
        True if successful
    """
    filepath = Path(filepath)
    temp_file = filepath.with_suffix('.tmp')
    
    try:
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temporary file
        with open(temp_file, mode) as f:
            f.write(content)
        
        # Atomic rename
        temp_file.replace(filepath)
        return True
        
    except Exception as e:
        logger.error(f"Error writing file {filepath}: {e}")
        # Clean up temp file if it exists
        if temp_file.exists():
            temp_file.unlink()
        return False


def safe_file_read(filepath: Union[str, Path],
                  mode: str = 'r',
                  default: Any = None) -> Any:
    """
    Safely read from a file with error handling.
    
    Args:
        filepath: Path to file
        mode: Read mode ('r' for text, 'rb' for binary)
        default: Default value if read fails
    
    Returns:
        File content or default value
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return default
    
    try:
        with open(filepath, mode) as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        return default


def load_config_file(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from various file formats.
    
    Args:
        filepath: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        logger.warning(f"Config file not found: {filepath}")
        return {}
    
    suffix = filepath.suffix.lower()
    
    try:
        if suffix == '.json':
            with open(filepath, 'r') as f:
                return json.load(f)
        
        elif suffix in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
                return yaml.safe_load(f) or {}
        
        elif suffix == '.toml':
            with open(filepath, 'r') as f:
                return toml.load(f)
        
        else:
            logger.error(f"Unsupported config format: {suffix}")
            return {}
            
    except Exception as e:
        logger.error(f"Error loading config {filepath}: {e}")
        return {}


def save_config_file(config: Dict[str, Any],
                    filepath: Union[str, Path]) -> bool:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save file
    
    Returns:
        True if successful
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    
    try:
        if suffix == '.json':
            content = json.dumps(config, indent=2)
        
        elif suffix in ['.yaml', '.yml']:
            content = yaml.dump(config, default_flow_style=False)
        
        elif suffix == '.toml':
            content = toml.dumps(config)
        
        else:
            logger.error(f"Unsupported config format: {suffix}")
            return False
        
        return safe_file_write(filepath, content)
        
    except Exception as e:
        logger.error(f"Error saving config {filepath}: {e}")
        return False


def get_file_hash(filepath: Union[str, Path],
                 algorithm: str = 'sha256') -> Optional[str]:
    """
    Calculate hash of a file.
    
    Args:
        filepath: Path to file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
    
    Returns:
        Hex digest or None if error
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return None
    
    try:
        if algorithm == 'md5':
            hasher = hashlib.md5()
        elif algorithm == 'sha1':
            hasher = hashlib.sha1()
        elif algorithm == 'sha256':
            hasher = hashlib.sha256()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        
        return hasher.hexdigest()
        
    except Exception as e:
        logger.error(f"Error hashing file {filepath}: {e}")
        return None


def get_file_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Get detailed information about a file.
    
    Args:
        filepath: Path to file
    
    Returns:
        Dictionary with file information
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return {'exists': False}
    
    stat = filepath.stat()
    
    info = {
        'exists': True,
        'path': str(filepath.absolute()),
        'name': filepath.name,
        'suffix': filepath.suffix,
        'size_bytes': stat.st_size,
        'size_mb': stat.st_size / (1024 * 1024),
        'created': stat.st_ctime,
        'modified': stat.st_mtime,
        'is_file': filepath.is_file(),
        'is_dir': filepath.is_dir(),
        'mime_type': mimetypes.guess_type(str(filepath))[0]
    }
    
    return info


def copy_file(source: Union[str, Path],
             destination: Union[str, Path],
             overwrite: bool = False) -> bool:
    """
    Copy a file with safety checks.
    
    Args:
        source: Source file path
        destination: Destination file path
        overwrite: Whether to overwrite existing file
    
    Returns:
        True if successful
    """
    source = Path(source)
    destination = Path(destination)
    
    if not source.exists():
        logger.error(f"Source file not found: {source}")
        return False
    
    if destination.exists() and not overwrite:
        logger.warning(f"Destination exists and overwrite=False: {destination}")
        return False
    
    try:
        # Ensure destination directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        shutil.copy2(source, destination)
        return True
        
    except Exception as e:
        logger.error(f"Error copying file: {e}")
        return False


def move_file(source: Union[str, Path],
             destination: Union[str, Path],
             overwrite: bool = False) -> bool:
    """
    Move a file with safety checks.
    
    Args:
        source: Source file path
        destination: Destination file path
        overwrite: Whether to overwrite existing file
    
    Returns:
        True if successful
    """
    source = Path(source)
    destination = Path(destination)
    
    if not source.exists():
        logger.error(f"Source file not found: {source}")
        return False
    
    if destination.exists() and not overwrite:
        logger.warning(f"Destination exists and overwrite=False: {destination}")
        return False
    
    try:
        # Ensure destination directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Move file
        shutil.move(str(source), str(destination))
        return True
        
    except Exception as e:
        logger.error(f"Error moving file: {e}")
        return False


def delete_file(filepath: Union[str, Path],
               secure: bool = False) -> bool:
    """
    Delete a file.
    
    Args:
        filepath: Path to file
        secure: Whether to securely overwrite before deletion
    
    Returns:
        True if successful
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return True  # Already deleted
    
    try:
        if secure and filepath.is_file():
            # Overwrite with random data before deletion
            size = filepath.stat().st_size
            with open(filepath, 'wb') as f:
                f.write(os.urandom(size))
        
        if filepath.is_file():
            filepath.unlink()
        elif filepath.is_dir():
            shutil.rmtree(filepath)
        
        return True
        
    except Exception as e:
        logger.error(f"Error deleting {filepath}: {e}")
        return False


def list_files(directory: Union[str, Path],
              pattern: str = '*',
              recursive: bool = False) -> List[Path]:
    """
    List files in a directory.
    
    Args:
        directory: Directory path
        pattern: File pattern (glob)
        recursive: Whether to search recursively
    
    Returns:
        List of file paths
    """
    directory = Path(directory)
    
    if not directory.exists() or not directory.is_dir():
        return []
    
    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))


def save_pickle(obj: Any,
               filepath: Union[str, Path]) -> bool:
    """
    Save object as pickle file.
    
    Args:
        obj: Object to pickle
        filepath: Path to save file
    
    Returns:
        True if successful
    """
    filepath = Path(filepath)
    
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        return True
    except Exception as e:
        logger.error(f"Error pickling object: {e}")
        return False


def load_pickle(filepath: Union[str, Path],
               default: Any = None) -> Any:
    """
    Load object from pickle file.
    
    Args:
        filepath: Path to pickle file
        default: Default value if load fails
    
    Returns:
        Unpickled object or default
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return default
    
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error unpickling file: {e}")
        return default


def get_directory_size(directory: Union[str, Path]) -> int:
    """
    Calculate total size of a directory.
    
    Args:
        directory: Directory path
    
    Returns:
        Size in bytes
    """
    directory = Path(directory)
    
    if not directory.exists() or not directory.is_dir():
        return 0
    
    total_size = 0
    for file in directory.rglob('*'):
        if file.is_file():
            total_size += file.stat().st_size
    
    return total_size


def clean_directory(directory: Union[str, Path],
                   older_than_days: Optional[int] = None,
                   pattern: str = '*') -> int:
    """
    Clean files from a directory.
    
    Args:
        directory: Directory path
        older_than_days: Only delete files older than this
        pattern: File pattern to match
    
    Returns:
        Number of files deleted
    """
    directory = Path(directory)
    
    if not directory.exists() or not directory.is_dir():
        return 0
    
    import time
    current_time = time.time()
    deleted_count = 0
    
    for file in directory.glob(pattern):
        if not file.is_file():
            continue
        
        # Check age if specified
        if older_than_days:
            file_age_days = (current_time - file.stat().st_mtime) / 86400
            if file_age_days < older_than_days:
                continue
        
        try:
            file.unlink()
            deleted_count += 1
        except Exception as e:
            logger.error(f"Error deleting {file}: {e}")
    
    return deleted_count