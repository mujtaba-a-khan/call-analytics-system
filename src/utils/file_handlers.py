"""
File Handling Utilities

Functions for file I/O operations, format detection,
and safe file handling.
"""

import hashlib
import json
import logging
import mimetypes
import os
import pickle
import shutil
from pathlib import Path
from typing import Any, cast

import toml
import yaml

logger = logging.getLogger(__name__)


def ensure_directory(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_file_write(
    filepath: str | Path,
    content: str | bytes,
    mode: str = "w",
) -> bool:
    """Safely write to a file using atomic operations."""
    filepath = Path(filepath)
    temp_file = filepath.with_suffix(".tmp")

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(temp_file, mode) as handle:
            handle.write(content)

        temp_file.replace(filepath)
        return True

    except Exception as exc:
        logger.error("Error writing file %s: %s", filepath, exc)
        if temp_file.exists():
            temp_file.unlink()
        return False


def safe_file_read(
    filepath: str | Path,
    mode: str = "r",
    default: Any = None,
) -> Any:
    """Safely read from a file with error handling."""
    filepath = Path(filepath)

    if not filepath.exists():
        return default

    try:
        with open(filepath, mode) as handle:
            return handle.read()
    except Exception as exc:
        logger.error("Error reading file %s: %s", filepath, exc)
        return default


def load_config_file(filepath: str | Path) -> dict[str, Any]:
    """Load configuration from JSON, YAML, or TOML files."""
    filepath = Path(filepath)

    if not filepath.exists():
        logger.warning("Config file not found: %s", filepath)
        return {}

    suffix = filepath.suffix.lower()

    try:
        if suffix == ".json":
            with open(filepath, encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                return cast(dict[str, Any], data)
            logger.error("JSON config must be an object at %s", filepath)
            return {}

        if suffix in {".yaml", ".yml"}:
            with open(filepath, encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
            if isinstance(data, dict):
                return cast(dict[str, Any], data)
            logger.error("YAML config must be a mapping at %s", filepath)
            return {}

        if suffix == ".toml":
            with open(filepath, encoding="utf-8") as handle:
                data = toml.load(handle)
            if isinstance(data, dict):
                return cast(dict[str, Any], data)
            logger.error("TOML config must be a table at %s", filepath)
            return {}

        logger.error("Unsupported config format: %s", suffix)
        return {}

    except Exception as exc:
        logger.error("Error loading config %s: %s", filepath, exc)
        return {}


def save_config_file(config: dict[str, Any], filepath: str | Path) -> bool:
    """Save configuration to JSON, YAML, or TOML files."""
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    try:
        if suffix == ".json":
            content = json.dumps(config, indent=2)
        elif suffix in {".yaml", ".yml"}:
            content = yaml.dump(config, default_flow_style=False)
        elif suffix == ".toml":
            content = toml.dumps(config)
        else:
            logger.error("Unsupported config format: %s", suffix)
            return False

        return safe_file_write(filepath, content)

    except Exception as exc:
        logger.error("Error saving config %s: %s", filepath, exc)
        return False


def get_file_hash(filepath: str | Path, algorithm: str = "sha256") -> str | None:
    """Calculate the hash of a file."""
    filepath = Path(filepath)

    if not filepath.exists():
        return None

    try:
        if algorithm == "md5":
            hasher = hashlib.md5()
        elif algorithm == "sha1":
            hasher = hashlib.sha1()
        elif algorithm == "sha256":
            hasher = hashlib.sha256()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        with open(filepath, "rb") as handle:
            for chunk in iter(lambda: handle.read(4096), b""):
                hasher.update(chunk)

        return hasher.hexdigest()

    except Exception as exc:
        logger.error("Error hashing file %s: %s", filepath, exc)
        return None


def get_file_info(filepath: str | Path) -> dict[str, Any]:
    """Return detailed information about a file."""
    filepath = Path(filepath)

    if not filepath.exists():
        return {"exists": False}

    stat = filepath.stat()

    return {
        "exists": True,
        "path": str(filepath.resolve()),
        "name": filepath.name,
        "suffix": filepath.suffix,
        "size_bytes": stat.st_size,
        "size_mb": stat.st_size / (1024 * 1024),
        "created": stat.st_ctime,
        "modified": stat.st_mtime,
        "is_file": filepath.is_file(),
        "is_dir": filepath.is_dir(),
        "mime_type": mimetypes.guess_type(str(filepath))[0],
    }


def copy_file(
    source: str | Path,
    destination: str | Path,
    overwrite: bool = False,
) -> bool:
    """Copy a file with safety checks."""
    source_path = Path(source)
    destination_path = Path(destination)

    if not source_path.exists():
        logger.error("Source file not found: %s", source_path)
        return False

    if destination_path.exists() and not overwrite:
        logger.warning("Destination exists and overwrite=False: %s", destination_path)
        return False

    try:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        return True
    except Exception as exc:
        logger.error("Error copying file from %s to %s: %s", source_path, destination_path, exc)
        return False


def move_file(
    source: str | Path,
    destination: str | Path,
    overwrite: bool = False,
) -> bool:
    """Move a file with safety checks."""
    source_path = Path(source)
    destination_path = Path(destination)

    if not source_path.exists():
        logger.error("Source file not found: %s", source_path)
        return False

    if destination_path.exists() and not overwrite:
        logger.warning("Destination exists and overwrite=False: %s", destination_path)
        return False

    try:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(destination_path))
        return True
    except Exception as exc:
        logger.error("Error moving file from %s to %s: %s", source_path, destination_path, exc)
        return False


def delete_file(filepath: str | Path, secure: bool = False) -> bool:
    """Delete a file with optional secure overwrite."""
    filepath = Path(filepath)

    if not filepath.exists():
        return True

    try:
        if secure and filepath.is_file():
            size = filepath.stat().st_size
            with open(filepath, "wb") as handle:
                handle.write(os.urandom(size))

        if filepath.is_file():
            filepath.unlink()
        elif filepath.is_dir():
            shutil.rmtree(filepath)

        return True

    except Exception as exc:
        logger.error("Error deleting %s: %s", filepath, exc)
        return False


def list_files(directory: str | Path, pattern: str = "*", recursive: bool = False) -> list[Path]:
    """List files in a directory."""
    directory_path = Path(directory)

    if not directory_path.exists() or not directory_path.is_dir():
        return []

    return list(directory_path.rglob(pattern) if recursive else directory_path.glob(pattern))


def save_pickle(obj: Any, filepath: str | Path) -> bool:
    """Serialize an object to a pickle file."""
    filepath = Path(filepath)

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as handle:
            pickle.dump(obj, handle)
        return True
    except Exception as exc:
        logger.error("Error saving pickle %s: %s", filepath, exc)
        return False


def load_pickle(filepath: str | Path, default: Any = None) -> Any:
    """Load an object from a pickle file."""
    filepath = Path(filepath)

    if not filepath.exists():
        return default

    try:
        with open(filepath, "rb") as handle:
            return pickle.load(handle)
    except Exception as exc:
        logger.error("Error loading pickle %s: %s", filepath, exc)
        return default


def get_directory_size(directory: str | Path) -> int:
    """Calculate the total size of a directory in bytes."""
    directory_path = Path(directory)

    if not directory_path.exists() or not directory_path.is_dir():
        return 0

    total_size = 0
    for file in directory_path.rglob("*"):
        if file.is_file():
            total_size += file.stat().st_size

    return total_size


def clean_directory(
    directory: str | Path,
    older_than_days: int | None = None,
    pattern: str = "*",
) -> int:
    """Delete files from a directory according to the provided filters."""
    directory_path = Path(directory)

    if not directory_path.exists() or not directory_path.is_dir():
        return 0

    import time

    current_time = time.time()
    deleted_count = 0

    for file in directory_path.glob(pattern):
        if not file.is_file():
            continue

        if older_than_days is not None:
            file_age_days = (current_time - file.stat().st_mtime) / 86400
            if file_age_days < older_than_days:
                continue

        try:
            file.unlink()
            deleted_count += 1
        except Exception as exc:
            logger.error("Error deleting %s: %s", file, exc)

    return deleted_count
