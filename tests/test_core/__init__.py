"""
Core Module Test Package

This package contains tests for core functionality including data schema,
audio processing, CSV processing, storage management, and labeling engine.
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Test configuration for core modules
CORE_TEST_DATA_DIR = Path(__file__).parent / 'test_data'
CORE_TEST_OUTPUT_DIR = Path(__file__).parent / 'test_output'

# Create test directories if they don't exist
CORE_TEST_DATA_DIR.mkdir(exist_ok=True)
CORE_TEST_OUTPUT_DIR.mkdir(exist_ok=True)

__all__ = [
    'CORE_TEST_DATA_DIR',
    'CORE_TEST_OUTPUT_DIR'
]