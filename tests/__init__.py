"""
Test Suite for Call Analytics System

This package contains comprehensive tests for all modules in the
Call Analytics System including unit tests, integration tests, and
performance tests.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test configuration
TEST_DATA_DIR = Path(__file__).parent / 'test_data'
TEST_OUTPUT_DIR = Path(__file__).parent / 'test_output'

# Create test directories if they don't exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

# Test fixtures and utilities
def setup_test_environment():
    """
    Setup test environment with necessary configurations.
    """
    os.environ['TESTING'] = 'true'
    os.environ['LOG_LEVEL'] = 'DEBUG'
    
    # Create test directories
    TEST_DATA_DIR.mkdir(exist_ok=True)
    TEST_OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Setup test logging
    from src.utils.logger import setup_logging
    setup_logging(
        log_level='DEBUG',
        log_dir=TEST_OUTPUT_DIR / 'logs',
        console_output=True,
        file_output=True
    )

def cleanup_test_environment():
    """
    Cleanup test environment after tests.
    """
    # Clean up test output files
    import shutil
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    
    # Reset environment variables
    if 'TESTING' in os.environ:
        del os.environ['TESTING']

# Package version
__version__ = '1.0.0'