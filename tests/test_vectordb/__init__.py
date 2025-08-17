"""
Vector Database Module Test Package

This package contains tests for vector database functionality including
ChromaDB client operations and document indexing.
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Test configuration for vector database modules
VECTORDB_TEST_DATA_DIR = Path(__file__).parent / 'test_data'
VECTORDB_TEST_OUTPUT_DIR = Path(__file__).parent / 'test_output'
VECTORDB_TEST_DB_DIR = Path(__file__).parent / 'test_db'

# Create test directories if they don't exist
VECTORDB_TEST_DATA_DIR.mkdir(exist_ok=True)
VECTORDB_TEST_OUTPUT_DIR.mkdir(exist_ok=True)
VECTORDB_TEST_DB_DIR.mkdir(exist_ok=True)

# Test collection names
TEST_COLLECTION = 'test_collection'
TEST_COLLECTION_BACKUP = 'test_collection_backup'

__all__ = [
    'VECTORDB_TEST_DATA_DIR',
    'VECTORDB_TEST_OUTPUT_DIR',
    'VECTORDB_TEST_DB_DIR',
    'TEST_COLLECTION',
    'TEST_COLLECTION_BACKUP'
]