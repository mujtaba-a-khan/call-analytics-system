"""
Pytest Configuration and Fixtures

This module contains pytest configuration, fixtures, and utilities
used across all test modules in the Call Analytics System.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
from unittest.mock import MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import test utilities
from tests import setup_test_environment, cleanup_test_environment


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """
    Configure pytest with custom settings and markers.
    
    Args:
        config: Pytest config object
    """
    # Add custom markers
    config.addinivalue_line(
        "markers", 
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers",
        "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers",
        "requires_model: marks tests that require ML models"
    )
    
    # Setup test environment
    setup_test_environment()


def pytest_unconfigure(config):
    """
    Cleanup after all tests are done.
    
    Args:
        config: Pytest config object
    """
    cleanup_test_environment()


# ============================================================================
# Session-scoped Fixtures
# ============================================================================

@pytest.fixture(scope='session')
def test_data_dir():
    """
    Provide test data directory path.
    
    Returns:
        Path to test data directory
    """
    data_dir = Path(__file__).parent / 'test_data'
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture(scope='session')
def test_output_dir():
    """
    Provide test output directory path.
    
    Returns:
        Path to test output directory
    """
    output_dir = Path(__file__).parent / 'test_output'
    output_dir.mkdir(exist_ok=True)
    yield output_dir
    # Cleanup after all tests
    if output_dir.exists():
        shutil.rmtree(output_dir)


# ============================================================================
# Function-scoped Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """
    Provide a temporary directory for test files.
    
    Yields:
        Path to temporary directory
    """
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def sample_call_data():
    """
    Generate sample call data for testing.
    
    Returns:
        DataFrame with sample call records
    """
    np.random.seed(42)
    num_records = 100
    
    data = pd.DataFrame({
        'call_id': [f'CALL_{i:04d}' for i in range(num_records)],
        'phone_number': [f'+123456789{i%10}' for i in range(num_records)],
        'timestamp': pd.date_range(start='2024-01-01', periods=num_records, freq='H'),
        'duration': np.random.randint(30, 600, num_records),
        'outcome': np.random.choice(['connected', 'no_answer', 'voicemail', 'busy'], num_records),
        'agent_id': [f'agent_{i%5:03d}' for i in range(num_records)],
        'campaign': np.random.choice(['sales', 'support', 'billing'], num_records),
        'revenue': np.random.choice([0, 0, 0, 50, 100, 200], num_records),
        'call_type': np.random.choice(['inbound', 'outbound'], num_records),
        'notes': [f'Test note {i}' for i in range(num_records)]
    })
    
    return data


@pytest.fixture
def sample_audio_file(temp_dir):
    """
    Create a sample audio file for testing.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        Path to sample audio file
    """
    audio_path = temp_dir / 'test_audio.wav'
    # Create mock WAV file with header
    audio_path.write_bytes(b'RIFF' + b'\x00' * 1000)
    return audio_path


@pytest.fixture
def sample_csv_file(temp_dir, sample_call_data):
    """
    Create a sample CSV file for testing.
    
    Args:
        temp_dir: Temporary directory fixture
        sample_call_data: Sample call data fixture
        
    Returns:
        Path to sample CSV file
    """
    csv_path = temp_dir / 'test_calls.csv'
    sample_call_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def mock_vector_store():
    """
    Create a mock vector store for testing.
    
    Returns:
        Mock vector store object
    """
    mock_store = MagicMock()
    
    # Setup common mock behaviors
    mock_store.query.return_value = {
        'documents': [['Test document']],
        'metadatas': [[{'call_id': 'CALL_001'}]],
        'distances': [[0.5]]
    }
    
    mock_store.add.return_value = True
    mock_store.delete.return_value = True
    mock_store.get_collection_size.return_value = 100
    
    return mock_store


@pytest.fixture
def mock_whisper_model():
    """
    Create a mock Whisper model for testing.
    
    Returns:
        Mock Whisper model object
    """
    mock_model = MagicMock()
    
    # Setup transcription behavior
    mock_model.transcribe.return_value = (
        [{'text': 'Test transcription', 'start': 0.0, 'end': 3.0}],
        {'language': 'en', 'language_probability': 0.99}
    )
    
    return mock_model


@pytest.fixture
def mock_embedding_model():
    """
    Create a mock embedding model for testing.
    
    Returns:
        Mock embedding model object
    """
    mock_model = MagicMock()
    
    # Setup embedding generation
    mock_model.encode.return_value = np.random.randn(1, 384)
    mock_model.get_sentence_embedding_dimension.return_value = 384
    
    return mock_model


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def test_config():
    """
    Provide test configuration.
    
    Returns:
        Dictionary with test configuration
    """
    return {
        'app': {
            'name': 'Call Analytics Test',
            'version': '1.0.0-test',
            'debug': True
        },
        'paths': {
            'data': 'test_data',
            'models': 'test_models',
            'logs': 'test_logs'
        },
        'whisper': {
            'model_size': 'tiny',
            'device': 'cpu',
            'compute_type': 'int8'
        },
        'embeddings': {
            'model': 'all-MiniLM-L6-v2',
            'dimension': 384
        }
    }


@pytest.fixture
def mock_storage_manager(sample_call_data):
    """
    Create a mock storage manager for testing.
    
    Args:
        sample_call_data: Sample call data fixture
        
    Returns:
        Mock storage manager object
    """
    mock_storage = MagicMock()
    
    # Setup common behaviors
    mock_storage.load_all_records.return_value = sample_call_data
    mock_storage.load_call_records.return_value = sample_call_data
    mock_storage.store_call_records.return_value = True
    mock_storage.get_unique_values.return_value = ['value1', 'value2']
    mock_storage.get_record_count.return_value = len(sample_call_data)
    
    return mock_storage


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def performance_timer():
    """
    Provide a performance timer for measuring test execution time.
    
    Yields:
        Timer object
    """
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            
        def start(self):
            self.start_time = time.time()
            
        def stop(self):
            self.end_time = time.time()
            
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    timer = Timer()
    timer.start()
    yield timer
    timer.stop()


@pytest.fixture
def capture_logs():
    """
    Capture log messages during tests.
    
    Yields:
        List of captured log records
    """
    import logging
    
    logs = []
    
    class LogCapture(logging.Handler):
        def emit(self, record):
            logs.append(record)
    
    handler = LogCapture()
    handler.setLevel(logging.DEBUG)
    
    # Add handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    
    yield logs
    
    # Remove handler
    root_logger.removeHandler(handler)


# ============================================================================
# Parametrized Test Data
# ============================================================================

def pytest_generate_tests(metafunc):
    """
    Generate parametrized test cases dynamically.
    
    Args:
        metafunc: Pytest metafunc object
    """
    # Parametrize audio formats
    if 'audio_format' in metafunc.fixturenames:
        metafunc.parametrize('audio_format', ['wav', 'mp3', 'flac', 'ogg'])
    
    # Parametrize CSV encodings
    if 'csv_encoding' in metafunc.fixturenames:
        metafunc.parametrize('csv_encoding', ['utf-8', 'latin-1', 'ascii'])
    
    # Parametrize embedding dimensions
    if 'embedding_dim' in metafunc.fixturenames:
        metafunc.parametrize('embedding_dim', [64, 128, 256, 384, 768])


# ============================================================================
# Markers and Skipping
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers or skip tests.
    
    Args:
        config: Pytest config object
        items: List of test items
    """
    # Skip tests requiring models if models not available
    skip_model = pytest.mark.skip(reason="ML models not available")
    
    for item in items:
        # Add markers based on test location
        if "test_ml" in str(item.fspath):
            item.add_marker(pytest.mark.requires_model)
        
        if "test_core" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        if "test_analysis" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Skip slow tests in quick mode
        if config.getoption("--quick"):
            if "slow" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Skipping slow test in quick mode"))


def pytest_addoption(parser):
    """
    Add custom command line options.
    
    Args:
        parser: Pytest parser object
    """
    parser.addoption(
        "--quick",
        action="store_true",
        default=False,
        help="Run quick tests only (skip slow tests)"
    )
    
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )