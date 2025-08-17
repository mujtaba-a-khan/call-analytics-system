"""
Machine Learning Module Test Package

This package contains tests for ML functionality including speech-to-text,
embeddings generation, and LLM integration.
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Test configuration for ML modules
ML_TEST_DATA_DIR = Path(__file__).parent / 'test_data'
ML_TEST_OUTPUT_DIR = Path(__file__).parent / 'test_output'
ML_TEST_MODELS_DIR = Path(__file__).parent / 'test_models'

# Create test directories if they don't exist
ML_TEST_DATA_DIR.mkdir(exist_ok=True)
ML_TEST_OUTPUT_DIR.mkdir(exist_ok=True)
ML_TEST_MODELS_DIR.mkdir(exist_ok=True)

# Mock model configurations for testing
TEST_WHISPER_CONFIG = {
    'model_size': 'tiny',
    'device': 'cpu',
    'compute_type': 'int8',
    'language': 'en'
}

TEST_EMBEDDING_CONFIG = {
    'model': 'all-MiniLM-L6-v2',
    'dimension': 384,
    'batch_size': 32
}

__all__ = [
    'ML_TEST_DATA_DIR',
    'ML_TEST_OUTPUT_DIR',
    'ML_TEST_MODELS_DIR',
    'TEST_WHISPER_CONFIG',
    'TEST_EMBEDDING_CONFIG'
]