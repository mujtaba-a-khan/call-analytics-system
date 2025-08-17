"""
Analysis Module Test Package

This package contains tests for analysis functionality including metrics
calculation, semantic search, query interpretation, and filtering.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Test configuration for analysis modules
ANALYSIS_TEST_DATA_DIR = Path(__file__).parent / 'test_data'
ANALYSIS_TEST_OUTPUT_DIR = Path(__file__).parent / 'test_output'

# Create test directories if they don't exist
ANALYSIS_TEST_DATA_DIR.mkdir(exist_ok=True)
ANALYSIS_TEST_OUTPUT_DIR.mkdir(exist_ok=True)

# Helper function to create sample call data for testing
def create_sample_call_data(num_records: int = 100) -> pd.DataFrame:
    """
    Create sample call data for testing analysis functions.
    
    Args:
        num_records: Number of records to generate
        
    Returns:
        DataFrame with sample call data
    """
    np.random.seed(42)  # For reproducible tests
    
    return pd.DataFrame({
        'call_id': [f'CALL_{i:04d}' for i in range(num_records)],
        'phone_number': [f'+1234567{i%100:03d}' for i in range(num_records)],
        'timestamp': pd.date_range(start='2024-01-01', periods=num_records, freq='H'),
        'duration': np.random.randint(30, 600, num_records),
        'outcome': np.random.choice(['connected', 'no_answer', 'voicemail', 'busy', 'failed'], num_records),
        'agent_id': [f'agent_{i%10:03d}' for i in range(num_records)],
        'campaign': np.random.choice(['sales', 'support', 'billing', 'retention'], num_records),
        'revenue': np.random.choice([0, 0, 0, 50, 100, 200], num_records),
        'call_type': np.random.choice(['inbound', 'outbound'], num_records),
        'notes': [f'Note for call {i}' for i in range(num_records)],
        'transcript': [f'Transcript content for call {i}' for i in range(num_records)]
    })

__all__ = [
    'ANALYSIS_TEST_DATA_DIR',
    'ANALYSIS_TEST_OUTPUT_DIR',
    'create_sample_call_data'
]