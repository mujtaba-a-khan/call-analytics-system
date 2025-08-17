# Test Suite Documentation

Comprehensive test suite for the Call Analytics System with modular organization and extensive coverage.

## Test Structure

```
tests/
├── __init__.py                    # Test package initialization
├── conftest.py                    # Pytest configuration and fixtures
├── test_README.md                    # Test Suite Readme
├── test_core/                     # Core module tests
│   ├── __init__.py
│   ├── test_audio_processor.py   # Audio processing tests
│   ├── test_csv_processor.py     # CSV processing tests
│   ├── test_data_schema.py       # Data validation tests
│   ├── test_labeling_engine.py   # Labeling system tests
│   └── test_storage_manager.py   # Storage management tests
├── test_ml/                       # Machine learning tests
│   ├── __init__.py
│   ├── test_whisper_stt.py       # Speech-to-text tests
│   ├── test_embeddings.py        # Embedding generation tests
│   └── test_llm_client.py        # LLM integration tests
├── test_analysis/                 # Analysis module tests
│   ├── __init__.py
│   ├── test_aggregations.py      # Metrics calculation tests
│   ├── test_filters.py           # Filtering logic tests
│   ├── test_semantic_search.py   # Semantic search tests
│   └── test_query_interpreter.py # Query interpretation tests
├── test_vectordb/                 # Vector database tests
│   ├── __init__.py
│   ├── test_chroma_client.py     # ChromaDB client tests
│   └── test_indexer.py           # Indexing tests
├── test_ui/                       # UI component tests
│   ├── __init__.py
│   ├── test_components.py        # UI component tests
│   └── test_pages.py             # Page functionality tests
├── test_utils/                    # Utility tests
│   ├── __init__.py
│   ├── test_validators.py        # Validation utility tests
│   └── test_formatters.py        # Formatting utility tests
└── test_data/                     # Test data files
    └── sample_calls.csv
```

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test module
pytest tests/test_core/

# Run specific test file
pytest tests/test_core/test_audio_processor.py

# Run specific test function
pytest tests/test_core/test_audio_processor.py::TestAudioProcessor::test_process_audio_wav
```

### Test Categories

```bash
# Run only unit tests
pytest tests/ -m unit

# Run only integration tests
pytest tests/ -m integration

# Run tests excluding slow tests
pytest tests/ -m "not slow"

# Run quick tests only (custom option)
pytest tests/ --quick

# Run including slow tests
pytest tests/ --runslow
```

### Test Coverage

```bash
# Generate coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Generate HTML coverage report
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html in browser

# Generate XML coverage report (for CI/CD)
pytest tests/ --cov=src --cov-report=xml
```

## Test Fixtures

### Common Fixtures

Available in `conftest.py`:

- `temp_dir`: Temporary directory for test files
- `sample_call_data`: DataFrame with sample call records
- `sample_audio_file`: Mock audio file for testing
- `sample_csv_file`: Sample CSV file with call data
- `mock_vector_store`: Mock vector database
- `mock_whisper_model`: Mock Whisper STT model
- `mock_embedding_model`: Mock embedding model
- `test_config`: Test configuration dictionary

### Using Fixtures

```python
def test_example(sample_call_data, temp_dir):
    """Example test using fixtures."""
    # sample_call_data is automatically provided
    assert len(sample_call_data) == 100
    
    # temp_dir is cleaned up automatically
    test_file = temp_dir / 'test.txt'
    test_file.write_text('test content')
```

## Test Markers

### Built-in Markers

- `@pytest.mark.slow`: Marks slow-running tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.requires_model`: Tests requiring ML models

### Using Markers

```python
@pytest.mark.slow
def test_large_dataset_processing():
    """This test takes a long time."""
    pass

@pytest.mark.requires_model
def test_whisper_transcription():
    """This test needs Whisper model."""
    pass
```

## Writing Tests

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Example Test Structure

```python
import unittest
from src.core.module import MyClass

class TestMyClass(unittest.TestCase):
    """Test cases for MyClass."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.instance = MyClass()
    
    def tearDown(self):
        """Clean up after tests."""
        self.instance.cleanup()
    
    def test_functionality(self):
        """Test specific functionality."""
        result = self.instance.method()
        self.assertEqual(result, expected_value)
```

## Continuous Integration

### GitHub Actions Configuration

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.13'
    
    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## Performance Testing

### Running Performance Tests

```bash
# Run performance benchmarks
pytest tests/ -m performance --benchmark-only

# Generate performance report
pytest tests/ --benchmark-autosave
```

### Example Performance Test

```python
@pytest.mark.performance
def test_processing_speed(benchmark, sample_call_data):
    """Benchmark processing speed."""
    result = benchmark(process_data, sample_call_data)
    assert result is not None
```

## Debugging Tests

### Using pytest debugger

```bash
# Drop into debugger on failure
pytest tests/ --pdb

# Drop into debugger at start of test
pytest tests/ --trace

# Show local variables on failure
pytest tests/ -l
```

### Logging in Tests

```python
def test_with_logging(caplog):
    """Test with log capture."""
    import logging
    
    # Your test code
    logging.info("Test message")
    
    # Check logs
    assert "Test message" in caplog.text
```

## Test Data Management

### Using Test Data

```python
from tests.test_analysis import create_sample_call_data

def test_with_data():
    """Test using sample data."""
    data = create_sample_call_data(num_records=500)
    assert len(data) == 500
```

### Mock Data Generation

```python
import pandas as pd
import numpy as np

def generate_test_calls(n=100):
    """Generate test call records."""
    return pd.DataFrame({
        'call_id': [f'CALL_{i:04d}' for i in range(n)],
        'duration': np.random.randint(30, 600, n),
        # ... other fields
    })
```

## Best Practices

### 1. Test Isolation
- Each test should be independent
- Use fixtures for setup/teardown
- Don't rely on test execution order

### 2. Clear Test Names
- Use descriptive test names
- Include what is being tested and expected outcome

### 3. Arrange-Act-Assert Pattern
```python
def test_calculation():
    # Arrange
    calculator = Calculator()
    
    # Act
    result = calculator.add(2, 3)
    
    # Assert
    assert result == 5
```

### 4. Use Appropriate Assertions
```python
# Good
assert result == expected_value
assert error_message in str(exception)

# Better with pytest
assert result == pytest.approx(3.14, rel=1e-2)
assert "error" in caplog.text
```

### 5. Mock External Dependencies
```python
@patch('src.module.external_api')
def test_with_mock(mock_api):
    mock_api.return_value = {'status': 'success'}
    result = function_using_api()
    assert result['status'] == 'success'
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure src is in Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **Missing Dependencies**
   ```bash
   pip install -e ".[test]"
   ```

3. **Slow Tests**
   ```bash
   # Skip slow tests
   pytest tests/ -m "not slow"
   ```

4. **Model-dependent Tests Failing**
   ```bash
   # Download required models first
   python scripts/download_models.py
   ```

## Test Reports

### Generating Reports

```bash
# JUnit XML report (for CI/CD)
pytest tests/ --junit-xml=test-results.xml

# HTML report
pytest tests/ --html=report.html --self-contained-html

# JSON report
pytest tests/ --json-report --json-report-file=report.json
```

## Contributing Tests

When adding new tests:

1. Follow the existing structure
2. Add appropriate markers
3. Include docstrings
4. Ensure tests are isolated
5. Update this README if needed

## License

Test suite is part of the Call Analytics System and follows the same license.