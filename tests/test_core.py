"""
Test Suite for Core Modules

This module contains comprehensive tests for core functionality including
data schema, audio processing, CSV processing, storage management, and
labeling engine.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import json

# Import modules to test
from src.core import (
    CallRecord,
    AudioProcessor,
    CSVProcessor,
    StorageManager,
    LabelingEngine,
    ValidationError
)
from src.core.data_schema import AudioFile, TranscriptionResult
from tests import TEST_DATA_DIR, TEST_OUTPUT_DIR


class TestDataSchema(unittest.TestCase):
    """Test cases for data schema and validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.valid_call_data = {
            'call_id': 'TEST_001',
            'phone_number': '+1234567890',
            'call_type': 'inbound',
            'outcome': 'connected',
            'duration': 300,
            'timestamp': datetime.now(),
            'agent_id': 'agent_001',
            'campaign': 'sales',
            'notes': 'Test call',
            'revenue': 100.0
        }
    
    def test_call_record_creation(self):
        """Test creating a valid CallRecord."""
        record = CallRecord(**self.valid_call_data)
        
        self.assertEqual(record.call_id, 'TEST_001')
        self.assertEqual(record.phone_number, '+1234567890')
        self.assertEqual(record.duration, 300)
        self.assertIsInstance(record.timestamp, datetime)
    
    def test_call_record_validation(self):
        """Test CallRecord validation rules."""
        # Test invalid phone number
        invalid_data = self.valid_call_data.copy()
        invalid_data['phone_number'] = 'invalid'
        
        with self.assertRaises(ValidationError):
            CallRecord(**invalid_data)
        
        # Test negative duration
        invalid_data = self.valid_call_data.copy()
        invalid_data['duration'] = -10
        
        with self.assertRaises(ValidationError):
            CallRecord(**invalid_data)
    
    def test_call_record_serialization(self):
        """Test CallRecord serialization to dictionary."""
        record = CallRecord(**self.valid_call_data)
        record_dict = record.dict()
        
        self.assertIsInstance(record_dict, dict)
        self.assertIn('call_id', record_dict)
        self.assertEqual(record_dict['call_id'], 'TEST_001')
    
    def test_audio_file_schema(self):
        """Test AudioFile schema validation."""
        audio_data = {
            'file_path': Path('/test/audio.wav'),
            'format': 'wav',
            'duration': 120.5,
            'sample_rate': 16000,
            'channels': 1,
            'file_size': 1024000
        }
        
        audio_file = AudioFile(**audio_data)
        self.assertEqual(audio_file.format, 'wav')
        self.assertEqual(audio_file.sample_rate, 16000)
    
    def test_transcription_result(self):
        """Test TranscriptionResult schema."""
        transcription_data = {
            'text': 'This is a test transcription',
            'language': 'en',
            'confidence': 0.95,
            'duration': 10.5,
            'segments': [],
            'word_timestamps': []
        }
        
        result = TranscriptionResult(**transcription_data)
        self.assertEqual(result.text, 'This is a test transcription')
        self.assertAlmostEqual(result.confidence, 0.95)


class TestAudioProcessor(unittest.TestCase):
    """Test cases for audio processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.processor = AudioProcessor(output_dir=self.temp_dir)
        
        # Create a mock audio file
        self.test_audio_path = self.temp_dir / 'test_audio.wav'
        self.test_audio_path.write_bytes(b'fake audio data')
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('src.core.audio_processor.sf.read')
    @patch('src.core.audio_processor.sf.write')
    def test_process_audio(self, mock_write, mock_read):
        """Test audio processing pipeline."""
        # Mock audio data
        mock_audio_data = np.random.randn(16000)  # 1 second at 16kHz
        mock_read.return_value = (mock_audio_data, 16000)
        
        # Process audio
        result_path = self.processor.process_audio(self.test_audio_path)
        
        # Verify processing
        self.assertTrue(mock_read.called)
        self.assertTrue(mock_write.called)
        self.assertIsInstance(result_path, Path)
    
    def test_validate_audio_format(self):
        """Test audio format validation."""
        # Test valid formats
        valid_formats = ['wav', 'mp3', 'mp4', 'flac']
        for fmt in valid_formats:
            test_file = self.temp_dir / f'test.{fmt}'
            test_file.write_bytes(b'data')
            self.assertTrue(self.processor.validate_format(test_file))
        
        # Test invalid format
        invalid_file = self.temp_dir / 'test.txt'
        invalid_file.write_bytes(b'data')
        self.assertFalse(self.processor.validate_format(invalid_file))
    
    @patch('src.core.audio_processor.librosa.load')
    def test_extract_audio_features(self, mock_load):
        """Test audio feature extraction."""
        mock_load.return_value = (np.random.randn(16000), 16000)
        
        features = self.processor.extract_features(self.test_audio_path)
        
        self.assertIn('duration', features)
        self.assertIn('sample_rate', features)
        self.assertIn('energy', features)


class TestCSVProcessor(unittest.TestCase):
    """Test cases for CSV processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.processor = CSVProcessor()
        
        # Create sample CSV data
        self.sample_data = pd.DataFrame({
            'phone': ['+1234567890', '+0987654321'],
            'date': ['2024-01-01 10:00:00', '2024-01-01 11:00:00'],
            'duration': [300, 450],
            'outcome': ['connected', 'voicemail'],
            'agent': ['agent_001', 'agent_002']
        })
        
        self.csv_path = self.temp_dir / 'test_calls.csv'
        self.sample_data.to_csv(self.csv_path, index=False)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_auto_map_fields(self):
        """Test automatic field mapping."""
        headers = ['phone', 'date', 'duration', 'outcome', 'agent']
        mapping = self.processor.auto_map_fields(headers)
        
        self.assertEqual(mapping['phone_number'], 'phone')
        self.assertEqual(mapping['timestamp'], 'date')
        self.assertEqual(mapping['duration'], 'duration')
        self.assertEqual(mapping['agent_id'], 'agent')
    
    def test_process_csv_file(self):
        """Test CSV file processing."""
        records = list(self.processor.process_csv_file(self.csv_path))
        
        self.assertEqual(len(records), 2)
        self.assertIsInstance(records[0], CallRecord)
        self.assertEqual(records[0].phone_number, '+1234567890')
    
    def test_validate_csv_structure(self):
        """Test CSV structure validation."""
        # Test valid CSV
        self.assertTrue(self.processor.validate_structure(self.csv_path))
        
        # Test invalid CSV (missing required fields)
        invalid_data = pd.DataFrame({'random_column': [1, 2, 3]})
        invalid_path = self.temp_dir / 'invalid.csv'
        invalid_data.to_csv(invalid_path, index=False)
        
        self.assertFalse(self.processor.validate_structure(invalid_path))
    
    def test_handle_encoding_issues(self):
        """Test handling different file encodings."""
        # Create CSV with different encoding
        utf8_path = self.temp_dir / 'utf8.csv'
        self.sample_data.to_csv(utf8_path, index=False, encoding='utf-8')
        
        encoding = self.processor.detect_encoding(utf8_path)
        self.assertIn(encoding.lower(), ['utf-8', 'ascii'])


class TestStorageManager(unittest.TestCase):
    """Test cases for storage management functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage = StorageManager(data_dir=self.temp_dir)
        
        # Create sample records
        self.sample_records = [
            CallRecord(
                call_id=f'TEST_{i:03d}',
                phone_number=f'+123456789{i}',
                call_type='inbound',
                outcome='connected',
                duration=300 + i * 10,
                timestamp=datetime.now() - timedelta(days=i),
                agent_id=f'agent_{i % 3:03d}',
                campaign='test',
                notes=f'Test call {i}',
                revenue=100.0 * i
            )
            for i in range(10)
        ]
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_store_call_records(self):
        """Test storing call records."""
        success = self.storage.store_call_records(self.sample_records)
        
        self.assertTrue(success)
        
        # Verify records were stored
        stored_path = self.temp_dir / 'call_records.parquet'
        self.assertTrue(stored_path.exists())
    
    def test_load_call_records(self):
        """Test loading call records."""
        # Store records first
        self.storage.store_call_records(self.sample_records)
        
        # Load records
        loaded_records = self.storage.load_all_records()
        
        self.assertIsInstance(loaded_records, pd.DataFrame)
        self.assertEqual(len(loaded_records), 10)
    
    def test_date_range_filtering(self):
        """Test loading records with date range filter."""
        # Store records
        self.storage.store_call_records(self.sample_records)
        
        # Load records from last 5 days
        start_date = datetime.now().date() - timedelta(days=5)
        end_date = datetime.now().date()
        
        filtered_records = self.storage.load_call_records(start_date, end_date)
        
        self.assertLessEqual(len(filtered_records), 6)  # Days 0-5
    
    def test_get_unique_values(self):
        """Test getting unique values for a column."""
        self.storage.store_call_records(self.sample_records)
        
        # Get unique agents
        agents = self.storage.get_unique_values('agent_id')
        
        self.assertEqual(len(agents), 3)  # agent_000, agent_001, agent_002
        self.assertIn('agent_000', agents)
    
    def test_backup_and_restore(self):
        """Test backup and restore functionality."""
        # Store original data
        self.storage.store_call_records(self.sample_records)
        
        # Create backup
        backup_path = self.temp_dir / 'backup'
        backup_success = self.storage.backup_data(backup_path)
        self.assertTrue(backup_success)
        
        # Clear data
        self.storage.clear_all_data()
        empty_records = self.storage.load_all_records()
        self.assertEqual(len(empty_records), 0)
        
        # Restore from backup
        restore_success = self.storage.restore_data(backup_path)
        self.assertTrue(restore_success)
        
        # Verify restored data
        restored_records = self.storage.load_all_records()
        self.assertEqual(len(restored_records), 10)


class TestLabelingEngine(unittest.TestCase):
    """Test cases for labeling engine functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = LabelingEngine()
        
        # Create sample call record
        self.sample_call = {
            'call_id': 'TEST_001',
            'phone_number': '+1234567890',
            'duration': 300,
            'outcome': 'connected',
            'notes': 'Customer complained about billing issue',
            'transcript': 'I need help with my bill. The charges are incorrect.'
        }
    
    def test_apply_labeling_rules(self):
        """Test applying labeling rules to calls."""
        # Define test rules
        rules = [
            {
                'name': 'billing_issue',
                'conditions': {
                    'text_contains': ['bill', 'billing', 'charge', 'payment']
                },
                'label': 'billing_inquiry'
            },
            {
                'name': 'complaint',
                'conditions': {
                    'text_contains': ['complaint', 'complain', 'unhappy', 'disappointed']
                },
                'label': 'customer_complaint'
            }
        ]
        
        self.engine.load_rules(rules)
        
        # Apply rules
        labels = self.engine.label_call(self.sample_call)
        
        self.assertIn('billing_inquiry', labels)
        self.assertIn('customer_complaint', labels)
    
    def test_duration_based_rules(self):
        """Test duration-based labeling rules."""
        rules = [
            {
                'name': 'long_call',
                'conditions': {
                    'duration_greater_than': 240
                },
                'label': 'long_duration'
            },
            {
                'name': 'short_call',
                'conditions': {
                    'duration_less_than': 60
                },
                'label': 'short_duration'
            }
        ]
        
        self.engine.load_rules(rules)
        
        # Test long call
        labels = self.engine.label_call(self.sample_call)
        self.assertIn('long_duration', labels)
        self.assertNotIn('short_duration', labels)
        
        # Test short call
        short_call = self.sample_call.copy()
        short_call['duration'] = 30
        labels = self.engine.label_call(short_call)
        self.assertIn('short_duration', labels)
        self.assertNotIn('long_duration', labels)
    
    def test_outcome_based_rules(self):
        """Test outcome-based labeling rules."""
        rules = [
            {
                'name': 'successful_call',
                'conditions': {
                    'outcome_equals': 'connected'
                },
                'label': 'successful_connection'
            }
        ]
        
        self.engine.load_rules(rules)
        labels = self.engine.label_call(self.sample_call)
        
        self.assertIn('successful_connection', labels)
    
    def test_complex_rules(self):
        """Test complex rules with multiple conditions."""
        rules = [
            {
                'name': 'priority_complaint',
                'conditions': {
                    'text_contains': ['complaint', 'unhappy'],
                    'duration_greater_than': 180,
                    'outcome_equals': 'connected'
                },
                'label': 'priority_escalation',
                'operator': 'AND'
            }
        ]
        
        self.engine.load_rules(rules)
        labels = self.engine.label_call(self.sample_call)
        
        self.assertIn('priority_escalation', labels)


def suite():
    """Create test suite for core modules."""
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestDataSchema))
    suite.addTest(unittest.makeSuite(TestAudioProcessor))
    suite.addTest(unittest.makeSuite(TestCSVProcessor))
    suite.addTest(unittest.makeSuite(TestStorageManager))
    suite.addTest(unittest.makeSuite(TestLabelingEngine))
    
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())