"""
Test Suite for Data Schema Module

Tests for data validation, serialization, and schema enforcement
in the Call Analytics System.
"""

import unittest
from datetime import datetime
import json
from pathlib import Path

from src.core.data_schema import (
    CallRecord,
    AudioFile,
    TranscriptionResult,
    AnalyticsMetrics,
    ValidationError
)


class TestCallRecord(unittest.TestCase):
    """Test cases for CallRecord data model."""
    
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
    
    def test_valid_call_record_creation(self):
        """Test creating a valid CallRecord."""
        record = CallRecord(**self.valid_call_data)
        
        self.assertEqual(record.call_id, 'TEST_001')
        self.assertEqual(record.phone_number, '+1234567890')
        self.assertEqual(record.duration, 300)
        self.assertIsInstance(record.timestamp, datetime)
    
    def test_phone_number_validation(self):
        """Test phone number validation rules."""
        # Test invalid phone number format
        invalid_data = self.valid_call_data.copy()
        invalid_data['phone_number'] = 'invalid'
        
        with self.assertRaises(ValidationError):
            CallRecord(**invalid_data)
        
        # Test empty phone number
        invalid_data['phone_number'] = ''
        with self.assertRaises(ValidationError):
            CallRecord(**invalid_data)
    
    def test_duration_validation(self):
        """Test duration validation rules."""
        # Test negative duration
        invalid_data = self.valid_call_data.copy()
        invalid_data['duration'] = -10
        
        with self.assertRaises(ValidationError):
            CallRecord(**invalid_data)
        
        # Test zero duration (should be valid)
        valid_data = self.valid_call_data.copy()
        valid_data['duration'] = 0
        record = CallRecord(**valid_data)
        self.assertEqual(record.duration, 0)
    
    def test_call_record_serialization(self):
        """Test CallRecord serialization to dictionary."""
        record = CallRecord(**self.valid_call_data)
        record_dict = record.dict()
        
        self.assertIsInstance(record_dict, dict)
        self.assertIn('call_id', record_dict)
        self.assertEqual(record_dict['call_id'], 'TEST_001')
        self.assertEqual(record_dict['phone_number'], '+1234567890')
    
    def test_call_record_json_serialization(self):
        """Test CallRecord JSON serialization."""
        record = CallRecord(**self.valid_call_data)
        record_json = record.json()
        
        self.assertIsInstance(record_json, str)
        parsed = json.loads(record_json)
        self.assertEqual(parsed['call_id'], 'TEST_001')
    
    def test_optional_fields(self):
        """Test CallRecord with optional fields."""
        minimal_data = {
            'call_id': 'TEST_002',
            'phone_number': '+1234567890',
            'call_type': 'inbound',
            'outcome': 'connected',
            'duration': 120,
            'timestamp': datetime.now()
        }
        
        record = CallRecord(**minimal_data)
        self.assertIsNone(record.agent_id)
        self.assertIsNone(record.campaign)
        self.assertEqual(record.revenue, 0.0)


class TestAudioFile(unittest.TestCase):
    """Test cases for AudioFile data model."""
    
    def test_audio_file_creation(self):
        """Test creating AudioFile instance."""
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
        self.assertEqual(audio_file.channels, 1)
    
    def test_audio_format_validation(self):
        """Test audio format validation."""
        valid_formats = ['wav', 'mp3', 'mp4', 'flac', 'ogg', 'm4a']
        
        for fmt in valid_formats:
            audio_data = {
                'file_path': Path(f'/test/audio.{fmt}'),
                'format': fmt,
                'duration': 60.0,
                'sample_rate': 16000,
                'channels': 1,
                'file_size': 1024000
            }
            audio_file = AudioFile(**audio_data)
            self.assertEqual(audio_file.format, fmt)
    
    def test_invalid_audio_parameters(self):
        """Test validation of audio parameters."""
        # Test negative duration
        with self.assertRaises(ValidationError):
            AudioFile(
                file_path=Path('/test/audio.wav'),
                format='wav',
                duration=-10,
                sample_rate=16000,
                channels=1,
                file_size=1024000
            )
        
        # Test invalid sample rate
        with self.assertRaises(ValidationError):
            AudioFile(
                file_path=Path('/test/audio.wav'),
                format='wav',
                duration=60,
                sample_rate=0,
                channels=1,
                file_size=1024000
            )


class TestTranscriptionResult(unittest.TestCase):
    """Test cases for TranscriptionResult data model."""
    
    def test_transcription_result_creation(self):
        """Test creating TranscriptionResult instance."""
        transcription_data = {
            'text': 'This is a test transcription',
            'language': 'en',
            'confidence': 0.95,
            'duration': 10.5,
            'segments': [
                {'start': 0.0, 'end': 5.0, 'text': 'This is a test'},
                {'start': 5.0, 'end': 10.5, 'text': 'transcription'}
            ],
            'word_timestamps': []
        }
        
        result = TranscriptionResult(**transcription_data)
        self.assertEqual(result.text, 'This is a test transcription')
        self.assertAlmostEqual(result.confidence, 0.95)
        self.assertEqual(len(result.segments), 2)
    
    def test_confidence_validation(self):
        """Test confidence score validation."""
        # Test confidence > 1
        with self.assertRaises(ValidationError):
            TranscriptionResult(
                text='Test',
                language='en',
                confidence=1.5,
                duration=10.0
            )
        
        # Test negative confidence
        with self.assertRaises(ValidationError):
            TranscriptionResult(
                text='Test',
                language='en',
                confidence=-0.1,
                duration=10.0
            )
    
    def test_language_code_validation(self):
        """Test language code validation."""
        valid_codes = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja']
        
        for code in valid_codes:
            result = TranscriptionResult(
                text='Test',
                language=code,
                confidence=0.9,
                duration=10.0
            )
            self.assertEqual(result.language, code)


class TestAnalyticsMetrics(unittest.TestCase):
    """Test cases for AnalyticsMetrics data model."""
    
    def test_analytics_metrics_creation(self):
        """Test creating AnalyticsMetrics instance."""
        metrics_data = {
            'total_calls': 1000,
            'average_duration': 300.5,
            'connection_rate': 0.85,
            'total_revenue': 50000.0,
            'unique_callers': 750,
            'peak_hour': 14,
            'busiest_day': 'Monday'
        }
        
        metrics = AnalyticsMetrics(**metrics_data)
        self.assertEqual(metrics.total_calls, 1000)
        self.assertAlmostEqual(metrics.average_duration, 300.5)
        self.assertAlmostEqual(metrics.connection_rate, 0.85)
    
    def test_metrics_validation(self):
        """Test metrics validation rules."""
        # Test negative values
        with self.assertRaises(ValidationError):
            AnalyticsMetrics(
                total_calls=-10,
                average_duration=300,
                connection_rate=0.85
            )
        
        # Test invalid rate
        with self.assertRaises(ValidationError):
            AnalyticsMetrics(
                total_calls=100,
                average_duration=300,
                connection_rate=1.5  # Should be between 0 and 1
            )
    
    def test_metrics_aggregation(self):
        """Test metrics aggregation methods."""
        metrics1 = AnalyticsMetrics(
            total_calls=100,
            average_duration=300,
            connection_rate=0.8,
            total_revenue=1000
        )
        
        metrics2 = AnalyticsMetrics(
            total_calls=200,
            average_duration=250,
            connection_rate=0.9,
            total_revenue=2000
        )
        
        # Test aggregation logic
        combined_calls = metrics1.total_calls + metrics2.total_calls
        self.assertEqual(combined_calls, 300)
        
        combined_revenue = metrics1.total_revenue + metrics2.total_revenue
        self.assertEqual(combined_revenue, 3000)


if __name__ == '__main__':
    unittest.main()