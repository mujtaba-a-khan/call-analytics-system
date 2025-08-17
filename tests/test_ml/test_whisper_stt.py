"""
Test Suite for Whisper Speech-to-Text Module

Tests for Whisper STT initialization, transcription, and batch processing
in the Call Analytics System.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import json

from src.ml.whisper_stt import WhisperSTT, TranscriptionConfig, TranscriptionResult
from tests.test_ml import ML_TEST_DATA_DIR, ML_TEST_OUTPUT_DIR, TEST_WHISPER_CONFIG


class TestWhisperSTT(unittest.TestCase):
    """Test cases for WhisperSTT class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create mock audio files
        self.test_audio = self.temp_dir / 'test_audio.wav'
        self.test_audio.write_bytes(b'RIFF' + b'\x00' * 1000)  # Mock WAV file
        
        # Mock audio batch
        self.audio_batch = []
        for i in range(3):
            audio_path = self.temp_dir / f'audio_{i}.wav'
            audio_path.write_bytes(b'RIFF' + b'\x00' * 1000)
            self.audio_batch.append(audio_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('src.ml.whisper_stt.WhisperModel')
    def test_whisper_initialization(self, mock_model):
        """Test Whisper STT initialization with different configurations."""
        # Initialize with default settings
        stt = WhisperSTT()
        mock_model.assert_called_once_with(
            'small',
            device='cpu',
            compute_type='int8'
        )
        
        # Initialize with custom settings
        mock_model.reset_mock()
        stt = WhisperSTT(
            model_size='medium',
            device='cuda',
            compute_type='float16'
        )
        mock_model.assert_called_once_with(
            'medium',
            device='cuda',
            compute_type='float16'
        )
    
    @patch('src.ml.whisper_stt.WhisperModel')
    def test_transcribe_audio_basic(self, mock_model):
        """Test basic audio transcription."""
        # Setup mock transcription
        mock_transcribe = MagicMock()
        mock_transcribe.return_value = (
            [{'text': 'This is a test transcription', 'start': 0.0, 'end': 3.0}],
            {'language': 'en', 'language_probability': 0.99}
        )
        mock_model.return_value.transcribe = mock_transcribe
        
        # Initialize and transcribe
        stt = WhisperSTT()
        result = stt.transcribe(self.test_audio)
        
        # Verify transcription
        self.assertEqual(result['text'], 'This is a test transcription')
        self.assertEqual(result['language'], 'en')
        self.assertIn('segments', result)
        self.assertEqual(len(result['segments']), 1)
        mock_transcribe.assert_called_once()
    
    @patch('src.ml.whisper_stt.WhisperModel')
    def test_transcribe_with_language(self, mock_model):
        """Test transcription with specific language."""
        mock_transcribe = MagicMock()
        mock_transcribe.return_value = (
            [{'text': 'Texto en español', 'start': 0.0, 'end': 2.0}],
            {'language': 'es', 'language_probability': 0.95}
        )
        mock_model.return_value.transcribe = mock_transcribe
        
        stt = WhisperSTT()
        result = stt.transcribe(self.test_audio, language='es')
        
        # Verify language was passed
        call_args = mock_transcribe.call_args
        self.assertEqual(call_args[1].get('language'), 'es')
        self.assertEqual(result['language'], 'es')
    
    @patch('src.ml.whisper_stt.WhisperModel')
    def test_transcribe_with_timestamps(self, mock_model):
        """Test transcription with word-level timestamps."""
        mock_transcribe = MagicMock()
        mock_transcribe.return_value = (
            [{
                'text': 'Hello world',
                'start': 0.0,
                'end': 2.0,
                'words': [
                    {'word': 'Hello', 'start': 0.0, 'end': 0.8},
                    {'word': 'world', 'start': 0.9, 'end': 2.0}
                ]
            }],
            {'language': 'en', 'language_probability': 0.99}
        )
        mock_model.return_value.transcribe = mock_transcribe
        
        stt = WhisperSTT()
        result = stt.transcribe(self.test_audio, word_timestamps=True)
        
        # Verify word timestamps
        call_args = mock_transcribe.call_args
        self.assertTrue(call_args[1].get('word_timestamps'))
        self.assertIn('word_timestamps', result)
        self.assertEqual(len(result['word_timestamps']), 2)
    
    @patch('src.ml.whisper_stt.WhisperModel')
    def test_batch_transcription(self, mock_model):
        """Test batch audio transcription."""
        mock_transcribe = MagicMock()
        mock_transcribe.return_value = (
            [{'text': 'Batch test', 'start': 0.0, 'end': 1.0}],
            {'language': 'en', 'language_probability': 0.99}
        )
        mock_model.return_value.transcribe = mock_transcribe
        
        stt = WhisperSTT()
        results = stt.batch_transcribe(self.audio_batch)
        
        # Verify batch processing
        self.assertEqual(len(results), 3)
        self.assertEqual(mock_transcribe.call_count, 3)
        for result in results:
            self.assertEqual(result['text'], 'Batch test')
    
    @patch('src.ml.whisper_stt.WhisperModel')
    def test_transcription_with_vad(self, mock_model):
        """Test transcription with Voice Activity Detection."""
        mock_transcribe = MagicMock()
        mock_transcribe.return_value = (
            [{'text': 'Speech detected', 'start': 1.0, 'end': 3.0}],
            {'language': 'en', 'language_probability': 0.99}
        )
        mock_model.return_value.transcribe = mock_transcribe
        
        stt = WhisperSTT()
        result = stt.transcribe(self.test_audio, vad_filter=True)
        
        # Verify VAD was enabled
        call_args = mock_transcribe.call_args
        self.assertTrue(call_args[1].get('vad_filter'))
    
    @patch('src.ml.whisper_stt.WhisperModel')
    def test_transcription_error_handling(self, mock_model):
        """Test error handling during transcription."""
        mock_transcribe = MagicMock()
        mock_transcribe.side_effect = Exception("Transcription failed")
        mock_model.return_value.transcribe = mock_transcribe
        
        stt = WhisperSTT()
        
        # Should handle error gracefully
        with self.assertRaises(Exception) as context:
            stt.transcribe(self.test_audio)
        
        self.assertIn("Transcription failed", str(context.exception))
    
    def test_supported_languages(self):
        """Test getting list of supported languages."""
        languages = WhisperSTT.get_supported_languages()
        
        self.assertIsInstance(languages, list)
        self.assertIn('en', languages)
        self.assertIn('es', languages)
        self.assertIn('fr', languages)
        self.assertIn('de', languages)
        self.assertGreater(len(languages), 20)  # Whisper supports many languages
    
    def test_transcription_caching(self):
        """Test caching of transcription results."""
        with patch('src.ml.whisper_stt.WhisperModel') as mock_model:
            mock_transcribe = MagicMock()
            mock_transcribe.return_value = (
                [{'text': 'Cached result', 'start': 0.0, 'end': 1.0}],
                {'language': 'en', 'language_probability': 0.99}
            )
            mock_model.return_value.transcribe = mock_transcribe
            
            stt = WhisperSTT(enable_cache=True)
            
            # First transcription
            result1 = stt.transcribe(self.test_audio)
            
            # Second transcription (should use cache)
            result2 = stt.transcribe(self.test_audio)
            
            # Verify cache was used
            self.assertEqual(mock_transcribe.call_count, 1)
            self.assertEqual(result1['text'], result2['text'])


class TestTranscriptionConfig(unittest.TestCase):
    """Test cases for TranscriptionConfig class."""
    
    def test_default_config(self):
        """Test default transcription configuration."""
        config = TranscriptionConfig()
        
        self.assertEqual(config.language, 'auto')
        self.assertFalse(config.word_timestamps)
        self.assertTrue(config.vad_filter)
        self.assertEqual(config.beam_size, 5)
        self.assertIsNone(config.initial_prompt)
    
    def test_custom_config(self):
        """Test custom transcription configuration."""
        config = TranscriptionConfig(
            language='en',
            word_timestamps=True,
            vad_filter=False,
            beam_size=10,
            initial_prompt="Customer service call",
            temperature=0.5
        )
        
        self.assertEqual(config.language, 'en')
        self.assertTrue(config.word_timestamps)
        self.assertFalse(config.vad_filter)
        self.assertEqual(config.beam_size, 10)
        self.assertEqual(config.initial_prompt, "Customer service call")
        self.assertEqual(config.temperature, 0.5)
    
    def test_config_to_dict(self):
        """Test converting configuration to dictionary."""
        config = TranscriptionConfig(
            language='fr',
            word_timestamps=True
        )
        
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['language'], 'fr')
        self.assertTrue(config_dict['word_timestamps'])
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid beam size
        with self.assertRaises(ValueError):
            TranscriptionConfig(beam_size=0)
        
        # Test invalid temperature
        with self.assertRaises(ValueError):
            TranscriptionConfig(temperature=2.0)
        
        # Test invalid language code
        with self.assertRaises(ValueError):
            TranscriptionConfig(language='invalid_lang')


class TestTranscriptionResult(unittest.TestCase):
    """Test cases for TranscriptionResult class."""
    
    def test_result_creation(self):
        """Test creating TranscriptionResult instance."""
        result = TranscriptionResult(
            text="Test transcription",
            language="en",
            confidence=0.95,
            duration=10.5,
            segments=[
                {'text': 'Test', 'start': 0.0, 'end': 1.0},
                {'text': 'transcription', 'start': 1.0, 'end': 10.5}
            ]
        )
        
        self.assertEqual(result.text, "Test transcription")
        self.assertEqual(result.language, "en")
        self.assertAlmostEqual(result.confidence, 0.95)
        self.assertEqual(len(result.segments), 2)
    
    def test_result_from_whisper_output(self):
        """Test creating result from Whisper model output."""
        whisper_output = (
            [
                {'text': 'Hello world', 'start': 0.0, 'end': 2.0},
                {'text': 'How are you', 'start': 2.0, 'end': 4.0}
            ],
            {'language': 'en', 'language_probability': 0.98}
        )
        
        result = TranscriptionResult.from_whisper_output(whisper_output)
        
        self.assertEqual(result.text, "Hello world How are you")
        self.assertEqual(result.language, "en")
        self.assertAlmostEqual(result.confidence, 0.98)
        self.assertEqual(len(result.segments), 2)
    
    def test_result_to_json(self):
        """Test converting result to JSON."""
        result = TranscriptionResult(
            text="JSON test",
            language="en",
            confidence=0.9,
            duration=5.0
        )
        
        json_str = result.to_json()
        parsed = json.loads(json_str)
        
        self.assertEqual(parsed['text'], "JSON test")
        self.assertEqual(parsed['language'], "en")
        self.assertAlmostEqual(parsed['confidence'], 0.9)


if __name__ == '__main__':
    unittest.main()