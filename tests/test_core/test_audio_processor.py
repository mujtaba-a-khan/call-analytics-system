"""
Test Suite for Audio Processor Module

Tests for audio file processing, format conversion, and feature extraction
in the Call Analytics System.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.core.audio_processor import AudioProcessor, AudioFormat, AudioMetadata
from tests.test_core import CORE_TEST_DATA_DIR, CORE_TEST_OUTPUT_DIR


class TestAudioProcessor(unittest.TestCase):
    """Test cases for AudioProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.processor = AudioProcessor(output_dir=self.temp_dir)
        
        # Create mock audio files
        self.test_wav = self.temp_dir / 'test.wav'
        self.test_mp3 = self.temp_dir / 'test.mp3'
        self.test_wav.write_bytes(b'RIFF' + b'\x00' * 100)  # Mock WAV header
        self.test_mp3.write_bytes(b'ID3' + b'\x00' * 100)   # Mock MP3 header
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('src.core.audio_processor.sf.read')
    @patch('src.core.audio_processor.sf.write')
    def test_process_audio_wav(self, mock_write, mock_read):
        """Test processing WAV audio files."""
        # Mock audio data - 1 second at 16kHz
        mock_audio_data = np.random.randn(16000)
        mock_read.return_value = (mock_audio_data, 16000)
        
        # Process audio
        result_path = self.processor.process_audio(self.test_wav)
        
        # Verify processing
        self.assertTrue(mock_read.called)
        self.assertTrue(mock_write.called)
        self.assertIsInstance(result_path, Path)
        self.assertTrue(str(result_path).endswith('.wav'))
    
    @patch('src.core.audio_processor.librosa.load')
    @patch('src.core.audio_processor.sf.write')
    def test_process_audio_mp3(self, mock_write, mock_load):
        """Test processing MP3 audio files."""
        # Mock audio data
        mock_audio_data = np.random.randn(16000)
        mock_load.return_value = (mock_audio_data, 16000)
        
        # Process MP3
        result_path = self.processor.process_audio(self.test_mp3)
        
        # Verify MP3 was loaded with librosa
        self.assertTrue(mock_load.called)
        self.assertTrue(mock_write.called)
    
    def test_validate_audio_format(self):
        """Test audio format validation."""
        # Test valid formats
        valid_formats = ['wav', 'mp3', 'mp4', 'flac', 'ogg', 'm4a']
        for fmt in valid_formats:
            test_file = self.temp_dir / f'test.{fmt}'
            test_file.write_bytes(b'data')
            self.assertTrue(self.processor.validate_format(test_file))
        
        # Test invalid format
        invalid_file = self.temp_dir / 'test.txt'
        invalid_file.write_bytes(b'data')
        self.assertFalse(self.processor.validate_format(invalid_file))
    
    @patch('src.core.audio_processor.sf.read')
    def test_normalize_audio(self, mock_read):
        """Test audio normalization."""
        # Create audio with varying amplitude
        audio_data = np.array([0.1, -0.5, 0.8, -0.9, 0.3])
        mock_read.return_value = (audio_data, 16000)
        
        # Normalize
        normalized = self.processor.normalize_audio(audio_data)
        
        # Check normalization
        self.assertAlmostEqual(np.max(np.abs(normalized)), 1.0, places=5)
        self.assertLessEqual(np.max(normalized), 1.0)
        self.assertGreaterEqual(np.min(normalized), -1.0)
    
    @patch('src.core.audio_processor.librosa.resample')
    def test_resample_audio(self, mock_resample):
        """Test audio resampling to target sample rate."""
        # Mock resampling
        original_audio = np.random.randn(48000)  # 1 second at 48kHz
        resampled_audio = np.random.randn(16000)  # 1 second at 16kHz
        mock_resample.return_value = resampled_audio
        
        # Resample
        result = self.processor.resample_audio(original_audio, 48000, 16000)
        
        # Verify resampling
        mock_resample.assert_called_once_with(
            original_audio,
            orig_sr=48000,
            target_sr=16000
        )
        self.assertEqual(len(result), 16000)
    
    def test_convert_to_mono(self):
        """Test stereo to mono conversion."""
        # Create stereo audio (2 channels)
        stereo_audio = np.array([
            [0.5, 0.3, 0.7],  # Left channel
            [0.4, 0.6, 0.2]   # Right channel
        ])
        
        # Convert to mono
        mono_audio = self.processor.convert_to_mono(stereo_audio)
        
        # Verify mono conversion
        self.assertEqual(mono_audio.ndim, 1)
        self.assertEqual(len(mono_audio), 3)
        # Check averaging
        np.testing.assert_array_almost_equal(
            mono_audio,
            np.mean(stereo_audio, axis=0)
        )
    
    @patch('src.core.audio_processor.librosa.load')
    def test_extract_audio_features(self, mock_load):
        """Test audio feature extraction."""
        # Mock audio data
        mock_audio = np.random.randn(16000)
        mock_load.return_value = (mock_audio, 16000)
        
        # Extract features
        features = self.processor.extract_features(self.test_wav)
        
        # Verify features
        self.assertIn('duration', features)
        self.assertIn('sample_rate', features)
        self.assertIn('energy', features)
        self.assertIn('zero_crossing_rate', features)
        
        # Check feature values
        self.assertAlmostEqual(features['duration'], 1.0, places=1)
        self.assertEqual(features['sample_rate'], 16000)
        self.assertIsInstance(features['energy'], float)
    
    def test_batch_process_audio(self):
        """Test batch audio processing."""
        # Create multiple test files
        test_files = []
        for i in range(3):
            test_file = self.temp_dir / f'test_{i}.wav'
            test_file.write_bytes(b'RIFF' + b'\x00' * 100)
            test_files.append(test_file)
        
        with patch('src.core.audio_processor.sf.read') as mock_read:
            mock_read.return_value = (np.random.randn(16000), 16000)
            
            # Batch process
            results = self.processor.batch_process(test_files)
            
            # Verify batch processing
            self.assertEqual(len(results), 3)
            self.assertEqual(mock_read.call_count, 3)
            for result in results:
                self.assertIsInstance(result, Path)


class TestAudioFormat(unittest.TestCase):
    """Test cases for AudioFormat enum."""
    
    def test_audio_format_values(self):
        """Test AudioFormat enum values."""
        self.assertEqual(AudioFormat.WAV.value, 'wav')
        self.assertEqual(AudioFormat.MP3.value, 'mp3')
        self.assertEqual(AudioFormat.FLAC.value, 'flac')
    
    def test_format_from_extension(self):
        """Test getting format from file extension."""
        self.assertEqual(AudioFormat.from_extension('.wav'), AudioFormat.WAV)
        self.assertEqual(AudioFormat.from_extension('.mp3'), AudioFormat.MP3)
        self.assertEqual(AudioFormat.from_extension('.MP3'), AudioFormat.MP3)
    
    def test_is_supported_format(self):
        """Test checking if format is supported."""
        self.assertTrue(AudioFormat.is_supported('wav'))
        self.assertTrue(AudioFormat.is_supported('mp3'))
        self.assertFalse(AudioFormat.is_supported('xyz'))


class TestAudioMetadata(unittest.TestCase):
    """Test cases for AudioMetadata class."""
    
    def test_metadata_creation(self):
        """Test creating AudioMetadata instance."""
        metadata = AudioMetadata(
            duration=120.5,
            sample_rate=16000,
            channels=1,
            bit_depth=16,
            format='wav'
        )
        
        self.assertEqual(metadata.duration, 120.5)
        self.assertEqual(metadata.sample_rate, 16000)
        self.assertEqual(metadata.channels, 1)
    
    def test_metadata_from_file(self):
        """Test extracting metadata from audio file."""
        with patch('src.core.audio_processor.sf.info') as mock_info:
            mock_info.return_value = MagicMock(
                duration=60.0,
                samplerate=44100,
                channels=2,
                subtype='PCM_16'
            )
            
            metadata = AudioMetadata.from_file(Path('/test/audio.wav'))
            
            self.assertEqual(metadata.duration, 60.0)
            self.assertEqual(metadata.sample_rate, 44100)
            self.assertEqual(metadata.channels, 2)
    
    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = AudioMetadata(
            duration=30.0,
            sample_rate=16000,
            channels=1,
            bit_depth=16,
            format='wav'
        )
        
        metadata_dict = metadata.to_dict()
        
        self.assertIsInstance(metadata_dict, dict)
        self.assertEqual(metadata_dict['duration'], 30.0)
        self.assertEqual(metadata_dict['sample_rate'], 16000)
        self.assertEqual(metadata_dict['format'], 'wav')


if __name__ == '__main__':
    unittest.main()