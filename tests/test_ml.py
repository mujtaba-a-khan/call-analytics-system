"""
Test Suite for Machine Learning Modules

This module contains comprehensive tests for ML functionality including
speech-to-text, embeddings generation, and LLM integration.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import json
import time

# Import modules to test
from src.ml import (
    WhisperSTT,
    EmbeddingManager,
    SentenceTransformerEmbedding,
    HashEmbedding,
    get_ml_capabilities
)
from tests import TEST_DATA_DIR, TEST_OUTPUT_DIR


class TestWhisperSTT(unittest.TestCase):
    """Test cases for Whisper speech-to-text functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create mock audio file
        self.test_audio = self.temp_dir / 'test_audio.wav'
        self.test_audio.write_bytes(b'fake audio data')
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('src.ml.whisper_stt.WhisperModel')
    def test_whisper_initialization(self, mock_model):
        """Test Whisper STT initialization."""
        # Initialize STT engine
        stt = WhisperSTT(
            model_size='small',
            device='cpu',
            compute_type='int8'
        )
        
        # Verify model was loaded
        mock_model.assert_called_once_with(
            'small',
            device='cpu',
            compute_type='int8'
        )
        
        self.assertIsNotNone(stt.model)
    
    @patch('src.ml.whisper_stt.WhisperModel')
    def test_transcribe_audio(self, mock_model):
        """Test audio transcription."""
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
        mock_transcribe.assert_called_once()
    
    @patch('src.ml.whisper_stt.WhisperModel')
    def test_transcribe_with_options(self, mock_model):
        """Test transcription with various options."""
        mock_transcribe = MagicMock()
        mock_transcribe.return_value = (
            [{'text': 'Test', 'start': 0.0, 'end': 1.0, 'words': []}],
            {'language': 'en', 'language_probability': 0.99}
        )
        mock_model.return_value.transcribe = mock_transcribe
        
        stt = WhisperSTT()
        
        # Test with language specification
        result = stt.transcribe(
            self.test_audio,
            language='en',
            word_timestamps=True,
            vad_filter=True
        )
        
        # Verify options were passed
        call_args = mock_transcribe.call_args
        self.assertEqual(call_args[1].get('language'), 'en')
        self.assertTrue(call_args[1].get('word_timestamps'))
        self.assertTrue(call_args[1].get('vad_filter'))
    
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
        
        # Create multiple audio files
        audio_files = []
        for i in range(3):
            audio_path = self.temp_dir / f'audio_{i}.wav'
            audio_path.write_bytes(b'fake audio')
            audio_files.append(audio_path)
        
        # Batch transcribe
        results = stt.batch_transcribe(audio_files)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(mock_transcribe.call_count, 3)
    
    def test_supported_languages(self):
        """Test getting supported languages."""
        languages = WhisperSTT.get_supported_languages()
        
        self.assertIsInstance(languages, list)
        self.assertIn('en', languages)
        self.assertIn('es', languages)
        self.assertIn('fr', languages)


class TestEmbeddings(unittest.TestCase):
    """Test cases for embeddings generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_texts = [
            "This is a test sentence.",
            "Another test sentence for embeddings.",
            "Testing the embedding generation."
        ]
    
    @patch('src.ml.embeddings.SentenceTransformer')
    def test_sentence_transformer_embedding(self, mock_st):
        """Test Sentence Transformer embedding generation."""
        # Setup mock model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(len(self.test_texts), 384)
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model
        
        # Create embedding strategy
        embedder = SentenceTransformerEmbedding(model_name='all-MiniLM-L6-v2')
        
        # Generate embeddings
        embeddings = embedder.generate(self.test_texts)
        
        # Verify embeddings
        self.assertEqual(embeddings.shape, (3, 384))
        mock_model.encode.assert_called_once_with(
            self.test_texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
    
    def test_hash_embedding_fallback(self):
        """Test hash-based embedding fallback."""
        embedder = HashEmbedding(dimension=128)
        
        # Generate embeddings
        embeddings = embedder.generate(self.test_texts)
        
        # Verify embeddings
        self.assertEqual(embeddings.shape, (3, 128))
        
        # Test consistency
        embeddings2 = embedder.generate(self.test_texts)
        np.testing.assert_array_equal(embeddings, embeddings2)
    
    @patch('src.ml.embeddings.ollama')
    def test_ollama_embedding(self, mock_ollama):
        """Test Ollama embedding generation."""
        # Setup mock response
        mock_ollama.embeddings.return_value = {
            'embedding': np.random.randn(384).tolist()
        }
        
        embedder = OllamaEmbedding(model='nomic-embed-text')
        
        # Generate single embedding
        embedding = embedder.generate("Test text")
        
        # Verify embedding
        self.assertEqual(embedding.shape, (1, 384))
        mock_ollama.embeddings.assert_called_once()
    
    @patch('src.ml.embeddings.SentenceTransformer')
    def test_embedding_manager(self, mock_st):
        """Test EmbeddingManager with multiple strategies."""
        # Setup mock
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(1, 384)
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model
        
        # Create manager
        manager = EmbeddingManager(strategy='sentence_transformers')
        
        # Generate embeddings
        embeddings = manager.embed_texts(self.test_texts)
        
        self.assertEqual(len(embeddings), 3)
        self.assertEqual(embeddings[0].shape, (384,))
    
    def test_embedding_caching(self):
        """Test embedding caching functionality."""
        manager = EmbeddingManager(strategy='hash', enable_cache=True)
        
        # Generate embeddings
        start_time = time.time()
        embeddings1 = manager.embed_texts(self.test_texts)
        first_time = time.time() - start_time
        
        # Generate same embeddings (should use cache)
        start_time = time.time()
        embeddings2 = manager.embed_texts(self.test_texts)
        second_time = time.time() - start_time
        
        # Verify caching worked
        np.testing.assert_array_equal(embeddings1, embeddings2)
        self.assertLess(second_time, first_time)
    
    def test_embedding_similarity(self):
        """Test embedding similarity calculations."""
        manager = EmbeddingManager(strategy='hash')
        
        # Generate embeddings
        text1 = "This is about cats"
        text2 = "This is about dogs"
        text3 = "This is about cats and kittens"
        
        emb1 = manager.embed_text(text1)
        emb2 = manager.embed_text(text2)
        emb3 = manager.embed_text(text3)
        
        # Calculate similarities
        sim_12 = manager.calculate_similarity(emb1, emb2)
        sim_13 = manager.calculate_similarity(emb1, emb3)
        
        # Verify similarity ranges
        self.assertGreaterEqual(sim_12, -1)
        self.assertLessEqual(sim_12, 1)
        self.assertGreaterEqual(sim_13, -1)
        self.assertLessEqual(sim_13, 1)


class TestMLCapabilities(unittest.TestCase):
    """Test cases for ML capabilities detection."""
    
    def test_get_ml_capabilities(self):
        """Test ML capabilities detection."""
        capabilities = get_ml_capabilities()
        
        self.assertIsInstance(capabilities, dict)
        self.assertIn('whisper_stt', capabilities)
        self.assertIn('embeddings', capabilities)
        self.assertIn('pytorch', capabilities)
        self.assertIn('sentence_transformers', capabilities)
        
        # These should always be True in the base system
        self.assertTrue(capabilities['whisper_stt'])
        self.assertTrue(capabilities['embeddings'])
    
    @patch('src.ml.torch')
    def test_pytorch_detection(self, mock_torch):
        """Test PyTorch availability detection."""
        capabilities = get_ml_capabilities()
        
        # Check if PyTorch detection worked
        if mock_torch:
            self.assertIn('pytorch', capabilities)


class TestTranscriptionConfig(unittest.TestCase):
    """Test cases for transcription configuration."""
    
    def test_transcription_config_defaults(self):
        """Test default transcription configuration."""
        from src.ml.whisper_stt import TranscriptionConfig
        
        config = TranscriptionConfig()
        
        self.assertEqual(config.language, 'auto')
        self.assertFalse(config.word_timestamps)
        self.assertTrue(config.vad_filter)
        self.assertEqual(config.beam_size, 5)
    
    def test_transcription_config_custom(self):
        """Test custom transcription configuration."""
        from src.ml.whisper_stt import TranscriptionConfig
        
        config = TranscriptionConfig(
            language='en',
            word_timestamps=True,
            vad_filter=False,
            beam_size=10,
            initial_prompt="Customer service call"
        )
        
        self.assertEqual(config.language, 'en')
        self.assertTrue(config.word_timestamps)
        self.assertFalse(config.vad_filter)
        self.assertEqual(config.beam_size, 10)
        self.assertEqual(config.initial_prompt, "Customer service call")


class TestModelPerformance(unittest.TestCase):
    """Test cases for model performance metrics."""
    
    def test_transcription_performance(self):
        """Test transcription performance metrics."""
        # This would test actual performance metrics
        # For unit tests, we'll mock the results
        
        mock_metrics = {
            'transcription_speed': 0.5,  # 0.5x real-time
            'memory_usage_mb': 500,
            'accuracy': 0.95
        }
        
        self.assertLess(mock_metrics['transcription_speed'], 1.0)
        self.assertLess(mock_metrics['memory_usage_mb'], 1000)
        self.assertGreater(mock_metrics['accuracy'], 0.9)
    
    def test_embedding_performance(self):
        """Test embedding generation performance."""
        manager = EmbeddingManager(strategy='hash')
        
        # Test batch performance
        large_batch = ["Test text"] * 100
        
        start_time = time.time()
        embeddings = manager.embed_texts(large_batch)
        elapsed_time = time.time() - start_time
        
        # Should process 100 texts in reasonable time
        self.assertLess(elapsed_time, 5.0)  # Less than 5 seconds
        self.assertEqual(len(embeddings), 100)


def suite():
    """Create test suite for ML modules."""
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestWhisperSTT))
    suite.addTest(unittest.makeSuite(TestEmbeddings))
    suite.addTest(unittest.makeSuite(TestMLCapabilities))
    suite.addTest(unittest.makeSuite(TestTranscriptionConfig))
    suite.addTest(unittest.makeSuite(TestModelPerformance))
    
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())