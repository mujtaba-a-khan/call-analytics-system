"""
Test Suite for Embeddings Module

Tests for embedding generation strategies including Sentence Transformers,
Ollama, and hash-based embeddings in the Call Analytics System.
"""

import unittest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

from src.ml.embeddings import (
    EmbeddingStrategy,
    SentenceTransformerEmbedding,
    OllamaEmbedding,
    HashEmbedding,
    EmbeddingManager
)
from tests.test_ml import ML_TEST_DATA_DIR, ML_TEST_OUTPUT_DIR, TEST_EMBEDDING_CONFIG


class TestSentenceTransformerEmbedding(unittest.TestCase):
    """Test cases for Sentence Transformer embedding strategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_texts = [
            "This is a test sentence.",
            "Another test sentence for embeddings.",
            "Testing the embedding generation system."
        ]
        self.single_text = "Single text for testing"
    
    @patch('src.ml.embeddings.SentenceTransformer')
    def test_initialization(self, mock_st):
        """Test Sentence Transformer initialization."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model
        
        embedder = SentenceTransformerEmbedding(model_name='all-MiniLM-L6-v2')
        
        mock_st.assert_called_once_with(
            'all-MiniLM-L6-v2',
            device='cpu',
            cache_folder=None
        )
        self.assertEqual(embedder.dimension, 384)
    
    @patch('src.ml.embeddings.SentenceTransformer')
    def test_generate_embeddings_batch(self, mock_st):
        """Test generating embeddings for multiple texts."""
        mock_model = MagicMock()
        mock_embeddings = np.random.randn(len(self.test_texts), 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model
        
        embedder = SentenceTransformerEmbedding()
        embeddings = embedder.generate(self.test_texts)
        
        self.assertEqual(embeddings.shape, (3, 384))
        mock_model.encode.assert_called_once_with(
            self.test_texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
    
    @patch('src.ml.embeddings.SentenceTransformer')
    def test_generate_embedding_single(self, mock_st):
        """Test generating embedding for single text."""
        mock_model = MagicMock()
        mock_embeddings = np.random.randn(1, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model
        
        embedder = SentenceTransformerEmbedding()
        embedding = embedder.generate(self.single_text)
        
        self.assertEqual(embedding.shape, (1, 384))
    
    @patch('src.ml.embeddings.SentenceTransformer')
    def test_custom_device_configuration(self, mock_st):
        """Test using custom device configuration."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model
        
        embedder = SentenceTransformerEmbedding(
            model_name='all-mpnet-base-v2',
            device='cuda'
        )
        
        mock_st.assert_called_with(
            'all-mpnet-base-v2',
            device='cuda',
            cache_folder=None
        )
    
    @patch('src.ml.embeddings.SentenceTransformer')
    def test_get_dimension(self, mock_st):
        """Test getting embedding dimension."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_st.return_value = mock_model
        
        embedder = SentenceTransformerEmbedding()
        dimension = embedder.get_dimension()
        
        self.assertEqual(dimension, 768)


class TestHashEmbedding(unittest.TestCase):
    """Test cases for hash-based embedding fallback."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_texts = [
            "Test text one",
            "Test text two",
            "Test text three"
        ]
    
    def test_hash_embedding_generation(self):
        """Test generating hash-based embeddings."""
        embedder = HashEmbedding(dimension=128)
        embeddings = embedder.generate(self.test_texts)
        
        self.assertEqual(embeddings.shape, (3, 128))
        # Check values are in expected range
        self.assertTrue(np.all(embeddings >= -1))
        self.assertTrue(np.all(embeddings <= 1))
    
    def test_hash_embedding_consistency(self):
        """Test that hash embeddings are consistent."""
        embedder = HashEmbedding(dimension=256)
        
        # Generate embeddings twice
        embeddings1 = embedder.generate(self.test_texts)
        embeddings2 = embedder.generate(self.test_texts)
        
        # Should be identical
        np.testing.assert_array_equal(embeddings1, embeddings2)
    
    def test_hash_embedding_single_text(self):
        """Test hash embedding for single text."""
        embedder = HashEmbedding(dimension=64)
        embedding = embedder.generate("Single text")
        
        self.assertEqual(embedding.shape, (1, 64))
    
    def test_hash_embedding_different_dimensions(self):
        """Test hash embeddings with different dimensions."""
        dimensions = [32, 64, 128, 256, 512]
        
        for dim in dimensions:
            embedder = HashEmbedding(dimension=dim)
            embeddings = embedder.generate(self.test_texts)
            self.assertEqual(embeddings.shape, (3, dim))
    
    def test_get_dimension(self):
        """Test getting hash embedding dimension."""
        embedder = HashEmbedding(dimension=200)
        self.assertEqual(embedder.get_dimension(), 200)


class TestOllamaEmbedding(unittest.TestCase):
    """Test cases for Ollama embedding strategy."""
    
    @patch('src.ml.embeddings.ollama')
    def test_ollama_initialization(self, mock_ollama):
        """Test Ollama embedding initialization."""
        embedder = OllamaEmbedding(model='nomic-embed-text')
        
        self.assertEqual(embedder.model, 'nomic-embed-text')
        self.assertEqual(embedder.dimension, 768)  # Default dimension
    
    @patch('src.ml.embeddings.ollama')
    def test_generate_ollama_embedding(self, mock_ollama):
        """Test generating embeddings with Ollama."""
        mock_embedding = np.random.randn(384).tolist()
        mock_ollama.embeddings.return_value = {
            'embedding': mock_embedding
        }
        
        embedder = OllamaEmbedding(model='nomic-embed-text', dimension=384)
        embedding = embedder.generate("Test text")
        
        self.assertEqual(embedding.shape, (1, 384))
        mock_ollama.embeddings.assert_called_once()
    
    @patch('src.ml.embeddings.ollama')
    def test_generate_ollama_batch(self, mock_ollama):
        """Test batch embedding generation with Ollama."""
        mock_embedding = np.random.randn(384).tolist()
        mock_ollama.embeddings.return_value = {
            'embedding': mock_embedding
        }
        
        embedder = OllamaEmbedding(dimension=384)
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = embedder.generate(texts)
        
        self.assertEqual(embeddings.shape, (3, 384))
        self.assertEqual(mock_ollama.embeddings.call_count, 3)
    
    @patch('src.ml.embeddings.ollama')
    def test_ollama_error_handling(self, mock_ollama):
        """Test error handling in Ollama embeddings."""
        mock_ollama.embeddings.side_effect = Exception("Ollama error")
        
        embedder = OllamaEmbedding()
        
        with self.assertRaises(Exception) as context:
            embedder.generate("Test text")
        
        self.assertIn("Ollama error", str(context.exception))


class TestEmbeddingManager(unittest.TestCase):
    """Test cases for EmbeddingManager class."""
    
    def test_manager_with_hash_strategy(self):
        """Test EmbeddingManager with hash strategy."""
        manager = EmbeddingManager(strategy='hash', dimension=128)
        
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = manager.embed_texts(texts)
        
        self.assertEqual(len(embeddings), 3)
        self.assertEqual(embeddings[0].shape, (128,))
    
    @patch('src.ml.embeddings.SentenceTransformer')
    def test_manager_with_sentence_transformer(self, mock_st):
        """Test EmbeddingManager with Sentence Transformer strategy."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(1, 384)
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model
        
        manager = EmbeddingManager(strategy='sentence_transformers')
        
        embedding = manager.embed_text("Test text")
        
        self.assertEqual(embedding.shape, (384,))
    
    def test_manager_caching(self):
        """Test caching functionality in EmbeddingManager."""
        manager = EmbeddingManager(strategy='hash', enable_cache=True)
        
        texts = ["Text 1", "Text 2", "Text 3"]
        
        # First embedding generation
        start_time = time.time()
        embeddings1 = manager.embed_texts(texts)
        first_time = time.time() - start_time
        
        # Second embedding generation (should use cache)
        start_time = time.time()
        embeddings2 = manager.embed_texts(texts)
        second_time = time.time() - start_time
        
        # Verify caching worked
        np.testing.assert_array_equal(embeddings1, embeddings2)
        # Second call should be faster due to caching
        self.assertLess(second_time, first_time * 0.5)
    
    def test_similarity_calculation(self):
        """Test similarity calculation between embeddings."""
        manager = EmbeddingManager(strategy='hash')
        
        emb1 = manager.embed_text("Hello world")
        emb2 = manager.embed_text("Hello world")
        emb3 = manager.embed_text("Goodbye world")
        
        # Same text should have similarity of 1
        sim_same = manager.calculate_similarity(emb1, emb2)
        self.assertAlmostEqual(sim_same, 1.0, places=5)
        
        # Different texts should have lower similarity
        sim_diff = manager.calculate_similarity(emb1, emb3)
        self.assertLess(sim_diff, 1.0)
        self.assertGreater(sim_diff, -1.0)
    
    def test_batch_similarity(self):
        """Test batch similarity calculation."""
        manager = EmbeddingManager(strategy='hash', dimension=64)
        
        query_embedding = manager.embed_text("Query text")
        corpus_embeddings = manager.embed_texts([
            "Similar text",
            "Different content",
            "Query text",
            "Random words"
        ])
        
        similarities = manager.batch_similarity(query_embedding, corpus_embeddings)
        
        self.assertEqual(len(similarities), 4)
        # Exact match should have highest similarity
        self.assertEqual(np.argmax(similarities), 2)
    
    def test_invalid_strategy(self):
        """Test handling of invalid embedding strategy."""
        with self.assertRaises(ValueError):
            EmbeddingManager(strategy='invalid_strategy')
    
    def test_get_model_info(self):
        """Test getting model information."""
        manager = EmbeddingManager(strategy='hash', dimension=256)
        info = manager.get_model_info()
        
        self.assertIn('strategy', info)
        self.assertIn('dimension', info)
        self.assertEqual(info['strategy'], 'hash')
        self.assertEqual(info['dimension'], 256)


class TestEmbeddingPerformance(unittest.TestCase):
    """Test cases for embedding performance metrics."""
    
    def test_embedding_speed(self):
        """Test embedding generation speed."""
        manager = EmbeddingManager(strategy='hash', dimension=128)
        
        # Generate embeddings for large batch
        large_batch = ["Test text"] * 1000
        
        start_time = time.time()
        embeddings = manager.embed_texts(large_batch)
        elapsed_time = time.time() - start_time
        
        # Should process 1000 texts quickly
        self.assertLess(elapsed_time, 5.0)
        self.assertEqual(len(embeddings), 1000)
    
    def test_memory_efficiency(self):
        """Test memory efficiency of embeddings."""
        manager = EmbeddingManager(strategy='hash', dimension=128)
        
        # Generate embeddings
        texts = ["Text"] * 100
        embeddings = manager.embed_texts(texts)
        
        # Check memory usage (128 dimensions * 100 texts * 4 bytes per float32)
        expected_size = 128 * 100 * 4
        actual_size = embeddings.nbytes
        
        # Should be close to expected size
        self.assertLess(actual_size, expected_size * 1.5)


if __name__ == '__main__':
    unittest.main()