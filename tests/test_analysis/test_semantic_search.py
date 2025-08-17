"""
Test Suite for Semantic Search Module

Tests for semantic search functionality, vector similarity matching,
and search result ranking in the Call Analytics System.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.analysis.semantic_search import (
    SemanticSearchEngine,
    SearchConfig,
    SearchResult,
    SimilarityScorer
)
from tests.test_analysis import create_sample_call_data, ANALYSIS_TEST_DATA_DIR


class TestSemanticSearchEngine(unittest.TestCase):
    """Test cases for SemanticSearchEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock vector store
        self.mock_vector_store = MagicMock()
        self.search_engine = SemanticSearchEngine(self.mock_vector_store)
        
        # Sample documents for testing
        self.sample_docs = [
            {
                'id': 'doc1',
                'content': 'Customer complained about billing issue and requested refund',
                'metadata': {'call_id': 'CALL_001', 'agent_id': 'agent_001', 'outcome': 'resolved'}
            },
            {
                'id': 'doc2',
                'content': 'Technical support for product installation and configuration',
                'metadata': {'call_id': 'CALL_002', 'agent_id': 'agent_002', 'outcome': 'connected'}
            },
            {
                'id': 'doc3',
                'content': 'Sales inquiry about pricing plans and discounts',
                'metadata': {'call_id': 'CALL_003', 'agent_id': 'agent_003', 'outcome': 'voicemail'}
            }
        ]
    
    def test_search_basic(self):
        """Test basic semantic search functionality."""
        # Setup mock search results
        mock_results = {
            'documents': [['Customer complained about billing issue']],
            'metadatas': [[{'call_id': 'CALL_001', 'agent_id': 'agent_001'}]],
            'distances': [[0.2]]
        }
        self.mock_vector_store.query.return_value = mock_results
        
        # Perform search
        results = self.search_engine.search(
            query='billing problem customer complaint',
            top_k=5
        )
        
        # Verify results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['call_id'], 'CALL_001')
        self.assertIn('score', results[0])
        self.assertGreaterEqual(results[0]['score'], 0)
        self.assertLessEqual(results[0]['score'], 1)
        
        # Verify search was called with correct parameters
        self.mock_vector_store.query.assert_called_once()
        call_args = self.mock_vector_store.query.call_args
        self.assertEqual(call_args[1]['n_results'], 5)
    
    def test_search_with_filters(self):
        """Test semantic search with metadata filters."""
        # Setup mock results
        mock_results = {
            'documents': [['Technical support content']],
            'metadatas': [[{'call_id': 'CALL_002', 'agent_id': 'agent_002', 'outcome': 'connected'}]],
            'distances': [[0.3]]
        }
        self.mock_vector_store.query.return_value = mock_results
        
        # Search with filters
        results = self.search_engine.search(
            query='technical issue help',
            top_k=10,
            filters={
                'agent_id': 'agent_002',
                'outcome': 'connected'
            }
        )
        
        # Verify filter was applied
        call_args = self.mock_vector_store.query.call_args
        self.assertIn('where', call_args[1])
        self.assertEqual(call_args[1]['where']['agent_id'], 'agent_002')
        self.assertEqual(call_args[1]['where']['outcome'], 'connected')
    
    def test_search_with_threshold(self):
        """Test search with similarity threshold filtering."""
        # Setup mock with multiple results of varying similarity
        mock_results = {
            'documents': [['Doc1'], ['Doc2'], ['Doc3']],
            'metadatas': [
                [{'call_id': 'CALL_001'}],
                [{'call_id': 'CALL_002'}],
                [{'call_id': 'CALL_003'}]
            ],
            'distances': [[0.1], [0.5], [0.9]]  # Different similarity scores
        }
        self.mock_vector_store.query.return_value = mock_results
        
        # Search with threshold
        results = self.search_engine.search(
            query='test query',
            top_k=10,
            threshold=0.6  # Should filter out the last result
        )
        
        # Only results with distance < 0.6 should be returned
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r['score'] > 0.4 for r in results))  # Score = 1 - distance
    
    def test_batch_search(self):
        """Test batch search functionality."""
        # Setup mock for multiple queries
        self.mock_vector_store.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        # Batch search
        queries = [
            'billing issue',
            'technical support',
            'sales inquiry'
        ]
        results = self.search_engine.batch_search(queries, top_k=5)
        
        # Should return results for each query
        self.assertEqual(len(results), 3)
        self.assertEqual(self.mock_vector_store.query.call_count, 3)
    
    def test_search_with_reranking(self):
        """Test search with result reranking."""
        # Setup mock results
        mock_results = {
            'documents': [['Doc1'], ['Doc2'], ['Doc3']],
            'metadatas': [
                [{'call_id': 'CALL_001', 'revenue': 100}],
                [{'call_id': 'CALL_002', 'revenue': 500}],
                [{'call_id': 'CALL_003', 'revenue': 200}]
            ],
            'distances': [[0.2], [0.3], [0.25]]
        }
        self.mock_vector_store.query.return_value = mock_results
        
        # Search with reranking by revenue
        results = self.search_engine.search(
            query='test',
            top_k=3,
            rerank_by='revenue'
        )
        
        # Results should be reordered by revenue (highest first)
        self.assertEqual(results[0]['call_id'], 'CALL_002')  # Highest revenue
        self.assertEqual(results[0]['revenue'], 500)
    
    def test_empty_search_results(self):
        """Test handling of empty search results."""
        # Setup mock with no results
        self.mock_vector_store.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        results = self.search_engine.search('nonexistent query')
        
        self.assertEqual(len(results), 0)
        self.assertIsInstance(results, list)


class TestSearchConfig(unittest.TestCase):
    """Test cases for SearchConfig class."""
    
    def test_default_config(self):
        """Test default search configuration."""
        config = SearchConfig()
        
        self.assertEqual(config.top_k, 10)
        self.assertEqual(config.threshold, 0.0)
        self.assertIsNone(config.filters)
        self.assertEqual(config.rerank_by, None)
        self.assertTrue(config.include_metadata)
    
    def test_custom_config(self):
        """Test custom search configuration."""
        config = SearchConfig(
            top_k=20,
            threshold=0.7,
            filters={'agent_id': 'agent_001'},
            rerank_by='relevance',
            include_metadata=False
        )
        
        self.assertEqual(config.top_k, 20)
        self.assertEqual(config.threshold, 0.7)
        self.assertEqual(config.filters['agent_id'], 'agent_001')
        self.assertEqual(config.rerank_by, 'relevance')
        self.assertFalse(config.include_metadata)
    
    def test_config_validation(self):
        """Test search configuration validation."""
        # Test invalid top_k
        with self.assertRaises(ValueError):
            SearchConfig(top_k=0)
        
        with self.assertRaises(ValueError):
            SearchConfig(top_k=-5)
        
        # Test invalid threshold
        with self.assertRaises(ValueError):
            SearchConfig(threshold=1.5)
        
        with self.assertRaises(ValueError):
            SearchConfig(threshold=-0.1)
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = SearchConfig(
            top_k=15,
            threshold=0.5,
            filters={'campaign': 'sales'}
        )
        
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['top_k'], 15)
        self.assertEqual(config_dict['threshold'], 0.5)
        self.assertEqual(config_dict['filters']['campaign'], 'sales')


class TestSearchResult(unittest.TestCase):
    """Test cases for SearchResult class."""
    
    def test_search_result_creation(self):
        """Test creating SearchResult instance."""
        result = SearchResult(
            id='doc1',
            score=0.85,
            content='Sample content',
            metadata={'call_id': 'CALL_001', 'agent_id': 'agent_001'}
        )
        
        self.assertEqual(result.id, 'doc1')
        self.assertAlmostEqual(result.score, 0.85)
        self.assertEqual(result.content, 'Sample content')
        self.assertEqual(result.metadata['call_id'], 'CALL_001')
    
    def test_search_result_comparison(self):
        """Test comparing SearchResult instances by score."""
        result1 = SearchResult('doc1', 0.9, 'Content 1', {})
        result2 = SearchResult('doc2', 0.7, 'Content 2', {})
        result3 = SearchResult('doc3', 0.9, 'Content 3', {})
        
        # Higher score should be "less than" for sorting (descending)
        self.assertLess(result1, result2)
        self.assertEqual(result1, result3)  # Equal scores
        self.assertGreater(result2, result1)
    
    def test_search_result_to_dict(self):
        """Test converting SearchResult to dictionary."""
        result = SearchResult(
            id='doc1',
            score=0.75,
            content='Test content',
            metadata={'key': 'value'}
        )
        
        result_dict = result.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict['id'], 'doc1')
        self.assertEqual(result_dict['score'], 0.75)
        self.assertEqual(result_dict['content'], 'Test content')
        self.assertEqual(result_dict['metadata']['key'], 'value')


class TestSimilarityScorer(unittest.TestCase):
    """Test cases for SimilarityScorer class."""
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        scorer = SimilarityScorer(metric='cosine')
        
        # Test identical vectors
        vec1 = np.array([1, 0, 1])
        vec2 = np.array([1, 0, 1])
        similarity = scorer.calculate(vec1, vec2)
        self.assertAlmostEqual(similarity, 1.0)
        
        # Test orthogonal vectors
        vec3 = np.array([1, 0, 0])
        vec4 = np.array([0, 1, 0])
        similarity = scorer.calculate(vec3, vec4)
        self.assertAlmostEqual(similarity, 0.0)
        
        # Test opposite vectors
        vec5 = np.array([1, 1])
        vec6 = np.array([-1, -1])
        similarity = scorer.calculate(vec5, vec6)
        self.assertAlmostEqual(similarity, -1.0)
    
    def test_euclidean_distance(self):
        """Test Euclidean distance calculation."""
        scorer = SimilarityScorer(metric='euclidean')
        
        # Test identical vectors
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([1, 2, 3])
        distance = scorer.calculate(vec1, vec2)
        self.assertAlmostEqual(distance, 0.0)
        
        # Test different vectors
        vec3 = np.array([0, 0, 0])
        vec4 = np.array([3, 4, 0])
        distance = scorer.calculate(vec3, vec4)
        self.assertAlmostEqual(distance, 5.0)  # 3-4-5 triangle
    
    def test_dot_product_similarity(self):
        """Test dot product similarity calculation."""
        scorer = SimilarityScorer(metric='dot_product')
        
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([4, 5, 6])
        similarity = scorer.calculate(vec1, vec2)
        
        expected = 1*4 + 2*5 + 3*6  # 4 + 10 + 18 = 32
        self.assertAlmostEqual(similarity, expected)
    
    def test_batch_similarity(self):
        """Test batch similarity calculation."""
        scorer = SimilarityScorer(metric='cosine')
        
        query_vec = np.array([1, 0, 1])
        corpus_vecs = np.array([
            [1, 0, 1],    # Identical
            [0, 1, 0],    # Orthogonal
            [-1, 0, -1],  # Opposite
            [1, 1, 1]     # Similar
        ])
        
        similarities = scorer.batch_calculate(query_vec, corpus_vecs)
        
        self.assertEqual(len(similarities), 4)
        self.assertAlmostEqual(similarities[0], 1.0)  # Identical
        self.assertAlmostEqual(similarities[1], 0.0)  # Orthogonal
        self.assertAlmostEqual(similarities[2], -1.0)  # Opposite
        self.assertGreater(similarities[3], 0.5)  # Similar
    
    def test_invalid_metric(self):
        """Test handling of invalid similarity metric."""
        with self.assertRaises(ValueError):
            SimilarityScorer(metric='invalid_metric')


if __name__ == '__main__':
    unittest.main()