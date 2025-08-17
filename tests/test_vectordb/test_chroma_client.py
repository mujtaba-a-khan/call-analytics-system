"""
Test Suite for ChromaDB Client Module

Tests for ChromaDB vector database client operations including
collection management, document operations, and search functionality.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.vectordb.chroma_client import (
    ChromaDBClient,
    CollectionConfig,
    SearchResult,
    DocumentMetadata
)
from tests.test_vectordb import (
    VECTORDB_TEST_DB_DIR,
    TEST_COLLECTION,
    TEST_COLLECTION_BACKUP
)


class TestChromaDBClient(unittest.TestCase):
    """Test cases for ChromaDBClient class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.client = ChromaDBClient(persist_directory=str(self.temp_dir))
        
        # Sample documents for testing
        self.sample_docs = [
            {
                'id': 'doc1',
                'text': 'Customer complaint about billing',
                'metadata': {'call_id': 'CALL_001', 'agent_id': 'agent_001'}
            },
            {
                'id': 'doc2',
                'text': 'Technical support request',
                'metadata': {'call_id': 'CALL_002', 'agent_id': 'agent_002'}
            },
            {
                'id': 'doc3',
                'text': 'Sales inquiry about pricing',
                'metadata': {'call_id': 'CALL_003', 'agent_id': 'agent_003'}
            }
        ]
        
        # Sample embeddings
        self.sample_embeddings = [
            np.random.rand(384).tolist(),
            np.random.rand(384).tolist(),
            np.random.rand(384).tolist()
        ]
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('src.vectordb.chroma_client.chromadb')
    def test_client_initialization(self, mock_chromadb):
        """Test ChromaDB client initialization."""
        mock_client = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client
        
        client = ChromaDBClient(persist_directory='/test/path')
        
        mock_chromadb.PersistentClient.assert_called_once_with(path='/test/path')
        self.assertIsNotNone(client.client)
    
    def test_create_collection(self):
        """Test creating a collection."""
        collection_name = TEST_COLLECTION
        
        # Create collection
        collection = self.client.create_collection(
            name=collection_name,
            metadata={'description': 'Test collection'}
        )
        
        self.assertIsNotNone(collection)
        
        # Verify collection exists
        collections = self.client.list_collections()
        self.assertIn(collection_name, collections)
    
    def test_delete_collection(self):
        """Test deleting a collection."""
        collection_name = TEST_COLLECTION
        
        # Create and then delete collection
        self.client.create_collection(collection_name)
        result = self.client.delete_collection(collection_name)
        
        self.assertTrue(result)
        
        # Verify collection is deleted
        collections = self.client.list_collections()
        self.assertNotIn(collection_name, collections)
    
    def test_add_documents(self):
        """Test adding documents to collection."""
        collection_name = TEST_COLLECTION
        self.client.create_collection(collection_name)
        
        # Add documents
        result = self.client.add_documents(
            collection_name=collection_name,
            documents=[doc['text'] for doc in self.sample_docs],
            embeddings=self.sample_embeddings,
            metadatas=[doc['metadata'] for doc in self.sample_docs],
            ids=[doc['id'] for doc in self.sample_docs]
        )
        
        self.assertTrue(result)
        
        # Verify documents were added
        collection_size = self.client.get_collection_size(collection_name)
        self.assertEqual(collection_size, 3)
    
    def test_search_documents(self):
        """Test searching documents in collection."""
        collection_name = TEST_COLLECTION
        self.client.create_collection(collection_name)
        
        # Add documents first
        self.client.add_documents(
            collection_name=collection_name,
            documents=[doc['text'] for doc in self.sample_docs],
            embeddings=self.sample_embeddings,
            metadatas=[doc['metadata'] for doc in self.sample_docs],
            ids=[doc['id'] for doc in self.sample_docs]
        )
        
        # Search with query embedding
        query_embedding = np.random.rand(384).tolist()
        results = self.client.search(
            collection_name=collection_name,
            query_embeddings=[query_embedding],
            n_results=2
        )
        
        self.assertIsNotNone(results)
        self.assertIn('documents', results)
        self.assertIn('metadatas', results)
        self.assertIn('distances', results)
    
    def test_search_with_filter(self):
        """Test searching with metadata filters."""
        collection_name = TEST_COLLECTION
        self.client.create_collection(collection_name)
        
        # Add documents
        self.client.add_documents(
            collection_name=collection_name,
            documents=[doc['text'] for doc in self.sample_docs],
            embeddings=self.sample_embeddings,
            metadatas=[doc['metadata'] for doc in self.sample_docs],
            ids=[doc['id'] for doc in self.sample_docs]
        )
        
        # Search with filter
        query_embedding = np.random.rand(384).tolist()
        results = self.client.search(
            collection_name=collection_name,
            query_embeddings=[query_embedding],
            n_results=5,
            where={'agent_id': 'agent_001'}
        )
        
        self.assertIsNotNone(results)
        # Should only return documents matching the filter
        if results['metadatas'][0]:
            for metadata in results['metadatas'][0]:
                self.assertEqual(metadata.get('agent_id'), 'agent_001')
    
    def test_update_documents(self):
        """Test updating documents in collection."""
        collection_name = TEST_COLLECTION
        self.client.create_collection(collection_name)
        
        # Add initial documents
        self.client.add_documents(
            collection_name=collection_name,
            documents=[self.sample_docs[0]['text']],
            embeddings=[self.sample_embeddings[0]],
            metadatas=[self.sample_docs[0]['metadata']],
            ids=[self.sample_docs[0]['id']]
        )
        
        # Update document
        updated_metadata = {'call_id': 'CALL_001', 'agent_id': 'agent_999', 'updated': True}
        result = self.client.update_documents(
            collection_name=collection_name,
            ids=['doc1'],
            metadatas=[updated_metadata]
        )
        
        self.assertTrue(result)
    
    def test_delete_documents(self):
        """Test deleting documents from collection."""
        collection_name = TEST_COLLECTION
        self.client.create_collection(collection_name)
        
        # Add documents
        self.client.add_documents(
            collection_name=collection_name,
            documents=[doc['text'] for doc in self.sample_docs],
            embeddings=self.sample_embeddings,
            metadatas=[doc['metadata'] for doc in self.sample_docs],
            ids=[doc['id'] for doc in self.sample_docs]
        )
        
        # Delete one document
        result = self.client.delete_documents(
            collection_name=collection_name,
            ids=['doc1']
        )
        
        self.assertTrue(result)
        
        # Verify deletion
        collection_size = self.client.get_collection_size(collection_name)
        self.assertEqual(collection_size, 2)
    
    def test_get_documents(self):
        """Test retrieving documents by IDs."""
        collection_name = TEST_COLLECTION
        self.client.create_collection(collection_name)
        
        # Add documents
        self.client.add_documents(
            collection_name=collection_name,
            documents=[doc['text'] for doc in self.sample_docs],
            embeddings=self.sample_embeddings,
            metadatas=[doc['metadata'] for doc in self.sample_docs],
            ids=[doc['id'] for doc in self.sample_docs]
        )
        
        # Get specific documents
        results = self.client.get_documents(
            collection_name=collection_name,
            ids=['doc1', 'doc2']
        )
        
        self.assertIsNotNone(results)
        self.assertEqual(len(results['ids']), 2)
        self.assertIn('doc1', results['ids'])
        self.assertIn('doc2', results['ids'])
    
    def test_collection_exists(self):
        """Test checking if collection exists."""
        collection_name = TEST_COLLECTION
        
        # Check non-existent collection
        exists = self.client.collection_exists(collection_name)
        self.assertFalse(exists)
        
        # Create collection
        self.client.create_collection(collection_name)
        
        # Check existing collection
        exists = self.client.collection_exists(collection_name)
        self.assertTrue(exists)


class TestCollectionConfig(unittest.TestCase):
    """Test cases for CollectionConfig class."""
    
    def test_config_creation(self):
        """Test creating CollectionConfig instance."""
        config = CollectionConfig(
            name='test_collection',
            embedding_function='default',
            metadata={'description': 'Test collection'},
            distance_metric='cosine'
        )
        
        self.assertEqual(config.name, 'test_collection')
        self.assertEqual(config.embedding_function, 'default')
        self.assertEqual(config.distance_metric, 'cosine')
        self.assertEqual(config.metadata['description'], 'Test collection')
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid distance metric
        with self.assertRaises(ValueError):
            CollectionConfig(
                name='test',
                distance_metric='invalid_metric'
            )
        
        # Test empty name
        with self.assertRaises(ValueError):
            CollectionConfig(name='')
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = CollectionConfig(
            name='test_collection',
            metadata={'key': 'value'}
        )
        
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['name'], 'test_collection')
        self.assertIn('metadata', config_dict)


class TestSearchResult(unittest.TestCase):
    """Test cases for SearchResult class."""
    
    def test_search_result_creation(self):
        """Test creating SearchResult instance."""
        result = SearchResult(
            id='doc1',
            score=0.95,
            document='Test document',
            metadata={'key': 'value'},
            embedding=[0.1, 0.2, 0.3]
        )
        
        self.assertEqual(result.id, 'doc1')
        self.assertAlmostEqual(result.score, 0.95)
        self.assertEqual(result.document, 'Test document')
        self.assertEqual(result.metadata['key'], 'value')
        self.assertEqual(len(result.embedding), 3)
    
    def test_search_result_from_chroma(self):
        """Test creating SearchResult from ChromaDB output."""
        chroma_result = {
            'ids': [['doc1', 'doc2']],
            'documents': [['Document 1', 'Document 2']],
            'metadatas': [[{'key1': 'value1'}, {'key2': 'value2'}]],
            'distances': [[0.1, 0.3]],
            'embeddings': None
        }
        
        results = SearchResult.from_chroma_results(chroma_result, 0)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].id, 'doc1')
        self.assertEqual(results[0].document, 'Document 1')
        self.assertAlmostEqual(results[0].score, 0.9)  # 1 - distance
    
    def test_search_result_ranking(self):
        """Test ranking search results by score."""
        results = [
            SearchResult('doc1', 0.7, 'Doc 1', {}),
            SearchResult('doc2', 0.9, 'Doc 2', {}),
            SearchResult('doc3', 0.8, 'Doc 3', {})
        ]
        
        # Sort by score (descending)
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        
        self.assertEqual(sorted_results[0].id, 'doc2')
        self.assertEqual(sorted_results[1].id, 'doc3')
        self.assertEqual(sorted_results[2].id, 'doc1')


class TestDocumentMetadata(unittest.TestCase):
    """Test cases for DocumentMetadata class."""
    
    def test_metadata_creation(self):
        """Test creating DocumentMetadata instance."""
        metadata = DocumentMetadata(
            call_id='CALL_001',
            agent_id='agent_001',
            timestamp='2024-01-01T10:00:00',
            campaign='sales',
            outcome='connected',
            duration=300,
            custom_fields={'key': 'value'}
        )
        
        self.assertEqual(metadata.call_id, 'CALL_001')
        self.assertEqual(metadata.agent_id, 'agent_001')
        self.assertEqual(metadata.duration, 300)
        self.assertEqual(metadata.custom_fields['key'], 'value')
    
    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = DocumentMetadata(
            call_id='CALL_001',
            agent_id='agent_001'
        )
        
        metadata_dict = metadata.to_dict()
        
        self.assertIsInstance(metadata_dict, dict)
        self.assertEqual(metadata_dict['call_id'], 'CALL_001')
        self.assertEqual(metadata_dict['agent_id'], 'agent_001')
    
    def test_metadata_filtering(self):
        """Test metadata filtering logic."""
        metadata = DocumentMetadata(
            call_id='CALL_001',
            agent_id='agent_001',
            campaign='sales'
        )
        
        # Test matching filter
        filter_dict = {'agent_id': 'agent_001', 'campaign': 'sales'}
        self.assertTrue(metadata.matches_filter(filter_dict))
        
        # Test non-matching filter
        filter_dict = {'agent_id': 'agent_002'}
        self.assertFalse(metadata.matches_filter(filter_dict))


if __name__ == '__main__':
    unittest.main()