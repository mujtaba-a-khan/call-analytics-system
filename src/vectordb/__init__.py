"""
Vector Database Package for Call Analytics System

This package provides vector database functionality for semantic search
and similarity matching using ChromaDB and other vector stores.
"""

# Import ChromaDB client
from .chroma_client import (
    ChromaDBClient,
    CollectionConfig,
    SearchResult,
    DocumentMetadata
)

# Import indexer
from .indexer import (
    VectorIndexer,
    IndexingConfig,
    IndexingResult,
    BatchIndexer
)

# Import retriever
from .retriever import (
    VectorRetriever,
    RetrievalConfig,
    RetrievalResult,
    SimilarityMetric
)

# Define package exports
__all__ = [
    # ChromaDB
    'ChromaDBClient',
    'CollectionConfig',
    'SearchResult',
    'DocumentMetadata',
    
    # Indexer
    'VectorIndexer',
    'IndexingConfig',
    'IndexingResult',
    'BatchIndexer',
    
    # Retriever
    'VectorRetriever',
    'RetrievalConfig',
    'RetrievalResult',
    'SimilarityMetric'
]

# Package version
__version__ = '1.0.0'

# Package configuration
DEFAULT_COLLECTION = 'call_records'
DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
DEFAULT_DISTANCE_METRIC = 'cosine'
DEFAULT_TOP_K = 10

def get_vector_db_info():
    """
    Get information about vector database configuration.
    
    Returns:
        Dictionary of configuration info
    """
    info = {
        'backend': 'chromadb',
        'default_collection': DEFAULT_COLLECTION,
        'default_embedding_model': DEFAULT_EMBEDDING_MODEL,
        'default_distance_metric': DEFAULT_DISTANCE_METRIC,
        'default_top_k': DEFAULT_TOP_K
    }
    
    # Check if ChromaDB is available
    try:
        import chromadb
        info['chromadb_version'] = chromadb.__version__
        info['chromadb_available'] = True
    except ImportError:
        info['chromadb_available'] = False
    
    return info