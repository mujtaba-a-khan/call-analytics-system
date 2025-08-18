"""
Vector Database Package for Call Analytics System

This package handles vector storage and retrieval for semantic search.
"""

__version__ = '1.0.0'

__all__ = [
    'ChromaDBClient',
    'VectorIndexer', 
    'VectorRetriever'
]

def __getattr__(name):
    """Lazy loading for vector database modules"""
    if name == 'ChromaDBClient':
        from .chroma_client import ChromaDBClient
        return ChromaDBClient
    elif name == 'VectorIndexer':
        from .indexer import VectorIndexer
        return VectorIndexer
    elif name == 'VectorRetriever':
        from .retriever import VectorRetriever
        return VectorRetriever
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")