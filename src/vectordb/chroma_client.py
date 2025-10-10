"""
ChromaDB Client Module

Provides interface to ChromaDB vector database for semantic search
and document retrieval. Handles embedding generation and similarity search.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logging.warning("ChromaDB not installed. Vector search will be unavailable.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not installed. Using fallback embeddings.")

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Handles text embedding generation for vector search.
    Supports multiple backends with fallback options.
    """

    def __init__(self, config: dict):
        """
        Initialize the embedding generator.

        Args:
            config: Configuration dictionary with embedding settings
        """
        self.provider = config.get('provider', 'sentence-transformers')
        self.model_name = config.get('model_name', 'all-MiniLM-L6-v2')
        self.dimension = config.get('dimension', 384)
        self.normalize = config.get('normalize', True)
        self.batch_size = config.get('batch_size', 64)

        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the embedding model based on provider"""
        if self.provider == 'sentence-transformers' and SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        else:
            logger.info("Using hash-based fallback embeddings")
            self.provider = 'hash'  # Fallback to hash embeddings

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        if self.provider == 'sentence-transformers' and self.model:
            # Use sentence-transformers
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=self.normalize,
                batch_size=self.batch_size,
                show_progress_bar=False
            )
            return embeddings.tolist()
        else:
            # Fallback to hash-based embeddings
            return [self._hash_embedding(text) for text in texts]

    def _hash_embedding(self, text: str) -> list[float]:
        """
        Generate a simple hash-based embedding as fallback.

        Args:
            text: Text to embed

        Returns:
            Fixed-dimension embedding vector
        """
        import numpy as np

        # Create a deterministic hash
        hasher = hashlib.sha256(text.encode())
        hash_bytes = hasher.digest()

        # Convert to fixed-dimension vector
        np.random.seed(int.from_bytes(hash_bytes[:4], 'little'))
        embedding = np.random.randn(self.dimension)

        # Normalize if requested
        if self.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding.tolist()


class ChromaClient:
    """
    Client for interacting with ChromaDB vector database.
    Handles document storage, retrieval, and semantic search.
    """

    def __init__(self, config: dict):
        """
        Initialize the ChromaDB client.

        Args:
            config: Configuration dictionary with database settings
        """
        if not CHROMA_AVAILABLE:
            raise RuntimeError("ChromaDB is not installed. Please install it first.")

        self.persist_dir = Path(config.get('persist_dir', 'data/vectorstore'))
        self.collection_name = config.get('collection_name', 'call_transcripts')
        self.distance_metric = config.get('distance_metric', 'cosine')

        # Create persist directory
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize embedding generator
        embedding_config = config.get('embeddings', {})
        self.embedder = EmbeddingGenerator(embedding_config)

        # Initialize ChromaDB client
        self.client = None
        self.collection = None
        self._initialize_client()

        logger.info(f"ChromaClient initialized with collection: {self.collection_name}")

    def _initialize_client(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric}
            )

            logger.info(f"ChromaDB collection ready: {self.collection.count()} documents")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise RuntimeError(f"Could not initialize ChromaDB: {e}") from e

    def add_documents(self,
                     documents: list[str],
                     ids: list[str],
                     metadatas: list[dict[str, Any]] | None = None) -> int:
        """
        Add documents to the vector database.

        Args:
            documents: List of document texts
            ids: List of unique document IDs
            metadatas: Optional list of metadata dictionaries

        Returns:
            Number of documents added
        """
        if not documents or not ids:
            return 0

        if len(documents) != len(ids):
            raise ValueError("Documents and IDs must have the same length")

        try:
            # Generate embeddings
            embeddings = self.embedder.generate_embeddings(documents)

            # Prepare metadatas if not provided
            if metadatas is None:
                metadatas = [{} for _ in documents]

            # Add to collection
            self.collection.upsert(
                documents=documents,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )

            logger.info(f"Added {len(documents)} documents to vector database")
            return len(documents)

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise VectorDBError(f"Could not add documents: {e}") from e

    def search(self,
              query_text: str,
              top_k: int = 10,
              filter_dict: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """
        Perform semantic search in the vector database.

        Args:
            query_text: Query text for search
            top_k: Number of results to return
            filter_dict: Optional metadata filters

        Returns:
            List of search results with documents, scores, and metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedder.generate_embeddings([query_text])[0]

            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_dict,
                include=["documents", "distances", "metadatas"]
            )

            # Format results
            formatted_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i],
                        'score': 1.0 - results['distances'][0][i],  # Convert distance to similarity
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                    })

            query_preview = query_text[:50]
            logger.debug(
                "Search returned %s results for query: %s...",
                len(formatted_results),
                query_preview,
            )
            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise VectorDBError(f"Could not perform search: {e}") from e

    def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """
        Retrieve documents by their IDs.

        Args:
            ids: List of document IDs to retrieve

        Returns:
            List of documents with their metadata
        """
        try:
            results = self.collection.get(
                ids=ids,
                include=["documents", "metadatas"]
            )

            formatted_results = []
            for i in range(len(results['ids'])):
                formatted_results.append({
                    'id': results['ids'][i],
                    'document': results['documents'][i],
                    'metadata': results['metadatas'][i] if results['metadatas'] else {}
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Failed to get documents by IDs: {e}")
            raise VectorDBError(f"Could not retrieve documents: {e}") from e

    def update_metadata(self, ids: list[str], metadatas: list[dict[str, Any]]) -> bool:
        """
        Update metadata for existing documents.

        Args:
            ids: List of document IDs to update
            metadatas: New metadata dictionaries

        Returns:
            True if successful
        """
        try:
            # Get existing documents
            existing = self.collection.get(ids=ids, include=["documents"])

            if not existing['ids']:
                logger.warning("No documents found to update")
                return False

            # Update with new metadata
            self.collection.update(
                ids=ids,
                metadatas=metadatas
            )

            logger.info(f"Updated metadata for {len(ids)} documents")
            return True

        except Exception as e:
            logger.error(f"Failed to update metadata: {e}")
            return False

    def delete_documents(self, ids: list[str]) -> int:
        """
        Delete documents from the database.

        Args:
            ids: List of document IDs to delete

        Returns:
            Number of documents deleted
        """
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from vector database")
            return len(ids)

        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise VectorDBError(f"Could not delete documents: {e}") from e

    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.

        Returns:
            True if successful
        """
        try:
            # Delete and recreate collection
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric}
            )

            logger.info("Collection cleared successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the vector database.

        Returns:
            Dictionary containing database statistics
        """
        try:
            count = self.collection.count()

            return {
                'total_documents': count,
                'collection_name': self.collection_name,
                'distance_metric': self.distance_metric,
                'persist_directory': str(self.persist_dir),
                'embedding_provider': self.embedder.provider,
                'embedding_dimension': self.embedder.dimension
            }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}


class VectorDBError(Exception):
    """Custom exception for vector database errors"""
    pass
