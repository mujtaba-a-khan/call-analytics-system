"""
Text Embeddings Module

Generates text embeddings for semantic search and similarity comparison.
Supports multiple embedding backends with fallback options.
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""

    @abstractmethod
    def generate(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        pass


class SentenceTransformerProvider(EmbeddingProvider):
    """
    Embedding provider using Sentence Transformers.
    High quality semantic embeddings for local use.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize Sentence Transformer provider.

        Args:
            model_name: Name of the sentence transformer model
        """
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.available = True
            logger.info(f"SentenceTransformer initialized with model: {model_name}")
        except ImportError:
            self.model = None
            self.dimension = 384  # Default dimension
            self.available = False
            logger.warning("sentence-transformers not installed")

    def generate(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for texts.

        Args:
            texts: List of texts to embed

        Returns:
            Array of embeddings
        """
        if not self.available:
            raise RuntimeError("SentenceTransformer not available")

        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        return embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension


class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider using Ollama's embedding models.
    """

    def __init__(self, model: str = "nomic-embed-text", endpoint: str = "http://localhost:11434"):
        """
        Initialize Ollama embedding provider.

        Args:
            model: Ollama embedding model name
            endpoint: Ollama API endpoint
        """
        self.model = model
        self.endpoint = endpoint
        self.dimension = self._get_dimension()
        self.available = self._test_connection()

        if self.available:
            logger.info(f"Ollama embeddings initialized with model: {model}")
        else:
            logger.warning("Ollama embedding service not available")

    def _test_connection(self) -> bool:
        """Test if Ollama service is available"""
        try:
            import requests
        except ImportError:
            return False

        try:
            response = requests.get(f"{self.endpoint}/api/tags", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _get_dimension(self) -> int:
        """Get embedding dimension for the model"""
        # Default dimensions for known models
        dimensions = {"nomic-embed-text": 768, "mxbai-embed-large": 1024, "all-minilm": 384}
        return dimensions.get(self.model, 384)

    def generate(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings using Ollama.

        Args:
            texts: List of texts to embed

        Returns:
            Array of embeddings
        """
        if not self.available:
            raise RuntimeError("Ollama embedding service not available")

        import requests

        embeddings = []

        for text in texts:
            try:
                response = requests.post(
                    f"{self.endpoint}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=10,
                )

                if response.status_code == 200:
                    data = response.json()
                    embedding = data.get("embedding", [])
                    embeddings.append(embedding)
                else:
                    # Fallback to zero vector
                    embeddings.append([0.0] * self.dimension)

            except Exception as e:
                logger.error(f"Error generating Ollama embedding: {e}")
                embeddings.append([0.0] * self.dimension)

        return np.array(embeddings)

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension


class HashEmbeddingProvider(EmbeddingProvider):
    """
    Simple hash-based embedding provider as fallback.
    Deterministic but not semantic.
    """

    def __init__(self, dimension: int = 384, seed: int = 42):
        """
        Initialize hash embedding provider.

        Args:
            dimension: Embedding dimension
            seed: Random seed for reproducibility
        """
        self.dimension = dimension
        self.seed = seed
        self.available = True
        logger.info(f"Hash embeddings initialized with dimension: {dimension}")

    def generate(self, texts: list[str]) -> np.ndarray:
        """
        Generate hash-based embeddings.

        Args:
            texts: List of texts to embed

        Returns:
            Array of embeddings
        """
        embeddings = []

        for text in texts:
            # Create deterministic hash
            text_hash = hashlib.sha256(f"{self.seed}:{text}".encode()).digest()

            # Convert hash to embedding vector
            np.random.seed(int.from_bytes(text_hash[:4], "little"))
            embedding = np.random.randn(self.dimension)

            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)

        return np.array(embeddings)

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension


class EmbeddingManager:
    """
    Manages text embeddings with multiple provider support and caching.
    """

    def __init__(self, config: dict):
        """
        Initialize the embedding manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.provider_name = config.get("provider", "sentence-transformers")
        self.cache = {}
        self.cache_enabled = config.get("cache_embeddings", True)

        # Initialize provider
        self.provider = self._initialize_provider()

        logger.info(f"EmbeddingManager initialized with provider: {self.provider_name}")

    def _initialize_provider(self) -> EmbeddingProvider:
        """
        Initialize the appropriate embedding provider.

        Returns:
            Embedding provider instance
        """
        if self.provider_name == "sentence-transformers":
            try:
                provider = SentenceTransformerProvider(
                    model_name=self.config.get("model_name", "all-MiniLM-L6-v2")
                )
                if provider.available:
                    return provider
            except Exception as e:
                logger.error(f"Failed to initialize SentenceTransformer: {e}")

        elif self.provider_name == "ollama":
            try:
                provider = OllamaEmbeddingProvider(
                    model=self.config.get("ollama_model", "nomic-embed-text"),
                    endpoint=self.config.get("ollama_endpoint", "http://localhost:11434"),
                )
                if provider.available:
                    return provider
            except Exception as e:
                logger.error(f"Failed to initialize Ollama embeddings: {e}")

        # Fallback to hash embeddings
        logger.info("Using hash embeddings as fallback")
        return HashEmbeddingProvider(
            dimension=self.config.get("dimension", 384), seed=self.config.get("seed", 42)
        )

    def generate_embeddings(
        self,
        texts: list[str],
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for texts with caching support.

        Args:
            texts: List of texts to embed
            use_cache: Whether to use cached embeddings

        Returns:
            Array of embeddings
        """
        if not texts:
            return np.array([])

        embeddings = []
        texts_to_generate = []
        text_indices = []

        # Check cache
        if use_cache and self.cache_enabled:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self.cache:
                    embeddings.append(self.cache[cache_key])
                else:
                    texts_to_generate.append(text)
                    text_indices.append(i)
        else:
            texts_to_generate = texts
            text_indices = list(range(len(texts)))

        # Generate new embeddings
        if texts_to_generate:
            new_embeddings = self.provider.generate(texts_to_generate)

            # Add to cache
            if self.cache_enabled:
                for text, embedding in zip(texts_to_generate, new_embeddings, strict=False):
                    cache_key = self._get_cache_key(text)
                    self.cache[cache_key] = embedding

            # Merge with cached embeddings
            if use_cache and self.cache_enabled:
                result = np.zeros((len(texts), self.provider.get_dimension()))

                # Fill in cached embeddings
                cache_idx = 0
                for i, _text in enumerate(texts):
                    if i not in text_indices:
                        result[i] = embeddings[cache_idx]
                        cache_idx += 1

                # Fill in new embeddings
                for i, idx in enumerate(text_indices):
                    result[idx] = new_embeddings[i]

                return result
            else:
                return new_embeddings
        else:
            return np.array(embeddings)

    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key for text.

        Args:
            text: Input text

        Returns:
            Cache key
        """
        return hashlib.sha256(f"{self.provider_name}:{text}".encode()).hexdigest()

    def compute_similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
        metric: str = "cosine",
    ) -> np.ndarray:
        """
        Compute similarity between embedding sets.

        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            metric: Similarity metric ('cosine', 'euclidean', 'dot')

        Returns:
            Similarity matrix
        """
        if metric == "cosine":
            from sklearn.metrics.pairwise import cosine_similarity

            return cosine_similarity(embeddings1, embeddings2)

        elif metric == "euclidean":
            from sklearn.metrics.pairwise import euclidean_distances

            # Convert distance to similarity
            distances = euclidean_distances(embeddings1, embeddings2)
            return 1 / (1 + distances)

        elif metric == "dot":
            return np.dot(embeddings1, embeddings2.T)

        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def find_most_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        top_k: int = 5,
        metric: str = "cosine",
    ) -> list[tuple[int, float]]:
        """
        Find most similar embeddings to query.

        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: Candidate embedding matrix
            top_k: Number of results to return
            metric: Similarity metric

        Returns:
            List of (index, score) tuples
        """
        # Reshape query if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Compute similarities
        similarities = self.compute_similarity(
            query_embedding, candidate_embeddings, metric
        ).flatten()

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return with scores
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]

        return results

    def clear_cache(self):
        """Clear the embedding cache"""
        self.cache.clear()
        logger.info("Embedding cache cleared")

    def get_info(self) -> dict[str, Any]:
        """
        Get information about the embedding manager.

        Returns:
            Information dictionary
        """
        return {
            "provider": self.provider_name,
            "dimension": self.provider.get_dimension(),
            "cache_enabled": self.cache_enabled,
            "cache_size": len(self.cache),
            "available": getattr(self.provider, "available", True),
        }
