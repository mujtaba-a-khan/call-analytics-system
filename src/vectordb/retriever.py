"""
Document Retriever Module

Handles retrieval of relevant documents from the vector database
with various search strategies and post-processing.
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Container for retrieval results"""

    documents: list[dict[str, Any]]
    query: str
    total_results: int
    search_time_ms: float
    filters_applied: dict[str, Any]


class DocumentRetriever:
    """
    Retrieves relevant documents from the vector database
    with support for filtering, reranking, and result processing.
    """

    def __init__(self, vector_db_client: Any, config: dict[str, Any] | None = None) -> None:
        """
        Initialize the document retriever.

        Args:
            vector_db_client: Vector database client instance
            config: Configuration dictionary
        """
        self.vector_db = vector_db_client
        self.config = config or {}

        # Retrieval settings
        self.min_similarity_score = self.config.get("min_similarity_score", 0.5)
        self.enable_reranking = self.config.get("enable_reranking", False)
        self.reranking_candidates = self.config.get("reranking_candidates", 50)

        logger.info("DocumentRetriever initialized")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        rerank: bool | None = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters
            rerank: Whether to rerank results (overrides config)

        Returns:
            RetrievalResult with documents and metadata
        """
        import time

        start_time = time.time()

        # Determine if reranking should be used
        use_reranking = rerank if rerank is not None else self.enable_reranking

        # Get more candidates if reranking
        search_k = self.reranking_candidates if use_reranking else top_k

        # Perform search
        results = self.vector_db.search(query_text=query, top_k=search_k, filter_dict=filters)

        # Filter by minimum similarity
        results = self._filter_by_similarity(results)

        # Rerank if enabled
        if use_reranking and len(results) > top_k:
            results = self._rerank_results(query, results, top_k)
        else:
            results = results[:top_k]

        # Post-process results
        results = self._post_process_results(results)

        # Calculate search time
        search_time_ms = (time.time() - start_time) * 1000

        return RetrievalResult(
            documents=results,
            query=query,
            total_results=len(results),
            search_time_ms=search_time_ms,
            filters_applied=filters or {},
        )

    def _filter_by_similarity(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Filter results by minimum similarity score.

        Args:
            results: Search results

        Returns:
            Filtered results
        """
        filtered = []

        for result in results:
            score = result.get("score", 0)
            if score >= self.min_similarity_score:
                filtered.append(result)
            else:
                # Stop filtering once we hit low scores (assuming sorted)
                break

        if len(filtered) < len(results):
            logger.debug(f"Filtered {len(results) - len(filtered)} low-similarity results")

        return filtered

    def _rerank_results(
        self, query: str, results: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        """
        Rerank search results using additional signals.

        Args:
            query: Original query
            results: Initial search results
            top_k: Number of results to return

        Returns:
            Reranked results
        """
        # Calculate additional ranking signals
        for result in results:
            # Original score
            base_score = result.get("score", 0.5)

            # Recency boost (if timestamp available)
            recency_score = self._calculate_recency_score(result)

            # Metadata quality score
            metadata_score = self._calculate_metadata_score(result)

            # Query-document overlap
            overlap_score = self._calculate_overlap_score(query, result)

            # Combine scores
            result["rerank_score"] = (
                base_score * 0.5
                + recency_score * 0.2
                + metadata_score * 0.15
                + overlap_score * 0.15
            )

        # Sort by rerank score
        results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        return results[:top_k]

    def _calculate_recency_score(self, result: dict[str, Any]) -> float:
        """
        Calculate recency score based on document timestamp.

        Args:
            result: Search result

        Returns:
            Recency score (0-1)
        """
        metadata = result.get("metadata", {})

        if "start_time" in metadata:
            try:
                # Parse timestamp
                if isinstance(metadata["start_time"], str):
                    doc_time = datetime.fromisoformat(metadata["start_time"])
                else:
                    doc_time = metadata["start_time"]

                # Calculate age in days
                age_days = (datetime.now() - doc_time).days

                # Exponential decay with 30-day half-life
                score = math.exp(-age_days / 30)

                return score

            except Exception:
                pass

        return 0.5  # Neutral score if no timestamp

    def _calculate_metadata_score(self, result: dict[str, Any]) -> float:
        """
        Calculate score based on metadata completeness.

        Args:
            result: Search result

        Returns:
            Metadata score (0-1)
        """
        metadata = result.get("metadata", {})

        # Important metadata fields
        important_fields = ["call_type", "outcome", "agent_id", "duration"]

        # Count populated fields
        populated = sum(1 for field in important_fields if metadata.get(field))

        return populated / len(important_fields)

    def _calculate_overlap_score(self, query: str, result: dict[str, Any]) -> float:
        """
        Calculate query-document term overlap.

        Args:
            query: Search query
            result: Search result

        Returns:
            Overlap score (0-1)
        """
        document = result.get("document", "").lower()
        query_terms = set(query.lower().split())

        if not query_terms or not document:
            return 0

        # Count matching terms
        matches = sum(1 for term in query_terms if term in document)

        return matches / len(query_terms)

    def _post_process_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Post-process results for presentation.

        Args:
            results: Raw search results

        Returns:
            Processed results
        """
        processed = []

        for result in results:
            # Create processed result
            processed_result = {
                "id": result.get("id"),
                "score": result.get("rerank_score", result.get("score", 0)),
                "document": result.get("document", ""),
                "metadata": result.get("metadata", {}),
                "snippet": self._extract_snippet(result.get("document", ""), max_length=200),
                "highlights": [],  # Could add query term highlighting
            }

            processed.append(processed_result)

        return processed

    def _extract_snippet(self, text: str, max_length: int = 200) -> str:
        """
        Extract a snippet from text.

        Args:
            text: Full text
            max_length: Maximum snippet length

        Returns:
            Text snippet
        """
        if len(text) <= max_length:
            return text

        # Try to break at sentence
        snippet = text[:max_length]
        last_period = snippet.rfind(".")

        if last_period > max_length * 0.5:
            return snippet[: last_period + 1]

        # Break at word boundary
        last_space = snippet.rfind(" ")
        if last_space > 0:
            return snippet[:last_space] + "..."

        return snippet + "..."

    def retrieve_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """
        Retrieve documents by their IDs.

        Args:
            ids: List of document IDs

        Returns:
            List of documents
        """
        try:
            results = self.vector_db.get_by_ids(ids)
            return self._post_process_results(results)
        except Exception as e:
            logger.error(f"Error retrieving documents by IDs: {e}")
            return []

    def retrieve_similar(self, document_id: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Retrieve documents similar to a given document.

        Args:
            document_id: Reference document ID
            top_k: Number of similar documents

        Returns:
            List of similar documents
        """
        # Get the reference document
        ref_docs = self.retrieve_by_ids([document_id])

        if not ref_docs:
            logger.warning(f"Reference document not found: {document_id}")
            return []

        ref_text = ref_docs[0].get("document", "")

        # Search for similar documents
        results = self.retrieve(
            query=ref_text[:1000],  # Use first 1000 chars as query
            top_k=top_k + 1,  # Get extra to exclude self
        )

        # Filter out the reference document
        similar = [doc for doc in results.documents if doc.get("id") != document_id]

        return similar[:top_k]

    def retrieve_with_feedback(
        self,
        query: str,
        positive_ids: list[str] | None = None,
        negative_ids: list[str] | None = None,
        top_k: int = 10,
    ) -> RetrievalResult:
        """
        Retrieve with relevance feedback.

        Args:
            query: Search query
            positive_ids: IDs of relevant documents
            negative_ids: IDs of irrelevant documents
            top_k: Number of results

        Returns:
            RetrievalResult with adjusted ranking
        """
        # Get initial results
        results = self.retrieve(query, top_k=top_k * 2)

        if positive_ids or negative_ids:
            # Adjust scores based on feedback
            for doc in results.documents:
                doc_id = doc.get("id")

                if positive_ids and doc_id in positive_ids:
                    # Boost similar to positive examples
                    doc["score"] *= 1.5

                if negative_ids and doc_id in negative_ids:
                    # Reduce similar to negative examples
                    doc["score"] *= 0.5

            # Re-sort by adjusted scores
            results.documents.sort(key=lambda x: x.get("score", 0), reverse=True)

            # Trim to top_k
            results.documents = results.documents[:top_k]
            results.total_results = len(results.documents)

        return results
