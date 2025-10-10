"""
Semantic Search Module

Implements semantic search capabilities using vector embeddings
to find relevant calls based on meaning rather than exact matches.
"""

import logging
import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SemanticSearchEngine:
    """
    Engine for performing semantic search over call transcripts.
    Uses embeddings and vector similarity to find relevant matches.
    """

    def __init__(self, vector_db_client=None, embedding_generator=None):
        """
        Initialize the semantic search engine.

        Args:
            vector_db_client: Vector database client for retrieval
            embedding_generator: Embedding generator for queries
        """
        self.vector_db = vector_db_client
        self.embedder = embedding_generator
        self.cached_embeddings = {}

        logger.info("SemanticSearchEngine initialized")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        threshold: float | None = None,
        rerank: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Perform semantic search for relevant calls.

        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Optional metadata filters
            threshold: Minimum similarity score (0-1) required for a result
            rerank: Whether to rerank results

        Returns:
            List of search results with scores
        """
        if not self.vector_db:
            logger.warning("No vector database configured for semantic search")
            return []

        try:
            # Get initial results from vector database
            vector_filters = self._build_vector_filters(filters)

            results = self.vector_db.search(
                query_text=query,
                top_k=top_k * 3 if rerank else top_k,  # Get more candidates for reranking
                filter_dict=vector_filters,
            )

            # Apply reranking if requested
            if rerank and results:
                results = self._rerank_results(query, results, top_k)

            # Apply post-retrieval filtering for fields not handled by vector DB
            if filters:
                results = self._apply_post_filters(results, filters)

            # Apply similarity threshold filtering when requested
            if threshold is not None:
                try:
                    threshold_value = float(threshold)
                except (TypeError, ValueError):
                    logger.warning("Invalid threshold '%s' provided; ignoring filter", threshold)
                    threshold_value = None

                if threshold_value is not None:
                    before_count = len(results)
                    results = [
                        result for result in results if result.get("score", 0.0) >= threshold_value
                    ]
                    if before_count and len(results) < before_count:
                        logger.debug(
                            "Filtered %d results below threshold %.2f",
                            before_count - len(results),
                            threshold_value,
                        )

            # Enhance results with additional metadata
            results = self._enhance_results(results)

            logger.info(
                "Semantic search returned %d results for query: '%s...'",
                len(results),
                query[:50],
            )
            return results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def _build_vector_filters(self, filters: dict[str, Any] | None) -> dict[str, Any] | None:
        """Translate UI filters into Chroma-compatible where clauses."""
        if not filters:
            return None

        clauses: list[dict[str, Any]] = []

        # Categorical filters
        for field, key in [
            ("selected_agents", "agent_id"),
            ("selected_campaigns", "campaign"),
            ("selected_outcomes", "outcome"),
            ("selected_types", "call_type"),
        ]:
            values = filters.get(field)
            if values:
                clauses.append({key: {"$in": values}})

        # Duration range (seconds/minutes depending on stored field)
        duration_range = filters.get("duration_range")
        if duration_range and len(duration_range) == 2:
            min_dur, max_dur = duration_range
            if min_dur and not math.isinf(min_dur):
                clauses.append({"duration": {"$gte": float(min_dur)}})
            if max_dur and not math.isinf(max_dur):
                clauses.append({"duration": {"$lte": float(max_dur)}})

        # Revenue range
        revenue_range = filters.get("revenue_range")
        if revenue_range and len(revenue_range) == 2:
            min_rev, max_rev = revenue_range
            if min_rev and not math.isinf(min_rev):
                clauses.append({"revenue": {"$gte": float(min_rev)}})
            if max_rev and not math.isinf(max_rev):
                clauses.append({"revenue": {"$lte": float(max_rev)}})

        if not clauses:
            return None

        if len(clauses) == 1:
            return clauses[0]

        return {"$and": clauses}

    def _apply_post_filters(
        self, results: list[dict[str, Any]], filters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Filter results using metadata for constraints the vector DB cannot handle."""
        if not results:
            return results

        filtered: list[dict[str, Any]] = []

        # Prepare date range bounds
        date_range = filters.get("date_range")
        start_date = end_date = None
        if date_range and len(date_range) == 2:
            start_date = self._parse_date(date_range[0])
            end_date = self._parse_date(date_range[1], end_of_day=True)

        for result in results:
            metadata = result.get("metadata", {}) or {}

            # Date range filtering
            if start_date or end_date:
                ts_value = metadata.get("timestamp") or metadata.get("start_time")
                ts = self._parse_date(ts_value)
                if ts is None:
                    continue
                if start_date and ts < start_date:
                    continue
                if end_date and ts > end_date:
                    continue

            filtered.append(result)

        return filtered

    @staticmethod
    def _parse_date(value: str | None, end_of_day: bool = False) -> datetime | None:
        """Parse a date/datetime string to datetime object."""
        if not value or isinstance(value, float) and math.isnan(value):
            return None

        if isinstance(value, datetime):
            dt = value
        else:
            str_value = str(value)

            # Handle date-only strings
            try:
                if "T" in str_value or " " in str_value:
                    dt = datetime.fromisoformat(str_value.replace("Z", "+00:00"))
                else:
                    dt = datetime.fromisoformat(str_value)
            except ValueError:
                return None

        if end_of_day and dt.time() == datetime.min.time():
            return dt.replace(hour=23, minute=59, second=59, microsecond=999999)

        return dt

    def _rerank_results(
        self, query: str, results: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        """
        Rerank search results using cross-encoder or advanced scoring.

        Args:
            query: Original query
            results: Initial search results
            top_k: Number of results to return after reranking

        Returns:
            Reranked results
        """
        # Calculate additional relevance scores
        for result in results:
            # Add query-document similarity features
            doc_text = result.get("document", "")

            # Length-normalized score
            doc_length = len(doc_text.split())
            length_penalty = 1.0 / (1.0 + np.log(max(doc_length, 1)))

            # Keyword overlap score
            query_words = set(query.lower().split())
            doc_words = set(doc_text.lower().split())
            overlap = len(query_words & doc_words) / max(len(query_words), 1)

            # Combine scores
            original_score = result.get("score", 0.5)
            rerank_score = original_score * 0.6 + overlap * 0.3 + length_penalty * 0.1

            result["rerank_score"] = rerank_score

        # Sort by rerank score
        results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        return results[:top_k]

    def _enhance_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Enhance search results with additional metadata and formatting.

        Args:
            results: Raw search results

        Returns:
            Enhanced results
        """
        for result in results:
            # Add snippet extraction
            document = result.get("document", "")
            result["snippet"] = self._extract_snippet(document, max_length=200)

            # Format metadata
            metadata = result.get("metadata", {})
            result["formatted_metadata"] = self._format_metadata(metadata)

            # Add relevance label
            score = result.get("score", 0)
            if score > 0.8:
                result["relevance"] = "High"
            elif score > 0.6:
                result["relevance"] = "Medium"
            else:
                result["relevance"] = "Low"

        return results

    def _extract_snippet(self, text: str, max_length: int = 200) -> str:
        """
        Extract a relevant snippet from the text.

        Args:
            text: Full text
            max_length: Maximum snippet length

        Returns:
            Text snippet
        """
        if len(text) <= max_length:
            return text

        # Try to break at sentence boundary
        snippet = text[:max_length]
        last_period = snippet.rfind(".")
        if last_period > max_length * 0.5:
            snippet = snippet[: last_period + 1]
        else:
            snippet += "..."

        return snippet

    def _format_metadata(self, metadata: dict[str, Any]) -> str:
        """
        Format metadata for display.

        Args:
            metadata: Raw metadata dictionary

        Returns:
            Formatted metadata string
        """
        formatted_parts = []

        if "start_time" in metadata:
            formatted_parts.append(f"Time: {metadata['start_time']}")

        if "agent_id" in metadata:
            formatted_parts.append(f"Agent: {metadata['agent_id']}")

        if "duration" in metadata:
            duration = float(metadata["duration"])
            formatted_parts.append(f"Duration: {duration:.1f}s")

        return " | ".join(formatted_parts)

    def find_similar_calls(self, call_id: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find calls similar to a given call.

        Args:
            call_id: ID of the reference call
            top_k: Number of similar calls to return

        Returns:
            List of similar calls
        """
        if not self.vector_db:
            return []

        try:
            # Get the reference call
            reference_calls = self.vector_db.get_by_ids([call_id])
            if not reference_calls:
                logger.warning(f"Reference call not found: {call_id}")
                return []

            reference_text = reference_calls[0].get("document", "")

            # Search for similar calls
            results = self.search(
                query=reference_text, top_k=top_k + 1, filters=None  # Get extra to exclude self
            )

            # Filter out the reference call itself
            results = [r for r in results if r.get("id") != call_id]

            return results[:top_k]

        except Exception as e:
            logger.error(f"Error finding similar calls: {e}")
            return []

    def cluster_search_results(
        self, results: list[dict[str, Any]], n_clusters: int = 3
    ) -> dict[int, list[dict[str, Any]]]:
        """
        Cluster search results into groups.

        Args:
            results: Search results to cluster
            n_clusters: Number of clusters

        Returns:
            Dictionary mapping cluster ID to results
        """
        if len(results) < n_clusters:
            # Not enough results to cluster
            return {0: results}

        try:
            from sklearn.cluster import KMeans

            # Get embeddings for clustering
            texts = [r.get("document", "") for r in results]

            if self.embedder:
                embeddings = self.embedder.generate_embeddings(texts)
            else:
                # Fallback to simple feature extraction
                embeddings = self._simple_text_features(texts)

            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)

            # Group results by cluster
            clusters = {}
            for idx, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(results[idx])

            logger.info(f"Clustered {len(results)} results into {len(clusters)} groups")
            return clusters

        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return {0: results}

    def _simple_text_features(self, texts: list[str]) -> np.ndarray:
        """
        Extract simple text features for fallback clustering.

        Args:
            texts: List of texts

        Returns:
            Feature matrix
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
        features = vectorizer.fit_transform(texts).toarray()

        return features

    def explain_relevance(self, query: str, result: dict[str, Any]) -> dict[str, Any]:
        """
        Explain why a result is relevant to the query.

        Args:
            query: Search query
            result: Search result

        Returns:
            Explanation dictionary
        """
        document = result.get("document", "")

        # Find matching keywords
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        matching_words = query_words & doc_words

        # Calculate various relevance metrics
        keyword_overlap = len(matching_words) / max(len(query_words), 1)

        # Find matching phrases
        matching_phrases = []
        query_lower = query.lower()
        doc_lower = document.lower()

        # Check for 2-word phrases
        query_bigrams = [
            f"{query_words[i]} {query_words[i+1]}"
            for i, query_words in enumerate(query_lower.split()[:-1])
        ]

        for bigram in query_bigrams:
            if bigram in doc_lower:
                matching_phrases.append(bigram)

        explanation = {
            "score": result.get("score", 0),
            "matching_keywords": list(matching_words),
            "keyword_overlap_ratio": keyword_overlap,
            "matching_phrases": matching_phrases,
            "relevance_label": result.get("relevance", "Unknown"),
        }

        return explanation


class HybridSearchEngine:
    """
    Combines semantic search with keyword search for improved results.
    """

    def __init__(self, semantic_engine: SemanticSearchEngine, df: pd.DataFrame):
        """
        Initialize hybrid search engine.

        Args:
            semantic_engine: Semantic search engine
            df: DataFrame containing call data for keyword search
        """
        self.semantic_engine = semantic_engine
        self.df = df

        logger.info("HybridSearchEngine initialized")

    def search(
        self, query: str, top_k: int = 10, semantic_weight: float = 0.7
    ) -> list[dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword approaches.

        Args:
            query: Search query
            top_k: Number of results to return
            semantic_weight: Weight for semantic results (0-1)

        Returns:
            Combined search results
        """
        # Get semantic search results
        semantic_results = self.semantic_engine.search(query, top_k=top_k * 2)

        # Get keyword search results
        keyword_results = self._keyword_search(query, top_k=top_k * 2)

        # Combine and rerank results
        combined_results = self._combine_results(semantic_results, keyword_results, semantic_weight)

        return combined_results[:top_k]

    def _keyword_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """
        Perform keyword-based search on the DataFrame.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            Keyword search results
        """
        if self.df.empty or "transcript" not in self.df.columns:
            return []

        # Simple keyword matching
        query_lower = query.lower()
        scores = []

        for idx, row in self.df.iterrows():
            transcript = str(row.get("transcript", "")).lower()

            # Calculate simple relevance score
            score = 0
            for word in query_lower.split():
                if word in transcript:
                    score += 1

            if score > 0:
                scores.append(
                    {
                        "id": row.get("call_id", idx),
                        "document": row.get("transcript", ""),
                        "score": score / len(query_lower.split()),
                        "metadata": row.to_dict(),
                        "source": "keyword",
                    }
                )

        # Sort by score
        scores.sort(key=lambda x: x["score"], reverse=True)

        return scores[:top_k]

    def _combine_results(
        self,
        semantic_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
        semantic_weight: float,
    ) -> list[dict[str, Any]]:
        """
        Combine and rerank results from different search methods.

        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            semantic_weight: Weight for semantic scores

        Returns:
            Combined and reranked results
        """
        combined = {}
        keyword_weight = 1.0 - semantic_weight

        # Add semantic results
        for result in semantic_results:
            call_id = result.get("id")
            if call_id:
                combined[call_id] = {
                    **result,
                    "combined_score": result.get("score", 0) * semantic_weight,
                    "sources": ["semantic"],
                }

        # Add or update with keyword results
        for result in keyword_results:
            call_id = result.get("id")
            if call_id:
                if call_id in combined:
                    # Update existing result
                    combined[call_id]["combined_score"] += result.get("score", 0) * keyword_weight
                    combined[call_id]["sources"].append("keyword")
                else:
                    # Add new result
                    combined[call_id] = {
                        **result,
                        "combined_score": result.get("score", 0) * keyword_weight,
                        "sources": ["keyword"],
                    }

        # Convert to list and sort by combined score
        results = list(combined.values())
        results.sort(key=lambda x: x["combined_score"], reverse=True)

        return results
