"""
Document Indexer Module

Handles indexing of call transcripts and metadata
into the vector database for semantic search.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """
    Indexes documents into the vector database with preprocessing
    and metadata extraction.
    """

    def __init__(self, vector_db_client, config: dict = None):
        """
        Initialize the document indexer.

        Args:
            vector_db_client: Vector database client instance
            config: Configuration dictionary
        """
        self.vector_db = vector_db_client
        self.config = config or {}

        # Indexing settings
        self.preprocess_text = self.config.get("preprocess_text", True)
        self.min_text_length = self.config.get("min_text_length", 50)
        self.max_text_length = self.config.get("max_text_length", 10000)
        self.text_fields = self.config.get("text_fields", ["transcript"])
        self.metadata_fields = self.config.get(
            "metadata_fields",
            [
                "call_id",
                "agent_id",
                "campaign",
                "call_type",
                "outcome",
                "start_time",
                "duration_seconds",
            ],
        )

        self.indexed_count = 0

        logger.info("DocumentIndexer initialized")

    def index_dataframe(
        self, df: pd.DataFrame, batch_size: int = 100, update_existing: bool = True
    ) -> int:
        """
        Index a DataFrame of calls into the vector database.

        Args:
            df: DataFrame containing call data
            batch_size: Number of documents to index at once
            update_existing: Whether to update existing documents

        Returns:
            Number of documents indexed
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for indexing")
            return 0

        # Prepare documents
        documents, ids, metadatas = self._prepare_documents(df)

        if not documents:
            logger.warning("No valid documents to index")
            return 0

        # Index in batches
        total_indexed = 0

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]

            try:
                # Check for existing documents if not updating
                if not update_existing:
                    existing = self._check_existing(batch_ids)
                    if existing:
                        # Filter out existing documents
                        new_indices = [
                            j for j, doc_id in enumerate(batch_ids) if doc_id not in existing
                        ]

                        if not new_indices:
                            logger.info(f"Skipping batch {i//batch_size + 1}: all documents exist")
                            continue

                        batch_docs = [batch_docs[j] for j in new_indices]
                        batch_ids = [batch_ids[j] for j in new_indices]
                        batch_meta = [batch_meta[j] for j in new_indices]

                # Index the batch
                count = self.vector_db.add_documents(
                    documents=batch_docs, ids=batch_ids, metadatas=batch_meta
                )

                total_indexed += count
                logger.info(f"Indexed batch {i//batch_size + 1}: {count} documents")

            except Exception as e:
                logger.error(f"Error indexing batch {i//batch_size + 1}: {e}")
                continue

        self.indexed_count += total_indexed
        logger.info(f"Total documents indexed: {total_indexed}")

        return total_indexed

    def _prepare_documents(self, df: pd.DataFrame) -> tuple:
        """
        Prepare documents from DataFrame for indexing.

        Args:
            df: DataFrame with call data

        Returns:
            Tuple of (documents, ids, metadatas)
        """
        documents = []
        ids = []
        metadatas = []

        for _, row in df.iterrows():
            # Extract document text
            doc_text = self._extract_document_text(row)

            # Skip if text is too short or too long
            if not self._validate_text(doc_text):
                continue

            # Preprocess text if enabled
            if self.preprocess_text:
                doc_text = self._preprocess_text(doc_text)

            # Extract document ID
            doc_id = str(row.get("call_id", ""))
            if not doc_id:
                logger.warning("Skipping document without call_id")
                continue

            # Extract metadata
            metadata = self._extract_metadata(row)

            documents.append(doc_text)
            ids.append(doc_id)
            metadatas.append(metadata)

        return documents, ids, metadatas

    def _extract_document_text(self, row: pd.Series) -> str:
        """
        Extract and combine text from specified fields.

        Args:
            row: DataFrame row

        Returns:
            Combined document text
        """
        text_parts = []

        for field in self.text_fields:
            if field in row and pd.notna(row[field]):
                text = str(row[field]).strip()

                # Treat placeholder values as missing
                if not text or text.lower() in {"none", "nan", "null"}:
                    continue

                # Add field label for context when not using the transcript field
                if field != "transcript":
                    text_parts.append(f"{field.upper()}: {text}")
                else:
                    text_parts.append(text)

        return "\n\n".join(text_parts).strip()

    def _validate_text(self, text: str) -> bool:
        """
        Validate text meets indexing criteria.

        Args:
            text: Document text

        Returns:
            True if valid
        """
        if not text or not text.strip():
            return False

        text_length = len(text.strip())

        if text_length < self.min_text_length:
            logger.debug(f"Text too short: {text_length} < {self.min_text_length}")
            return False

        if text_length > self.max_text_length:
            logger.debug(f"Text too long: {text_length} > {self.max_text_length}")
            return False

        return True

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for indexing.

        Args:
            text: Raw text

        Returns:
            Preprocessed text
        """
        # Remove extra whitespace
        text = " ".join(text.split())

        # Truncate if needed
        if len(text) > self.max_text_length:
            text = text[: self.max_text_length] + "..."

        return text

    def _extract_metadata(self, row: pd.Series) -> dict[str, Any]:
        """
        Extract metadata from DataFrame row.

        Args:
            row: DataFrame row

        Returns:
            Metadata dictionary
        """
        metadata = {}

        for field in self.metadata_fields:
            if field in row:
                value = row[field]

                # Handle different data types
                if pd.isna(value):
                    metadata[field] = None
                elif isinstance(value, (pd.Timestamp, datetime)):
                    metadata[field] = value.isoformat()
                elif isinstance(value, (int, float)):
                    metadata[field] = float(value)
                else:
                    metadata[field] = str(value)

        # Add indexing timestamp
        metadata["indexed_at"] = datetime.now().isoformat()

        return metadata

    def _check_existing(self, ids: list[str]) -> set:
        """
        Check which document IDs already exist in the database.

        Args:
            ids: List of document IDs to check

        Returns:
            Set of existing IDs
        """
        try:
            existing_docs = self.vector_db.get_by_ids(ids)
            return {doc["id"] for doc in existing_docs if doc.get("id")}
        except Exception as e:
            logger.error(f"Error checking existing documents: {e}")
            return set()

    def reindex_all(self, df: pd.DataFrame) -> int:
        """
        Reindex all documents, clearing existing index first.

        Args:
            df: DataFrame with all call data

        Returns:
            Number of documents indexed
        """
        logger.info("Starting full reindex")

        # Clear existing collection
        try:
            self.vector_db.clear_collection()
            logger.info("Cleared existing collection")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return 0

        # Index all documents
        return self.index_dataframe(df, update_existing=True)

    def update_document(
        self, call_id: str, text: str | None = None, metadata: dict[str, Any] | None = None
    ) -> bool:
        """
        Update a single document in the index.

        Args:
            call_id: Document ID to update
            text: New text (if updating)
            metadata: New metadata (if updating)

        Returns:
            True if successful
        """
        try:
            if text:
                # Need to reindex with new text
                self.vector_db.delete_documents([call_id])
                self.vector_db.add_documents(
                    documents=[text], ids=[call_id], metadatas=[metadata] if metadata else None
                )
            elif metadata:
                # Just update metadata
                self.vector_db.update_metadata([call_id], [metadata])

            logger.info(f"Updated document: {call_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating document {call_id}: {e}")
            return False

    def delete_documents(self, call_ids: list[str]) -> int:
        """
        Delete documents from the index.

        Args:
            call_ids: List of document IDs to delete

        Returns:
            Number of documents deleted
        """
        try:
            count = self.vector_db.delete_documents(call_ids)
            logger.info(f"Deleted {count} documents")
            return count
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return 0

    def get_index_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the current index.

        Returns:
            Statistics dictionary
        """
        stats = self.vector_db.get_statistics()
        stats["indexed_this_session"] = self.indexed_count

        return stats

    def export_index_metadata(self, output_path: Path):
        """
        Export index metadata for backup or analysis.

        Args:
            output_path: Path to save metadata
        """
        try:
            # Get all document IDs (this might be limited by the database)
            # For large collections, you might need pagination
            sample_results = self.vector_db.search(query_text="", top_k=1000)

            metadata_list = []
            for result in sample_results:
                metadata_list.append(
                    {
                        "id": result.get("id"),
                        "metadata": result.get("metadata", {}),
                        "snippet": result.get("document", "")[:100],
                    }
                )

            # Save to JSON
            with open(output_path, "w") as f:
                json.dump(
                    {
                        "exported_at": datetime.now().isoformat(),
                        "document_count": len(metadata_list),
                        "documents": metadata_list,
                    },
                    f,
                    indent=2,
                )

            logger.info(f"Exported index metadata to {output_path}")

        except Exception as e:
            logger.error(f"Error exporting index metadata: {e}")
