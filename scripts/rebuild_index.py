"""
Vector Index Rebuild Script for Call Analytics System

This script rebuilds the vector database index from stored call records,
transcripts, and notes for semantic search functionality.
"""

import sys
import os
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import time
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import shutil

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging, get_logger
from src.core.storage_manager import StorageManager
from src.vectordb.chroma_client import ChromaDBClient
from src.ml.embeddings import EmbeddingManager
from src.vectordb.indexer import VectorIndexer


class IndexRebuilder:
    """
    Handles rebuilding of vector database index from call records.
    """
    
    def __init__(self, 
                 storage_manager: StorageManager,
                 vector_client: ChromaDBClient,
                 embedding_manager: EmbeddingManager,
                 logger: logging.Logger):
        """
        Initialize index rebuilder.
        
        Args:
            storage_manager: Storage manager for accessing call records
            vector_client: Vector database client
            embedding_manager: Embedding generation manager
            logger: Logger instance
        """
        self.storage_manager = storage_manager
        self.vector_client = vector_client
        self.embedding_manager = embedding_manager
        self.logger = logger
        self.indexer = VectorIndexer(vector_client, embedding_manager)
        
    def backup_existing_index(self, backup_dir: Path) -> bool:
        """
        Backup existing vector index before rebuilding.
        
        Args:
            backup_dir: Directory to store backup
            
        Returns:
            True if backup successful, False otherwise
        """
        try:
            self.logger.info("Backing up existing index...")
            
            # Create backup directory with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = backup_dir / f"index_backup_{timestamp}"
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Get current index location
            index_path = Path('data/vector_db')  # Default ChromaDB location
            
            if index_path.exists():
                # Copy index files
                shutil.copytree(index_path, backup_path / 'vector_db', dirs_exist_ok=True)
                
                # Save backup metadata
                metadata = {
                    'timestamp': timestamp,
                    'record_count': self.vector_client.get_collection_size(),
                    'collections': self.vector_client.list_collections()
                }
                
                with open(backup_path / 'metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                self.logger.info(f"Backup created at {backup_path}")
                return True
            else:
                self.logger.info("No existing index to backup")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to backup index: {e}")
            return False
    
    def prepare_documents(self, batch_size: int = 100) -> List[Dict[str, Any]]:
        """
        Prepare documents from call records for indexing.
        
        Args:
            batch_size: Number of records to process at once
            
        Returns:
            List of prepared documents
        """
        self.logger.info("Preparing documents for indexing...")
        
        documents = []
        
        try:
            # Load all call records
            records = self.storage_manager.load_all_records()
            
            if records.empty:
                self.logger.warning("No records found to index")
                return documents
            
            self.logger.info(f"Found {len(records)} records to index")
            
            # Process records in batches
            for i in tqdm(range(0, len(records), batch_size), desc="Preparing documents"):
                batch = records.iloc[i:i+batch_size]
                
                for _, record in batch.iterrows():
                    # Create document for indexing
                    doc = {
                        'id': record.get('call_id', f"doc_{i}"),
                        'content': self._create_searchable_content(record),
                        'metadata': {
                            'call_id': record.get('call_id'),
                            'phone_number': record.get('phone_number'),
                            'timestamp': str(record.get('timestamp')),
                            'agent_id': record.get('agent_id'),
                            'campaign': record.get('campaign'),
                            'outcome': record.get('outcome'),
                            'duration': record.get('duration'),
                            'call_type': record.get('call_type'),
                            'revenue': record.get('revenue', 0)
                        }
                    }
                    
                    # Add transcript if available
                    if 'transcript' in record and record['transcript']:
                        doc['transcript'] = record['transcript']
                    
                    # Add notes if available
                    if 'notes' in record and record['notes']:
                        doc['notes'] = record['notes']
                    
                    documents.append(doc)
            
            self.logger.info(f"Prepared {len(documents)} documents for indexing")
            return documents
            
        except Exception as e:
            self.logger.error(f"Error preparing documents: {e}")
            return documents
    
    def _create_searchable_content(self, record: pd.Series) -> str:
        """
        Create searchable content from a call record.
        
        Args:
            record: Call record as pandas Series
            
        Returns:
            Searchable text content
        """
        content_parts = []
        
        # Add transcript if available
        if 'transcript' in record and record['transcript']:
            content_parts.append(f"Transcript: {record['transcript']}")
        
        # Add notes if available
        if 'notes' in record and record['notes']:
            content_parts.append(f"Notes: {record['notes']}")
        
        # Add call metadata
        metadata_text = f"Call from {record.get('phone_number', 'unknown')} "
        metadata_text += f"on {record.get('timestamp', 'unknown date')} "
        metadata_text += f"with outcome {record.get('outcome', 'unknown')} "
        metadata_text += f"handled by agent {record.get('agent_id', 'unknown')}"
        content_parts.append(metadata_text)
        
        # Add campaign information
        if 'campaign' in record and record['campaign']:
            content_parts.append(f"Campaign: {record['campaign']}")
        
        # Combine all parts
        return " ".join(content_parts)
    
    def rebuild_index(self, 
                      documents: List[Dict[str, Any]],
                      collection_name: str = 'call_records',
                      batch_size: int = 100,
                      clear_existing: bool = True) -> bool:
        """
        Rebuild the vector index with prepared documents.
        
        Args:
            documents: List of documents to index
            collection_name: Name of the collection
            batch_size: Batch size for indexing
            clear_existing: Whether to clear existing index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Starting index rebuild for {len(documents)} documents...")
            
            # Clear existing index if requested
            if clear_existing:
                self.logger.info("Clearing existing index...")
                self.vector_client.delete_collection(collection_name)
                self.vector_client.create_collection(collection_name)
            
            # Index documents in batches
            indexed_count = 0
            failed_count = 0
            
            for i in tqdm(range(0, len(documents), batch_size), desc="Indexing documents"):
                batch = documents[i:i+batch_size]
                
                try:
                    # Index batch
                    self.indexer.index_batch(batch, collection_name)
                    indexed_count += len(batch)
                    
                except Exception as e:
                    self.logger.error(f"Failed to index batch {i//batch_size}: {e}")
                    failed_count += len(batch)
            
            # Log results
            self.logger.info(f"Index rebuild complete:")
            self.logger.info(f"  - Documents indexed: {indexed_count}")
            self.logger.info(f"  - Documents failed: {failed_count}")
            
            # Verify index
            collection_size = self.vector_client.get_collection_size(collection_name)
            self.logger.info(f"  - Collection size: {collection_size}")
            
            return failed_count == 0
            
        except Exception as e:
            self.logger.error(f"Index rebuild failed: {e}")
            return False
    
    def verify_index(self, sample_queries: List[str] = None) -> bool:
        """
        Verify the rebuilt index with sample queries.
        
        Args:
            sample_queries: List of sample queries to test
            
        Returns:
            True if verification passes, False otherwise
        """
        try:
            self.logger.info("Verifying rebuilt index...")
            
            # Default sample queries
            if sample_queries is None:
                sample_queries = [
                    "customer complaint",
                    "billing issue",
                    "technical support",
                    "refund request",
                    "positive feedback"
                ]
            
            # Test each query
            all_passed = True
            for query in sample_queries:
                try:
                    results = self.vector_client.search(
                        query_text=query,
                        n_results=5
                    )
                    
                    if results:
                        self.logger.info(f"✓ Query '{query}' returned {len(results)} results")
                    else:
                        self.logger.warning(f"✗ Query '{query}' returned no results")
                        
                except Exception as e:
                    self.logger.error(f"✗ Query '{query}' failed: {e}")
                    all_passed = False
            
            return all_passed
            
        except Exception as e:
            self.logger.error(f"Index verification failed: {e}")
            return False
    
    def generate_index_stats(self) -> Dict[str, Any]:
        """
        Generate statistics about the vector index.
        
        Returns:
            Dictionary of index statistics
        """
        try:
            stats = {
                'timestamp': datetime.now().isoformat(),
                'collections': {},
                'total_documents': 0,
                'embedding_model': self.embedding_manager.get_model_info()
            }
            
            # Get stats for each collection
            collections = self.vector_client.list_collections()
            
            for collection in collections:
                collection_stats = {
                    'size': self.vector_client.get_collection_size(collection),
                    'metadata_fields': self.vector_client.get_metadata_fields(collection)
                }
                stats['collections'][collection] = collection_stats
                stats['total_documents'] += collection_stats['size']
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to generate index stats: {e}")
            return {}


def main():
    """
    Main function to run the index rebuild script.
    """
    parser = argparse.ArgumentParser(
        description='Rebuild vector database index for Call Analytics System'
    )
    
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data'),
        help='Data directory containing call records'
    )
    
    parser.add_argument(
        '--vector-db-path',
        type=Path,
        default=Path('data/vector_db'),
        help='Path to vector database'
    )
    
    parser.add_argument(
        '--backup-dir',
        type=Path,
        default=Path('backups'),
        help='Directory for index backups'
    )
    
    parser.add_argument(
        '--collection-name',
        default='call_records',
        help='Name of the collection to rebuild'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for indexing'
    )
    
    parser.add_argument(
        '--embedding-model',
        default='all-MiniLM-L6-v2',
        help='Embedding model to use'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip backup of existing index'
    )
    
    parser.add_argument(
        '--no-clear',
        action='store_true',
        help='Do not clear existing index'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify index after rebuild'
    )
    
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Only show index statistics'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level='INFO', console_output=True)
    logger = get_logger(__name__)
    
    try:
        # Initialize components
        logger.info("Initializing components...")
        
        # Storage manager
        storage_manager = StorageManager(data_dir=args.data_dir)
        
        # Vector database client
        vector_client = ChromaDBClient(persist_directory=str(args.vector_db_path))
        
        # Embedding manager
        embedding_manager = EmbeddingManager(model_name=args.embedding_model)
        
        # Create rebuilder
        rebuilder = IndexRebuilder(
            storage_manager,
            vector_client,
            embedding_manager,
            logger
        )
        
        if args.stats_only:
            # Show statistics only
            stats = rebuilder.generate_index_stats()
            logger.info("Current index statistics:")
            logger.info(json.dumps(stats, indent=2))
            
        else:
            # Backup existing index
            if not args.no_backup:
                if not rebuilder.backup_existing_index(args.backup_dir):
                    logger.error("Backup failed, aborting rebuild")
                    sys.exit(1)
            
            # Prepare documents
            documents = rebuilder.prepare_documents(batch_size=args.batch_size)
            
            if not documents:
                logger.warning("No documents to index")
                sys.exit(0)
            
            # Rebuild index
            success = rebuilder.rebuild_index(
                documents,
                collection_name=args.collection_name,
                batch_size=args.batch_size,
                clear_existing=not args.no_clear
            )
            
            if not success:
                logger.error("Index rebuild failed")
                sys.exit(1)
            
            # Verify if requested
            if args.verify:
                if rebuilder.verify_index():
                    logger.info("✓ Index verification passed")
                else:
                    logger.warning("⚠ Index verification failed")
            
            # Generate and show statistics
            stats = rebuilder.generate_index_stats()
            logger.info("\nFinal index statistics:")
            logger.info(json.dumps(stats, indent=2))
            
            # Save stats to file
            stats_file = args.data_dir / 'index_stats.json'
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Statistics saved to {stats_file}")
            
            logger.info("\n✓ Index rebuild completed successfully!")
            
    except Exception as e:
        logger.error(f"Script failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()