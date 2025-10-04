"""
Vector Index Rebuild Script for Call Analytics System

This script rebuilds the vector database index from stored call records,
transcripts, and notes for semantic search functionality.
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
import shutil

import pandas as pd
import toml

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging, get_logger
from src.core.storage_manager import StorageManager
from src.vectordb.chroma_client import ChromaClient
from src.vectordb.indexer import DocumentIndexer


class IndexRebuilder:
    """High-level helper for rebuilding the semantic search index."""

    def __init__(
        self,
        storage_manager: StorageManager,
        vector_client: ChromaClient,
        indexer: DocumentIndexer,
        logger: logging.Logger,
    ) -> None:
        self.storage_manager = storage_manager
        self.vector_client = vector_client
        self.indexer = indexer
        self.logger = logger

    def backup_existing_index(self, backup_dir: Path) -> bool:
        """Backup the current vector store files if they exist."""
        try:
            persist_dir: Path = self.vector_client.persist_dir
            if not persist_dir.exists() or not any(persist_dir.iterdir()):
                self.logger.info("No existing index to backup")
                return True

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = backup_dir / f"index_backup_{timestamp}"
            backup_path.mkdir(parents=True, exist_ok=True)

            shutil.copytree(persist_dir, backup_path / 'vectorstore', dirs_exist_ok=True)

            metadata = self.vector_client.get_statistics()
            metadata['timestamp'] = timestamp

            with open(backup_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

            self.logger.info("Backup created at %s", backup_path)
            return True

        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error("Failed to backup index: %s", exc)
            return False

    def load_records(self) -> pd.DataFrame:
        """Load all call records from storage."""
        return self.storage_manager.load_all_records()

    def rebuild_index(
        self,
        records: pd.DataFrame,
        batch_size: int = 100,
        clear_existing: bool = True,
    ) -> int:
        """Rebuild or update the vector index from the provided records."""
        if records.empty:
            self.logger.warning("No records available to index")
            return 0

        if clear_existing:
            self.logger.info("Clearing existing collection before indexing")
            indexed_count = self.indexer.reindex_all(records)
        else:
            indexed_count = self.indexer.index_dataframe(
                records,
                batch_size=batch_size,
                update_existing=True,
            )

        self.logger.info("Indexed %d document(s)", indexed_count)
        return indexed_count

    def verify_index(self, sample_queries: Optional[List[str]] = None) -> bool:
        """Run a few sample queries to sanity-check the rebuilt index."""
        if sample_queries is None:
            sample_queries = [
                "customer complaint",
                "billing issue",
                "technical support",
                "refund request",
                "positive feedback",
            ]

        all_passed = True
        for query in sample_queries:
            try:
                results = self.vector_client.search(query_text=query, top_k=5)
                if results:
                    self.logger.info("✓ Query '%s' returned %d result(s)", query, len(results))
                else:
                    self.logger.warning("✗ Query '%s' returned no results", query)
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.error("✗ Query '%s' failed: %s", query, exc)
                all_passed = False

        return all_passed

    def generate_index_stats(self) -> Dict[str, Any]:
        """Return statistics reported by the vector client."""
        try:
            stats = self.vector_client.get_statistics()
            stats['generated_at'] = datetime.now().isoformat()
            return stats
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error("Failed to generate index stats: %s", exc)
            return {}


def load_vector_config(config_path: Path, persist_dir: Optional[Path]) -> Dict[str, Any]:
    """Load vector store configuration and apply CLI overrides."""
    config_data = toml.load(config_path)
    vector_cfg = dict(config_data.get('vectorstore', {}))

    if persist_dir is not None:
        vector_cfg['persist_dir'] = str(persist_dir)

    return vector_cfg


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
        '--vector-config',
        type=Path,
        default=Path('config/vectorstore.toml'),
        help='Path to vector store configuration TOML file'
    )

    parser.add_argument(
        '--vector-db-path',
        type=Path,
        default=Path('data/vectorstore'),
        help='Override vector store persistence directory'
    )
    
    parser.add_argument(
        '--backup-dir',
        type=Path,
        default=Path('backups'),
        help='Directory for index backups'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for indexing'
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
        storage_manager = StorageManager(base_path=args.data_dir)

        # Vector database client + indexer
        vector_config = load_vector_config(args.vector_config, args.vector_db_path)
        vector_client = ChromaClient(vector_config)

        indexing_config = dict(vector_config.get('indexing', {}))
        indexing_config.setdefault('text_fields', ['transcript', 'notes'])
        indexing_config.setdefault('min_text_length', 10)
        indexing_config.setdefault(
            'metadata_fields',
            [
                'call_id',
                'agent_id',
                'campaign',
                'call_type',
                'outcome',
                'timestamp',
                'duration',
                'revenue',
            ],
        )

        indexer = DocumentIndexer(vector_client, config=indexing_config)

        rebuilder = IndexRebuilder(
            storage_manager=storage_manager,
            vector_client=vector_client,
            indexer=indexer,
            logger=logger,
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

            # Load records
            records = rebuilder.load_records()

            if records is None or records.empty:
                logger.warning("No records to index")
                sys.exit(0)

            # Rebuild index
            indexed_count = rebuilder.rebuild_index(
                records,
                batch_size=args.batch_size,
                clear_existing=not args.no_clear,
            )

            if indexed_count == 0:
                logger.warning("No documents were indexed")
            
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
