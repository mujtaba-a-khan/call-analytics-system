"""
Storage Manager Module

Handles persistent storage of call data, including
saving, loading, and managing data snapshots.
"""

import pandas as pd
import json
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import shutil
import hashlib

logger = logging.getLogger(__name__)


class StorageManager:
    """
    Manages persistent storage of call analytics data.
    Supports multiple formats and provides versioning capabilities.
    """
    
    def __init__(self, base_path: str = "data"):
        """
        Initialize the storage manager.
        
        Args:
            base_path: Base directory for data storage
        """
        self.base_path = Path(base_path)
        self.processed_path = self.base_path / "processed"
        self.snapshots_path = self.base_path / "snapshots"
        self.cache_path = self.base_path / "cache"
        self.exports_path = self.base_path / "exports"
        
        # Create necessary directories
        self._create_directories()
        
        # Initialize metadata
        self.metadata_file = self.base_path / "storage_metadata.json"
        self.metadata = self._load_metadata()
        
        logger.info(f"StorageManager initialized at {self.base_path}")
    
    def _create_directories(self):
        """Create necessary storage directories"""
        directories = [
            self.base_path,
            self.processed_path,
            self.snapshots_path,
            self.cache_path,
            self.exports_path
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """
        Load storage metadata from file.
        
        Returns:
            Metadata dictionary
        """
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
                return self._default_metadata()
        else:
            return self._default_metadata()
    
    def _default_metadata(self) -> Dict[str, Any]:
        """
        Create default metadata structure.
        
        Returns:
            Default metadata dictionary
        """
        return {
            'version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            'last_modified': datetime.now().isoformat(),
            'snapshots': [],
            'total_calls_processed': 0,
            'storage_format': 'parquet'
        }
    
    def _save_metadata(self):
        """Save metadata to file"""
        self.metadata['last_modified'] = datetime.now().isoformat()
        
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            logger.debug("Metadata saved successfully")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def save_dataframe(self, 
                      df: pd.DataFrame, 
                      name: str,
                      format: str = 'parquet',
                      create_snapshot: bool = False) -> Path:
        """
        Save a DataFrame to storage.
        
        Args:
            df: DataFrame to save
            name: Name for the saved file
            format: Storage format ('parquet', 'csv', 'pickle')
            create_snapshot: Whether to create a snapshot
        
        Returns:
            Path to the saved file
        """
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'parquet':
            filename = f"{name}_{timestamp}.parquet"
            filepath = self.processed_path / filename
            df.to_parquet(filepath, index=False)
        elif format == 'csv':
            filename = f"{name}_{timestamp}.csv"
            filepath = self.processed_path / filename
            df.to_csv(filepath, index=False)
        elif format == 'pickle':
            filename = f"{name}_{timestamp}.pkl"
            filepath = self.processed_path / filename
            df.to_pickle(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved DataFrame to {filepath}")
        
        # Create snapshot if requested
        if create_snapshot:
            self.create_snapshot(df, name)
        
        # Update metadata
        self.metadata['total_calls_processed'] += len(df)
        self._save_metadata()
        
        return filepath
    
    def load_dataframe(self, filepath: Path) -> pd.DataFrame:
        """
        Load a DataFrame from storage.
        
        Args:
            filepath: Path to the file
        
        Returns:
            Loaded DataFrame
        """
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
        elif filepath.suffix == '.csv':
            df = pd.read_csv(filepath)
        elif filepath.suffix == '.pkl':
            df = pd.read_pickle(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Loaded DataFrame from {filepath}: {len(df)} rows")
        return df
    
    def create_snapshot(self, df: pd.DataFrame, name: str) -> str:
        """
        Create a snapshot of the current data.
        
        Args:
            df: DataFrame to snapshot
            name: Name for the snapshot
        
        Returns:
            Snapshot ID
        """
        # Generate snapshot ID
        snapshot_id = hashlib.md5(
            f"{name}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:8]
        
        # Create snapshot directory
        snapshot_dir = self.snapshots_path / snapshot_id
        snapshot_dir.mkdir(exist_ok=True)
        
        # Save data
        df.to_parquet(snapshot_dir / "data.parquet", index=False)
        
        # Save snapshot metadata
        snapshot_metadata = {
            'id': snapshot_id,
            'name': name,
            'created_at': datetime.now().isoformat(),
            'rows': len(df),
            'columns': list(df.columns),
            'size_mb': (snapshot_dir / "data.parquet").stat().st_size / (1024 * 1024)
        }
        
        with open(snapshot_dir / "metadata.json", 'w') as f:
            json.dump(snapshot_metadata, f, indent=2)
        
        # Update main metadata
        self.metadata['snapshots'].append(snapshot_metadata)
        self._save_metadata()
        
        logger.info(f"Created snapshot: {snapshot_id}")
        return snapshot_id
    
    def load_snapshot(self, snapshot_id: str) -> pd.DataFrame:
        """
        Load data from a snapshot.
        
        Args:
            snapshot_id: ID of the snapshot to load
        
        Returns:
            DataFrame from the snapshot
        """
        snapshot_dir = self.snapshots_path / snapshot_id
        
        if not snapshot_dir.exists():
            raise ValueError(f"Snapshot not found: {snapshot_id}")
        
        data_file = snapshot_dir / "data.parquet"
        df = pd.read_parquet(data_file)
        
        logger.info(f"Loaded snapshot {snapshot_id}: {len(df)} rows")
        return df
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """
        List all available snapshots.
        
        Returns:
            List of snapshot metadata dictionaries
        """
        return self.metadata.get('snapshots', [])
    
    def delete_snapshot(self, snapshot_id: str):
        """
        Delete a snapshot.
        
        Args:
            snapshot_id: ID of the snapshot to delete
        """
        snapshot_dir = self.snapshots_path / snapshot_id
        
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
            logger.info(f"Deleted snapshot directory: {snapshot_id}")
        
        # Update metadata
        self.metadata['snapshots'] = [
            s for s in self.metadata['snapshots'] 
            if s['id'] != snapshot_id
        ]
        self._save_metadata()
    
    def save_cache(self, key: str, data: Any) -> Path:
        """
        Save data to cache.
        
        Args:
            key: Cache key
            data: Data to cache
        
        Returns:
            Path to cached file
        """
        # Hash the key for filename
        cache_filename = hashlib.md5(key.encode()).hexdigest() + '.pkl'
        cache_file = self.cache_path / cache_filename
        
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        
        logger.debug(f"Cached data with key: {key}")
        return cache_file
    
    def load_cache(self, key: str) -> Optional[Any]:
        """
        Load data from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached data or None if not found
        """
        cache_filename = hashlib.md5(key.encode()).hexdigest() + '.pkl'
        cache_file = self.cache_path / cache_filename
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                logger.debug(f"Loaded cached data for key: {key}")
                return data
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
                return None
        
        return None
    
    def clear_cache(self):
        """Clear all cached data"""
        cache_files = list(self.cache_path.glob("*.pkl"))
        
        for cache_file in cache_files:
            try:
                cache_file.unlink()
            except Exception as e:
                logger.error(f"Error deleting cache file {cache_file}: {e}")
        
        logger.info(f"Cleared {len(cache_files)} cache files")
    
    def export_data(self, 
                   df: pd.DataFrame,
                   filename: str,
                   format: str = 'csv') -> Path:
        """
        Export data to a specific format for download.
        
        Args:
            df: DataFrame to export
            filename: Name for the export file
            format: Export format ('csv', 'excel', 'json')
        
        Returns:
            Path to the exported file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'csv':
            export_file = self.exports_path / f"{filename}_{timestamp}.csv"
            df.to_csv(export_file, index=False)
        elif format == 'excel':
            export_file = self.exports_path / f"{filename}_{timestamp}.xlsx"
            df.to_excel(export_file, index=False, engine='openpyxl')
        elif format == 'json':
            export_file = self.exports_path / f"{filename}_{timestamp}.json"
            df.to_json(export_file, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported data to {export_file}")
        return export_file
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about storage usage.
        
        Returns:
            Dictionary with storage statistics
        """
        def get_directory_size(path: Path) -> float:
            """Calculate total size of a directory in MB"""
            total_size = 0
            for file in path.rglob('*'):
                if file.is_file():
                    total_size += file.stat().st_size
            return total_size / (1024 * 1024)
        
        stats = {
            'total_size_mb': get_directory_size(self.base_path),
            'processed_size_mb': get_directory_size(self.processed_path),
            'snapshots_size_mb': get_directory_size(self.snapshots_path),
            'cache_size_mb': get_directory_size(self.cache_path),
            'exports_size_mb': get_directory_size(self.exports_path),
            'snapshot_count': len(self.metadata.get('snapshots', [])),
            'total_calls_processed': self.metadata.get('total_calls_processed', 0),
            'last_modified': self.metadata.get('last_modified', 'Unknown')
        }
        
        return stats
    
    def cleanup_old_files(self, days: int = 30):
        """
        Clean up files older than specified days.
        
        Args:
            days: Number of days to keep files
        """
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        deleted_count = 0
        
        # Clean up processed files
        for file in self.processed_path.glob('*'):
            if file.is_file() and file.stat().st_mtime < cutoff_date:
                try:
                    file.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Error deleting file {file}: {e}")
        
        # Clean up exports
        for file in self.exports_path.glob('*'):
            if file.is_file() and file.stat().st_mtime < cutoff_date:
                try:
                    file.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Error deleting file {file}: {e}")
        
        logger.info(f"Cleaned up {deleted_count} old files")
        
        # Clear old cache
        self.clear_cache()