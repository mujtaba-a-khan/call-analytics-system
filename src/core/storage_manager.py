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
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, date
import shutil
import hashlib

logger = logging.getLogger(__name__)


class StorageManager:
    """
    Manages persistent storage of call analytics data.
    Supports multiple formats and provides versioning capabilities.
    """
    
    def __init__(self, base_path: Union[str, Path] = "data"):
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
        
        # Cache for loaded data
        self._data_cache = None
        self._cache_timestamp = None
        
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
            'storage_format': 'parquet',
            'primary_file': 'call_records.parquet'
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
            Path to saved file
        """
        # Determine file path
        if format == 'parquet':
            file_path = self.processed_path / f"{name}.parquet"
            df.to_parquet(file_path, index=False)
        elif format == 'csv':
            file_path = self.processed_path / f"{name}.csv"
            df.to_csv(file_path, index=False)
        elif format == 'pickle':
            file_path = self.processed_path / f"{name}.pkl"
            df.to_pickle(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Create snapshot if requested
        if create_snapshot:
            self.create_snapshot(df, name)
        
        # Update metadata
        self.metadata['total_calls_processed'] = len(df)
        self.metadata['primary_file'] = file_path.name
        self._save_metadata()
        
        # Clear cache since data has changed
        self._data_cache = None
        
        logger.info(f"Saved {len(df)} records to {file_path}")
        return file_path
    
    def load_dataframe(self, name: str, format: str = 'parquet') -> Optional[pd.DataFrame]:
        """
        Load a DataFrame from storage.
        
        Args:
            name: Name of the file to load
            format: Storage format
            
        Returns:
            Loaded DataFrame or None if not found
        """
        try:
            if format == 'parquet':
                file_path = self.processed_path / f"{name}.parquet"
                if file_path.exists():
                    return pd.read_parquet(file_path)
            elif format == 'csv':
                file_path = self.processed_path / f"{name}.csv"
                if file_path.exists():
                    return pd.read_csv(file_path)
            elif format == 'pickle':
                file_path = self.processed_path / f"{name}.pkl"
                if file_path.exists():
                    return pd.read_pickle(file_path)
            
            logger.warning(f"File not found: {name}.{format}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading dataframe: {e}")
            return None
    
    def load_all_records(self) -> pd.DataFrame:
        """
        Load all call records from storage.
        
        Returns:
            DataFrame with all records or empty DataFrame if none found
        """
        # Check cache first
        if self._data_cache is not None:
            cache_age = (datetime.now() - self._cache_timestamp).seconds
            if cache_age < 300:  # Cache for 5 minutes
                return self._data_cache.copy()
        
        # Try to load the primary file
        primary_file = self.metadata.get('primary_file', 'call_records.parquet')
        
        # Try different formats
        for format in ['parquet', 'csv', 'pickle']:
            name = primary_file.rsplit('.', 1)[0] if '.' in primary_file else primary_file
            df = self.load_dataframe(name, format)
            if df is not None:
                # Update cache
                self._data_cache = df
                self._cache_timestamp = datetime.now()
                return df.copy()
        
        # If no file found, check for any data files
        for file_path in self.processed_path.glob('*.parquet'):
            try:
                df = pd.read_parquet(file_path)
                self._data_cache = df
                self._cache_timestamp = datetime.now()
                return df.copy()
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        logger.warning("No call records found in storage")
        return pd.DataFrame()
    
    def load_call_records(self, 
                         start_date: Optional[Union[datetime, date]] = None,
                         end_date: Optional[Union[datetime, date]] = None) -> pd.DataFrame:
        """
        Load call records filtered by date range.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            Filtered DataFrame
        """
        # Load all records
        df = self.load_all_records()
        
        if df.empty:
            return df
        
        # Check if timestamp column exists
        if 'timestamp' not in df.columns:
            logger.warning("No timestamp column found in data")
            return df
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Apply date filters
        if start_date is not None:
            if isinstance(start_date, date) and not isinstance(start_date, datetime):
                start_date = datetime.combine(start_date, datetime.min.time())
            df = df[df['timestamp'] >= start_date]
        
        if end_date is not None:
            if isinstance(end_date, date) and not isinstance(end_date, datetime):
                end_date = datetime.combine(end_date, datetime.max.time())
            df = df[df['timestamp'] <= end_date]
        
        logger.info(f"Loaded {len(df)} records for date range {start_date} to {end_date}")
        return df
    
    def get_unique_values(self, column: str) -> List[Any]:
        """
        Get unique values for a specific column.
        
        Args:
            column: Column name
            
        Returns:
            List of unique values
        """
        df = self.load_all_records()
        
        if df.empty or column not in df.columns:
            logger.warning(f"Column '{column}' not found in data")
            return []
        
        unique_values = df[column].dropna().unique().tolist()
        logger.debug(f"Found {len(unique_values)} unique values for column '{column}'")
        return sorted(unique_values)
    
    def get_available_fields(self) -> List[str]:
        """
        Get list of available fields/columns in the data.
        
        Returns:
            List of column names
        """
        df = self.load_all_records()
        
        if df.empty:
            logger.warning("No data available to get fields from")
            return []
        
        fields = df.columns.tolist()
        logger.debug(f"Available fields: {fields}")
        return fields
    
    def get_record_count(self) -> int:
        """
        Get total number of records in storage.
        
        Returns:
            Number of records
        """
        df = self.load_all_records()
        return len(df)
    
    def get_date_range(self) -> Optional[Tuple[datetime, datetime]]:
        """
        Get the date range of available data.
        
        Returns:
            Tuple of (min_date, max_date) or None if no data
        """
        df = self.load_all_records()
        
        if df.empty or 'timestamp' not in df.columns:
            logger.warning("No timestamp data available")
            return None
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        min_date = df['timestamp'].min()
        max_date = df['timestamp'].max()
        
        return (min_date, max_date)
    
    def append_records(self, new_df: pd.DataFrame, deduplicate: bool = True) -> int:
        """
        Append new records to existing data.
        
        Args:
            new_df: DataFrame with new records
            deduplicate: Whether to remove duplicates
            
        Returns:
            Number of records added
        """
        existing_df = self.load_all_records()
        
        # Combine dataframes
        if existing_df.empty:
            combined_df = new_df
        else:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Remove duplicates if requested
        if deduplicate and 'call_id' in combined_df.columns:
            original_count = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['call_id'], keep='last')
            duplicates_removed = original_count - len(combined_df)
            if duplicates_removed > 0:
                logger.info(f"Removed {duplicates_removed} duplicate records")
        
        # Save the combined data
        self.save_dataframe(combined_df, 'call_records')
        
        records_added = len(combined_df) - len(existing_df)
        logger.info(f"Added {records_added} new records")
        return records_added
    
    def create_snapshot(self, df: pd.DataFrame, name: str) -> Path:
        """
        Create a snapshot of the current data.
        
        Args:
            df: DataFrame to snapshot
            name: Name for the snapshot
            
        Returns:
            Path to snapshot file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        snapshot_name = f"{name}_{timestamp}"
        snapshot_path = self.snapshots_path / f"{snapshot_name}.parquet"
        
        df.to_parquet(snapshot_path, index=False)
        
        # Update metadata
        self.metadata['snapshots'].append({
            'name': snapshot_name,
            'path': str(snapshot_path),
            'created_at': datetime.now().isoformat(),
            'record_count': len(df)
        })
        self._save_metadata()
        
        logger.info(f"Created snapshot: {snapshot_name}")
        return snapshot_path
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """
        List available snapshots.
        
        Returns:
            List of snapshot metadata
        """
        return self.metadata.get('snapshots', [])
    
    def load_snapshot(self, snapshot_name: str) -> Optional[pd.DataFrame]:
        """
        Load a specific snapshot.
        
        Args:
            snapshot_name: Name of the snapshot
            
        Returns:
            DataFrame or None if not found
        """
        for snapshot in self.metadata.get('snapshots', []):
            if snapshot['name'] == snapshot_name:
                snapshot_path = Path(snapshot['path'])
                if snapshot_path.exists():
                    return pd.read_parquet(snapshot_path)
        
        logger.warning(f"Snapshot not found: {snapshot_name}")
        return None
    
    def export_data(self, df: pd.DataFrame, filename: str, format: str = 'csv') -> Path:
        """
        Export data to the exports directory.
        
        Args:
            df: DataFrame to export
            filename: Export filename
            format: Export format
            
        Returns:
            Path to exported file
        """
        if format == 'csv':
            export_path = self.exports_path / f"{filename}.csv"
            df.to_csv(export_path, index=False)
        elif format == 'excel':
            export_path = self.exports_path / f"{filename}.xlsx"
            df.to_excel(export_path, index=False)
        elif format == 'json':
            export_path = self.exports_path / f"{filename}.json"
            df.to_json(export_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported {len(df)} records to {export_path}")
        return export_path
    
    def clear_cache(self):
        """Clear the internal data cache"""
        self._data_cache = None
        self._cache_timestamp = None
        logger.info("Data cache cleared")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        stats = {
            'total_records': self.get_record_count(),
            'snapshots_count': len(self.metadata.get('snapshots', [])),
            'last_modified': self.metadata.get('last_modified'),
            'storage_format': self.metadata.get('storage_format', 'parquet')
        }
        
        # Calculate storage size
        total_size = 0
        for path in [self.processed_path, self.snapshots_path]:
            for file in path.glob('*'):
                if file.is_file():
                    total_size += file.stat().st_size
        
        stats['storage_size_mb'] = round(total_size / (1024 * 1024), 2)
        
        # Get date range
        date_range = self.get_date_range()
        if date_range:
            stats['earliest_date'] = date_range[0].isoformat()
            stats['latest_date'] = date_range[1].isoformat()
        
        return stats
    
    def clear_import_history(self):
        """Clear import history from metadata"""
        if 'import_history' in self.metadata:
            self.metadata['import_history'] = []
            self._save_metadata()
            logger.info("Import history cleared")
    
    def add_import_record(self, record: Dict[str, Any]):
        """
        Add an import record to history.

        Args:
            record: Import record dictionary
        """
        if 'import_history' not in self.metadata:
            self.metadata['import_history'] = []

        # Ensure values are JSON serializable (e.g., convert datetimes)
        serializable_record = {}
        for key, value in record.items():
            if isinstance(value, (datetime, date)):
                serializable_record[key] = value.isoformat()
            else:
                serializable_record[key] = value

        self.metadata['import_history'].append(serializable_record)
        self._save_metadata()
        logger.info(f"Added import record: {record.get('filename', 'unknown')}")
    
    def get_import_history(self) -> pd.DataFrame:
        """
        Get import history as a DataFrame.
        
        Returns:
            DataFrame with import history
        """
        history = self.metadata.get('import_history', [])
        if history:
            return pd.DataFrame(history)
        return pd.DataFrame()
