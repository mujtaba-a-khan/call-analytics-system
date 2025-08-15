"""
CSV Processing Module

Handles importing, validation, and processing of CSV files
containing call transcripts and metadata.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import chardet
import json

logger = logging.getLogger(__name__)


class CSVProcessor:
    """
    Processes CSV files containing call data.
    Handles various formats and encodings, validates data,
    and standardizes output format.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the CSV processor with configuration.
        
        Args:
            config: Configuration dictionary with CSV settings
        """
        self.encoding = config.get('encoding', 'utf-8')
        self.required_columns = config.get('required_columns', [
            'call_id', 'start_time', 'duration_seconds', 'transcript'
        ])
        self.optional_columns = config.get('optional_columns', [
            'agent_id', 'campaign', 'customer_name', 
            'product_name', 'amount', 'order_id'
        ])
        self.date_formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
            '%m/%d/%Y %H:%M:%S',
            '%m/%d/%Y',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ'
        ]
        
        logger.info("CSVProcessor initialized")
    
    def detect_encoding(self, file_path: Path) -> str:
        """
        Detect the encoding of a CSV file.
        
        Args:
            file_path: Path to the CSV file
        
        Returns:
            Detected encoding string
        """
        try:
            with open(file_path, 'rb') as f:
                # Read first 10000 bytes for detection
                raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                
                logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
                
                # Fall back to utf-8 if confidence is low
                if confidence < 0.7:
                    logger.warning("Low confidence in encoding detection, using UTF-8")
                    return 'utf-8'
                
                return encoding
        except Exception as e:
            logger.error(f"Error detecting encoding: {e}")
            return 'utf-8'
    
    def read_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Read a CSV file with automatic encoding detection.
        
        Args:
            file_path: Path to the CSV file
        
        Returns:
            DataFrame with the CSV data
        """
        # Detect encoding
        encoding = self.detect_encoding(file_path)
        
        try:
            # Try reading with detected encoding
            df = pd.read_csv(file_path, encoding=encoding)
            logger.info(f"Successfully read CSV with {len(df)} rows")
            return df
        except UnicodeDecodeError:
            # Fallback to different encodings
            encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
            for enc in encodings_to_try:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    logger.info(f"Successfully read CSV with {enc} encoding")
                    return df
                except UnicodeDecodeError:
                    continue
            
            # If all fail, read with error handling
            logger.warning("Reading CSV with error handling")
            return pd.read_csv(file_path, encoding='utf-8', errors='ignore')
    
    def validate_columns(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that required columns are present.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            Tuple of (is_valid, missing_columns)
        """
        missing_columns = []
        
        for col in self.required_columns:
            if col not in df.columns:
                missing_columns.append(col)
        
        is_valid = len(missing_columns) == 0
        
        if not is_valid:
            logger.warning(f"Missing required columns: {missing_columns}")
        else:
            logger.info("All required columns present")
        
        return is_valid, missing_columns
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names and add missing optional columns.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with standardized columns
        """
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Standardize column names (lowercase, replace spaces with underscores)
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Map common variations to standard names
        column_mapping = {
            'id': 'call_id',
            'callid': 'call_id',
            'call': 'call_id',
            'timestamp': 'start_time',
            'datetime': 'start_time',
            'date': 'start_time',
            'duration': 'duration_seconds',
            'length': 'duration_seconds',
            'agent': 'agent_id',
            'rep': 'agent_id',
            'text': 'transcript',
            'transcription': 'transcript',
            'customer': 'customer_name',
            'product': 'product_name',
            'value': 'amount',
            'price': 'amount'
        }
        
        # Apply mapping
        df.rename(columns=column_mapping, inplace=True)
        
        # Add missing optional columns with None values
        for col in self.optional_columns:
            if col not in df.columns:
                df[col] = None
        
        logger.info(f"Standardized {len(df.columns)} columns")
        return df
    
    def parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse date columns to datetime format.
        
        Args:
            df: DataFrame with date columns
        
        Returns:
            DataFrame with parsed dates
        """
        if 'start_time' not in df.columns:
            logger.warning("No start_time column to parse")
            return df
        
        # Try different date formats
        for date_format in self.date_formats:
            try:
                df['start_time'] = pd.to_datetime(df['start_time'], format=date_format)
                logger.info(f"Successfully parsed dates with format: {date_format}")
                return df
            except (ValueError, TypeError):
                continue
        
        # If no format works, use pandas automatic parsing
        try:
            df['start_time'] = pd.to_datetime(df['start_time'], infer_datetime_format=True)
            logger.info("Dates parsed using automatic inference")
        except Exception as e:
            logger.error(f"Failed to parse dates: {e}")
            # Set to current time as fallback
            df['start_time'] = datetime.now()
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the data.
        
        Args:
            df: DataFrame to clean
        
        Returns:
            Cleaned DataFrame
        """
        # Remove duplicate rows
        original_len = len(df)
        df = df.drop_duplicates(subset=['call_id'], keep='first')
        if len(df) < original_len:
            logger.info(f"Removed {original_len - len(df)} duplicate rows")
        
        # Clean transcript text
        if 'transcript' in df.columns:
            # Remove extra whitespace
            df['transcript'] = df['transcript'].str.strip()
            df['transcript'] = df['transcript'].str.replace(r'\s+', ' ', regex=True)
            
            # Remove null transcripts
            df = df[df['transcript'].notna()]
            df = df[df['transcript'] != '']
        
        # Convert duration to numeric
        if 'duration_seconds' in df.columns:
            df['duration_seconds'] = pd.to_numeric(df['duration_seconds'], errors='coerce')
            df = df[df['duration_seconds'] > 0]
        
        # Convert amount to numeric
        if 'amount' in df.columns:
            # Remove currency symbols and convert
            df['amount'] = df['amount'].astype(str).str.replace(r'[$,]', '', regex=True)
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Standardize agent IDs
        if 'agent_id' in df.columns:
            df['agent_id'] = df['agent_id'].astype(str).str.strip().str.upper()
        
        # Standardize campaign names
        if 'campaign' in df.columns:
            df['campaign'] = df['campaign'].astype(str).str.strip()
        
        logger.info(f"Cleaned data: {len(df)} rows remaining")
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and return metrics.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            Dictionary with quality metrics
        """
        metrics = {
            'total_rows': len(df),
            'missing_values': {},
            'data_types': {},
            'value_ranges': {},
            'quality_score': 100.0
        }
        
        # Check missing values
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            metrics['missing_values'][col] = {
                'count': int(missing_count),
                'percentage': round(missing_pct, 2)
            }
            
            # Deduct from quality score for missing required columns
            if col in self.required_columns and missing_pct > 0:
                metrics['quality_score'] -= min(missing_pct, 20)
        
        # Check data types
        for col in df.columns:
            metrics['data_types'][col] = str(df[col].dtype)
        
        # Check value ranges for numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            metrics['value_ranges'][col] = {
                'min': float(df[col].min()) if not df[col].isna().all() else None,
                'max': float(df[col].max()) if not df[col].isna().all() else None,
                'mean': float(df[col].mean()) if not df[col].isna().all() else None
            }
        
        # Check transcript quality
        if 'transcript' in df.columns:
            avg_length = df['transcript'].str.len().mean()
            if avg_length < 100:
                metrics['quality_score'] -= 10
                logger.warning(f"Short transcripts detected: avg length {avg_length:.0f}")
        
        metrics['quality_score'] = max(0, metrics['quality_score'])
        
        return metrics
    
    def process_file(self, file_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process a single CSV file end-to-end.
        
        Args:
            file_path: Path to the CSV file
        
        Returns:
            Tuple of (processed_dataframe, quality_metrics)
        """
        logger.info(f"Processing CSV file: {file_path}")
        
        # Read the file
        df = self.read_csv(file_path)
        
        # Standardize columns
        df = self.standardize_columns(df)
        
        # Validate required columns
        is_valid, missing = self.validate_columns(df)
        if not is_valid:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Parse dates
        df = self.parse_dates(df)
        
        # Clean data
        df = self.clean_data(df)
        
        # Validate quality
        metrics = self.validate_data_quality(df)
        
        logger.info(f"Processing complete: {len(df)} rows, quality score: {metrics['quality_score']:.1f}")
        
        return df, metrics
    
    def process_multiple_files(self, file_paths: List[Path]) -> pd.DataFrame:
        """
        Process multiple CSV files and combine them.
        
        Args:
            file_paths: List of paths to CSV files
        
        Returns:
            Combined DataFrame
        """
        all_dfs = []
        all_metrics = []
        
        for file_path in file_paths:
            try:
                df, metrics = self.process_file(file_path)
                all_dfs.append(df)
                all_metrics.append(metrics)
                logger.info(f"Successfully processed: {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                continue
        
        if not all_dfs:
            logger.warning("No files were successfully processed")
            return pd.DataFrame()
        
        # Combine all DataFrames
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Remove duplicates across files
        original_len = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['call_id'], keep='first')
        if len(combined_df) < original_len:
            logger.info(f"Removed {original_len - len(combined_df)} duplicates across files")
        
        # Save metrics summary
        self._save_processing_summary(all_metrics)
        
        logger.info(f"Combined {len(file_paths)} files: {len(combined_df)} total rows")
        return combined_df
    
    def _save_processing_summary(self, metrics_list: List[Dict[str, Any]]):
        """
        Save a summary of processing metrics.
        
        Args:
            metrics_list: List of metrics dictionaries
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'files_processed': len(metrics_list),
            'total_rows': sum(m['total_rows'] for m in metrics_list),
            'average_quality_score': sum(m['quality_score'] for m in metrics_list) / len(metrics_list) if metrics_list else 0,
            'individual_metrics': metrics_list
        }
        
        summary_path = Path('data/processing_summary.json')
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Processing summary saved to {summary_path}")
    
    def export_to_standard_format(self, df: pd.DataFrame, output_path: Path):
        """
        Export DataFrame to standard CSV format.
        
        Args:
            df: DataFrame to export
            output_path: Path for output file
        """
        # Ensure all required columns are present
        export_columns = self.required_columns + [
            col for col in self.optional_columns if col in df.columns
        ]
        
        # Export with standard column order
        df[export_columns].to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Exported {len(df)} rows to {output_path}")