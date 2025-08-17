"""
Test Suite for CSV Processor Module

Tests for CSV file processing, field mapping, and data validation
in the Call Analytics System.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from src.core.csv_processor import CSVProcessor, CSVExporter
from src.core.data_schema import CallRecord
from tests.test_core import CORE_TEST_DATA_DIR, CORE_TEST_OUTPUT_DIR


class TestCSVProcessor(unittest.TestCase):
    """Test cases for CSVProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.processor = CSVProcessor()
        
        # Create sample CSV data
        self.sample_data = pd.DataFrame({
            'phone': ['+1234567890', '+0987654321', '+1122334455'],
            'date': ['2024-01-01 10:00:00', '2024-01-01 11:00:00', '2024-01-01 12:00:00'],
            'duration': [300, 450, 180],
            'outcome': ['connected', 'voicemail', 'no_answer'],
            'agent': ['agent_001', 'agent_002', 'agent_003'],
            'campaign': ['sales', 'support', 'billing'],
            'revenue': [100.0, 0.0, 50.0]
        })
        
        self.csv_path = self.temp_dir / 'test_calls.csv'
        self.sample_data.to_csv(self.csv_path, index=False)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_auto_map_fields(self):
        """Test automatic field mapping from CSV headers."""
        headers = ['phone', 'date', 'duration', 'outcome', 'agent', 'campaign', 'revenue']
        mapping = self.processor.auto_map_fields(headers)
        
        self.assertEqual(mapping['phone_number'], 'phone')
        self.assertEqual(mapping['timestamp'], 'date')
        self.assertEqual(mapping['duration'], 'duration')
        self.assertEqual(mapping['agent_id'], 'agent')
        self.assertEqual(mapping['outcome'], 'outcome')
        self.assertEqual(mapping['campaign'], 'campaign')
        self.assertEqual(mapping['revenue'], 'revenue')
    
    def test_process_csv_file(self):
        """Test processing a CSV file into CallRecord objects."""
        records = list(self.processor.process_csv_file(self.csv_path))
        
        self.assertEqual(len(records), 3)
        self.assertIsInstance(records[0], CallRecord)
        self.assertEqual(records[0].phone_number, '+1234567890')
        self.assertEqual(records[0].duration, 300)
        self.assertEqual(records[0].outcome, 'connected')
    
    def test_validate_csv_structure(self):
        """Test CSV structure validation."""
        # Test valid CSV
        self.assertTrue(self.processor.validate_structure(self.csv_path))
        
        # Test CSV missing required fields
        invalid_data = pd.DataFrame({'random_column': [1, 2, 3]})
        invalid_path = self.temp_dir / 'invalid.csv'
        invalid_data.to_csv(invalid_path, index=False)
        
        self.assertFalse(self.processor.validate_structure(invalid_path))
    
    def test_handle_missing_values(self):
        """Test handling missing values in CSV."""
        # Create CSV with missing values
        data_with_nulls = self.sample_data.copy()
        data_with_nulls.loc[0, 'agent'] = np.nan
        data_with_nulls.loc[1, 'revenue'] = np.nan
        
        null_csv_path = self.temp_dir / 'null_data.csv'
        data_with_nulls.to_csv(null_csv_path, index=False)
        
        # Process file
        records = list(self.processor.process_csv_file(null_csv_path))
        
        # Check handling of missing values
        self.assertEqual(records[0].agent_id, 'unassigned')
        self.assertEqual(records[1].revenue, 0.0)
    
    def test_detect_encoding(self):
        """Test automatic encoding detection."""
        # Create CSV with UTF-8 encoding
        utf8_path = self.temp_dir / 'utf8.csv'
        self.sample_data.to_csv(utf8_path, index=False, encoding='utf-8')
        
        encoding = self.processor.detect_encoding(utf8_path)
        self.assertIn(encoding.lower(), ['utf-8', 'ascii'])
        
        # Create CSV with Latin-1 encoding
        latin1_data = self.sample_data.copy()
        latin1_data.loc[0, 'agent'] = 'agént_001'  # Add accented character
        latin1_path = self.temp_dir / 'latin1.csv'
        latin1_data.to_csv(latin1_path, index=False, encoding='latin-1')
        
        encoding = self.processor.detect_encoding(latin1_path)
        self.assertIsNotNone(encoding)
    
    def test_process_large_csv_batch(self):
        """Test batch processing of large CSV files."""
        # Create large CSV
        large_data = pd.DataFrame({
            'phone': [f'+1{i:09d}' for i in range(1000)],
            'date': pd.date_range('2024-01-01', periods=1000, freq='H').astype(str),
            'duration': np.random.randint(30, 600, 1000),
            'outcome': np.random.choice(['connected', 'no_answer'], 1000),
            'agent': [f'agent_{i%10:03d}' for i in range(1000)]
        })
        
        large_csv_path = self.temp_dir / 'large_data.csv'
        large_data.to_csv(large_csv_path, index=False)
        
        # Process in batches
        batch_count = 0
        def batch_callback(batch):
            nonlocal batch_count
            batch_count += 1
            self.assertLessEqual(len(batch), self.processor.chunk_size)
        
        total_processed, total_errors = self.processor.process_csv_batch(
            large_csv_path,
            batch_callback=batch_callback
        )
        
        self.assertEqual(total_processed, 1000)
        self.assertGreater(batch_count, 0)
    
    def test_custom_delimiter(self):
        """Test processing CSV with custom delimiter."""
        # Create tab-delimited file
        tsv_data = self.sample_data.copy()
        tsv_path = self.temp_dir / 'test_data.tsv'
        tsv_data.to_csv(tsv_path, sep='\t', index=False)
        
        # Process with custom delimiter
        processor = CSVProcessor(delimiter='\t')
        records = list(processor.process_csv_file(tsv_path))
        
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0].phone_number, '+1234567890')


class TestCSVExporter(unittest.TestCase):
    """Test cases for CSVExporter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.exporter = CSVExporter()
        
        # Create sample CallRecord objects
        self.sample_records = [
            CallRecord(
                call_id=f'CALL_{i:03d}',
                phone_number=f'+123456789{i}',
                call_type='inbound',
                outcome='connected',
                duration=300 + i * 10,
                timestamp=datetime(2024, 1, 1, 10, i),
                agent_id=f'agent_{i:03d}',
                campaign='test',
                notes=f'Test call {i}',
                revenue=100.0 * i
            )
            for i in range(5)
        ]
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_export_call_records(self):
        """Test exporting CallRecord objects to CSV."""
        output_path = self.temp_dir / 'exported_calls.csv'
        
        self.exporter.export_call_records(
            self.sample_records,
            output_path
        )
        
        # Verify export
        self.assertTrue(output_path.exists())
        
        # Read exported data
        df = pd.read_csv(output_path)
        self.assertEqual(len(df), 5)
        self.assertIn('call_id', df.columns)
        self.assertIn('phone_number', df.columns)
        self.assertEqual(df.iloc[0]['call_id'], 'CALL_000')
    
    def test_export_with_column_selection(self):
        """Test exporting specific columns only."""
        output_path = self.temp_dir / 'selected_columns.csv'
        
        self.exporter.export_call_records(
            self.sample_records,
            output_path,
            columns=['call_id', 'phone_number', 'duration']
        )
        
        # Read exported data
        df = pd.read_csv(output_path)
        self.assertEqual(len(df.columns), 3)
        self.assertIn('call_id', df.columns)
        self.assertIn('phone_number', df.columns)
        self.assertIn('duration', df.columns)
        self.assertNotIn('revenue', df.columns)
    
    def test_export_analytics_summary(self):
        """Test exporting analytics summary to CSV."""
        analytics_data = {
            'call_metrics': {
                'total_calls': 1000,
                'average_duration': 250.5,
                'connection_rate': 0.85
            },
            'agent_performance': {
                'top_agent': 'agent_001',
                'average_calls_per_agent': 50
            },
            'revenue_metrics': {
                'total_revenue': 50000,
                'average_revenue_per_call': 50
            }
        }
        
        output_path = self.temp_dir / 'analytics_summary.csv'
        self.exporter.export_analytics_summary(analytics_data, output_path)
        
        # Verify export
        self.assertTrue(output_path.exists())
        
        # Read exported data
        df = pd.read_csv(output_path)
        self.assertIn('Category', df.columns)
        self.assertIn('Metric', df.columns)
        self.assertIn('Value', df.columns)
        self.assertGreater(len(df), 0)
    
    def test_custom_date_format(self):
        """Test exporting with custom date format."""
        exporter = CSVExporter(date_format='%Y/%m/%d %I:%M %p')
        output_path = self.temp_dir / 'custom_date.csv'
        
        exporter.export_call_records(
            self.sample_records,
            output_path
        )
        
        # Read and verify date format
        df = pd.read_csv(output_path)
        # Check that timestamp follows the custom format
        self.assertIn('/', df.iloc[0]['timestamp'])
        self.assertTrue(any(x in df.iloc[0]['timestamp'] for x in ['AM', 'PM']))


if __name__ == '__main__':
    unittest.main()