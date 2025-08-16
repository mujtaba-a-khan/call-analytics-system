"""
Test Suite for Analysis Modules

This module contains comprehensive tests for analysis functionality including
metrics calculation, semantic search, query interpretation, and filtering.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json

# Import modules to test
from src.analysis import (
    MetricsCalculator,
    SemanticSearchEngine,
    QueryInterpreter,
    AdvancedFilters
)
from src.analysis.aggregations import CallMetrics, AgentMetrics
from tests import TEST_DATA_DIR, TEST_OUTPUT_DIR


class TestMetricsCalculator(unittest.TestCase):
    """Test cases for metrics calculation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample call data
        self.sample_data = pd.DataFrame({
            'call_id': [f'CALL_{i:03d}' for i in range(100)],
            'phone_number': [f'+123456789{i%10}' for i in range(100)],
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
            'duration': np.random.randint(30, 600, 100),
            'outcome': np.random.choice(['connected', 'no_answer', 'voicemail', 'busy'], 100),
            'agent_id': [f'agent_{i%5:03d}' for i in range(100)],
            'campaign': np.random.choice(['sales', 'support', 'billing'], 100),
            'revenue': np.random.choice([0, 0, 0, 100, 200], 100),
            'call_type': np.random.choice(['inbound', 'outbound'], 100)
        })
        
        self.calculator = MetricsCalculator()
    
    def test_calculate_basic_metrics(self):
        """Test basic metrics calculation."""
        metrics = self.calculator.calculate_basic_metrics(self.sample_data)
        
        self.assertIn('total_calls', metrics)
        self.assertIn('average_duration', metrics)
        self.assertIn('connection_rate', metrics)
        self.assertIn('total_revenue', metrics)
        
        self.assertEqual(metrics['total_calls'], 100)
        self.assertIsInstance(metrics['average_duration'], float)
        self.assertGreaterEqual(metrics['connection_rate'], 0)
        self.assertLessEqual(metrics['connection_rate'], 100)
    
    def test_calculate_agent_metrics(self):
        """Test agent-specific metrics calculation."""
        agent_metrics = self.calculator.calculate_agent_metrics(self.sample_data)
        
        self.assertIsInstance(agent_metrics, pd.DataFrame)
        self.assertIn('total_calls', agent_metrics.columns)
        self.assertIn('avg_duration', agent_metrics.columns)
        self.assertIn('connection_rate', agent_metrics.columns)
        
        # Should have 5 unique agents
        self.assertEqual(len(agent_metrics), 5)
    
    def test_calculate_campaign_metrics(self):
        """Test campaign-specific metrics calculation."""
        campaign_metrics = self.calculator.calculate_campaign_metrics(self.sample_data)
        
        self.assertIsInstance(campaign_metrics, pd.DataFrame)
        self.assertIn('total_calls', campaign_metrics.columns)
        self.assertIn('total_revenue', campaign_metrics.columns)
        
        # Should have 3 campaigns
        self.assertEqual(len(campaign_metrics), 3)
    
    def test_calculate_time_series_metrics(self):
        """Test time series metrics calculation."""
        # Calculate hourly metrics
        hourly_metrics = self.calculator.calculate_time_series_metrics(
            self.sample_data,
            frequency='H'
        )
        
        self.assertIsInstance(hourly_metrics, pd.DataFrame)
        self.assertIn('call_volume', hourly_metrics.columns)
        self.assertEqual(len(hourly_metrics), 100)  # One per hour
        
        # Calculate daily metrics
        daily_metrics = self.calculator.calculate_time_series_metrics(
            self.sample_data,
            frequency='D'
        )
        
        self.assertLess(len(daily_metrics), len(hourly_metrics))
    
    def test_calculate_peak_hours(self):
        """Test peak hours analysis."""
        peak_hours = self.calculator.calculate_peak_hours(self.sample_data)
        
        self.assertIsInstance(peak_hours, pd.DataFrame)
        self.assertIn('hour', peak_hours.columns)
        self.assertIn('call_count', peak_hours.columns)
        
        # Should have up to 24 hours
        self.assertLessEqual(len(peak_hours), 24)
    
    def test_calculate_conversion_metrics(self):
        """Test conversion and revenue metrics."""
        conversion_metrics = self.calculator.calculate_conversion_metrics(self.sample_data)
        
        self.assertIn('conversion_rate', conversion_metrics)
        self.assertIn('average_revenue_per_call', conversion_metrics)
        self.assertIn('revenue_calls_count', conversion_metrics)
        
        self.assertGreaterEqual(conversion_metrics['conversion_rate'], 0)
        self.assertLessEqual(conversion_metrics['conversion_rate'], 100)


class TestSemanticSearchEngine(unittest.TestCase):
    """Test cases for semantic search functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock vector store
        self.mock_vector_store = MagicMock()
        
        # Create search engine
        self.search_engine = SemanticSearchEngine(self.mock_vector_store)
        
        # Sample documents
        self.sample_docs = [
            {
                'id': 'doc1',
                'content': 'Customer complained about billing issue',
                'metadata': {'call_id': 'CALL_001', 'agent_id': 'agent_001'}
            },
            {
                'id': 'doc2',
                'content': 'Technical support for product installation',
                'metadata': {'call_id': 'CALL_002', 'agent_id': 'agent_002'}
            },
            {
                'id': 'doc3',
                'content': 'Sales inquiry about pricing plans',
                'metadata': {'call_id': 'CALL_003', 'agent_id': 'agent_003'}
            }
        ]
    
    def test_search_basic(self):
        """Test basic semantic search."""
        # Setup mock search results
        mock_results = {
            'documents': [['Customer complained about billing issue']],
            'metadatas': [[{'call_id': 'CALL_001'}]],
            'distances': [[0.2]]
        }
        self.mock_vector_store.query.return_value = mock_results
        
        # Perform search
        results = self.search_engine.search(
            query='billing problem',
            top_k=5
        )
        
        # Verify results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['call_id'], 'CALL_001')
        self.assertIn('score', results[0])
        
        # Verify search was called
        self.mock_vector_store.query.assert_called_once()
    
    def test_search_with_filters(self):
        """Test semantic search with metadata filters."""
        # Setup mock
        mock_results = {
            'documents': [['Technical support']],
            'metadatas': [[{'call_id': 'CALL_002', 'agent_id': 'agent_002'}]],
            'distances': [[0.3]]
        }
        self.mock_vector_store.query.return_value = mock_results
        
        # Search with filters
        results = self.search_engine.search(
            query='technical issue',
            top_k=5,
            filters={'agent_id': 'agent_002'}
        )
        
        # Verify filter was applied
        call_args = self.mock_vector_store.query.call_args
        self.assertIn('where', call_args[1])
        self.assertEqual(call_args[1]['where']['agent_id'], 'agent_002')
    
    def test_search_threshold_filtering(self):
        """Test search with similarity threshold."""
        # Setup mock with multiple results
        mock_results = {
            'documents': [['Doc1', 'Doc2', 'Doc3']],
            'metadatas': [[
                {'call_id': 'CALL_001'},
                {'call_id': 'CALL_002'},
                {'call_id': 'CALL_003'}
            ]],
            'distances': [[0.1, 0.5, 0.9]]  # Different similarity scores
        }
        self.mock_vector_store.query.return_value = mock_results
        
        # Search with threshold
        results = self.search_engine.search(
            query='test query',
            top_k=10,
            threshold=0.6  # Should filter out the last result
        )
        
        # Only results with distance < 0.6 should be returned
        self.assertEqual(len(results), 2)
    
    def test_batch_search(self):
        """Test batch search functionality."""
        # Setup mock
        self.mock_vector_store.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        # Batch search
        queries = ['query1', 'query2', 'query3']
        results = self.search_engine.batch_search(queries, top_k=5)
        
        # Should return results for each query
        self.assertEqual(len(results), 3)
        self.assertEqual(self.mock_vector_store.query.call_count, 3)


class TestQueryInterpreter(unittest.TestCase):
    """Test cases for natural language query interpretation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.interpreter = QueryInterpreter()
    
    def test_interpret_metric_query(self):
        """Test interpreting metric-related queries."""
        # Test average duration query
        query = "What is the average call duration last week?"
        intent = self.interpreter.interpret(query)
        
        self.assertEqual(intent['type'], 'metric')
        self.assertEqual(intent['metric'], 'average_duration')
        self.assertEqual(intent['time_range'], 'last_week')
        
        # Test total calls query
        query = "How many calls did we have today?"
        intent = self.interpreter.interpret(query)
        
        self.assertEqual(intent['type'], 'metric')
        self.assertEqual(intent['metric'], 'total_calls')
        self.assertEqual(intent['time_range'], 'today')
    
    def test_interpret_search_query(self):
        """Test interpreting search queries."""
        query = "Find calls about billing issues"
        intent = self.interpreter.interpret(query)
        
        self.assertEqual(intent['type'], 'search')
        self.assertIn('billing', intent['search_terms'])
        
        query = "Show me customer complaints from yesterday"
        intent = self.interpreter.interpret(query)
        
        self.assertEqual(intent['type'], 'search')
        self.assertIn('complaint', intent['search_terms'])
        self.assertEqual(intent['time_range'], 'yesterday')
    
    def test_interpret_comparison_query(self):
        """Test interpreting comparison queries."""
        query = "Compare this week's performance to last week"
        intent = self.interpreter.interpret(query)
        
        self.assertEqual(intent['type'], 'comparison')
        self.assertEqual(intent['compare'], 'periods')
        self.assertIn('this_week', intent['periods'])
        self.assertIn('last_week', intent['periods'])
    
    def test_interpret_agent_query(self):
        """Test interpreting agent-related queries."""
        query = "Show me top performing agents"
        intent = self.interpreter.interpret(query)
        
        self.assertEqual(intent['type'], 'analysis')
        self.assertEqual(intent['analysis_type'], 'performance')
        self.assertEqual(intent['entity'], 'agent')
    
    def test_extract_entities(self):
        """Test entity extraction from queries."""
        # Test date extraction
        query = "Show calls from January 15 to January 20"
        entities = self.interpreter.extract_entities(query)
        
        self.assertIn('date_range', entities)
        
        # Test agent extraction
        query = "How many calls did agent_001 handle?"
        entities = self.interpreter.extract_entities(query)
        
        self.assertIn('agent_id', entities)
        self.assertEqual(entities['agent_id'], 'agent_001')
    
    def test_handle_ambiguous_queries(self):
        """Test handling ambiguous queries."""
        query = "Show me the data"
        intent = self.interpreter.interpret(query)
        
        self.assertEqual(intent['type'], 'general')
        self.assertIn('clarification_needed', intent)


class TestAdvancedFilters(unittest.TestCase):
    """Test cases for advanced filtering functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.filters = AdvancedFilters()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'call_id': [f'CALL_{i:03d}' for i in range(100)],
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
            'duration': np.random.randint(30, 600, 100),
            'outcome': np.random.choice(['connected', 'no_answer', 'voicemail'], 100),
            'agent_id': [f'agent_{i%5:03d}' for i in range(100)],
            'revenue': np.random.uniform(0, 500, 100),
            'notes': [''] * 100
        })
    
    def test_date_range_filter(self):
        """Test date range filtering."""
        # Filter for specific date range
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 2)
        
        filtered = self.filters.apply_date_range(
            self.sample_data,
            start_date,
            end_date
        )
        
        # Should have ~24 hours of data
        self.assertLess(len(filtered), len(self.sample_data))
        self.assertGreater(len(filtered), 0)
    
    def test_outcome_filter(self):
        """Test outcome filtering."""
        # Filter for connected calls only
        filtered = self.filters.apply_outcome_filter(
            self.sample_data,
            outcomes=['connected']
        )
        
        # All results should be connected
        self.assertTrue(all(filtered['outcome'] == 'connected'))
        self.assertLess(len(filtered), len(self.sample_data))
    
    def test_duration_range_filter(self):
        """Test duration range filtering."""
        # Filter for calls between 1-5 minutes
        filtered = self.filters.apply_duration_range(
            self.sample_data,
            min_duration=60,
            max_duration=300
        )
        
        # All durations should be in range
        self.assertTrue(all(filtered['duration'] >= 60))
        self.assertTrue(all(filtered['duration'] <= 300))
    
    def test_revenue_filter(self):
        """Test revenue filtering."""
        # Filter for revenue > 100
        filtered = self.filters.apply_revenue_filter(
            self.sample_data,
            min_revenue=100
        )
        
        # All revenue should be >= 100
        self.assertTrue(all(filtered['revenue'] >= 100))
    
    def test_text_search_filter(self):
        """Test text search in notes/transcripts."""
        # Add some notes with keywords
        self.sample_data.loc[0, 'notes'] = 'Customer complained about billing'
        self.sample_data.loc[1, 'notes'] = 'Technical support issue resolved'
        self.sample_data.loc[2, 'notes'] = 'Billing inquiry handled'
        
        # Search for billing
        filtered = self.filters.apply_text_search(
            self.sample_data,
            search_query='billing'
        )
        
        # Should find the billing-related calls
        self.assertEqual(len(filtered), 2)
    
    def test_complex_filter_chain(self):
        """Test chaining multiple filters."""
        # Apply multiple filters
        filtered = self.sample_data.copy()
        
        # Date range
        filtered = self.filters.apply_date_range(
            filtered,
            datetime(2024, 1, 1),
            datetime(2024, 1, 3)
        )
        
        # Outcome
        filtered = self.filters.apply_outcome_filter(
            filtered,
            outcomes=['connected', 'voicemail']
        )
        
        # Duration
        filtered = self.filters.apply_duration_range(
            filtered,
            min_duration=60,
            max_duration=400
        )
        
        # Should have filtered data
        self.assertLess(len(filtered), len(self.sample_data))
        
        # Verify all filters applied
        self.assertTrue(all(filtered['outcome'].isin(['connected', 'voicemail'])))
        self.assertTrue(all(filtered['duration'] >= 60))
        self.assertTrue(all(filtered['duration'] <= 400))
    
    def test_filter_with_aggregation(self):
        """Test filtering with aggregation."""
        # Filter and aggregate
        filtered = self.filters.apply_outcome_filter(
            self.sample_data,
            outcomes=['connected']
        )
        
        # Aggregate by agent
        aggregated = filtered.groupby('agent_id').agg({
            'call_id': 'count',
            'duration': 'mean',
            'revenue': 'sum'
        })
        
        self.assertIsInstance(aggregated, pd.DataFrame)
        self.assertEqual(len(aggregated), 5)  # 5 unique agents


class TestTimeSeriesAnalysis(unittest.TestCase):
    """Test cases for time series analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create time series data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        self.time_series_data = pd.DataFrame({
            'date': dates,
            'call_volume': np.random.poisson(100, 30),
            'avg_duration': np.random.normal(300, 50, 30),
            'connection_rate': np.random.uniform(0.7, 0.95, 30)
        })
        
        self.calculator = MetricsCalculator()
    
    def test_calculate_moving_average(self):
        """Test moving average calculation."""
        # Calculate 7-day moving average
        ma_result = self.calculator.calculate_moving_average(
            self.time_series_data,
            column='call_volume',
            window=7
        )
        
        self.assertEqual(len(ma_result), len(self.time_series_data))
        # First 6 values should be NaN
        self.assertTrue(ma_result[:6].isna().all())
        # Rest should have values
        self.assertFalse(ma_result[6:].isna().any())
    
    def test_detect_trends(self):
        """Test trend detection in time series."""
        # Create trending data
        trending_data = self.time_series_data.copy()
        trending_data['call_volume'] = range(30)  # Increasing trend
        
        trend = self.calculator.detect_trend(
            trending_data,
            column='call_volume'
        )
        
        self.assertEqual(trend, 'increasing')
        
        # Test decreasing trend
        trending_data['call_volume'] = range(29, -1, -1)
        trend = self.calculator.detect_trend(
            trending_data,
            column='call_volume'
        )
        
        self.assertEqual(trend, 'decreasing')
    
    def test_calculate_seasonality(self):
        """Test seasonality detection."""
        # Create data with weekly seasonality
        seasonal_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=28, freq='D'),
            'call_volume': [100 + 20 * np.sin(2 * np.pi * i / 7) for i in range(28)]
        })
        
        seasonality = self.calculator.detect_seasonality(
            seasonal_data,
            column='call_volume',
            period=7
        )
        
        self.assertTrue(seasonality['has_seasonality'])
        self.assertEqual(seasonality['period'], 7)


def suite():
    """Create test suite for analysis modules."""
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestMetricsCalculator))
    suite.addTest(unittest.makeSuite(TestSemanticSearchEngine))
    suite.addTest(unittest.makeSuite(TestQueryInterpreter))
    suite.addTest(unittest.makeSuite(TestAdvancedFilters))
    suite.addTest(unittest.makeSuite(TestTimeSeriesAnalysis))
    
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())