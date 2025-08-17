"""
Test Suite for Filters Module

Tests for advanced filtering functionality including date ranges,
numeric filters, and complex filter chains in the Call Analytics System.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date

from src.analysis.filters import (
    AdvancedFilters,
    FilterCriteria,
    FilterOperator,
    DateRangeFilter,
    NumericRangeFilter
)
from tests.test_analysis import create_sample_call_data, ANALYSIS_TEST_DATA_DIR


class TestAdvancedFilters(unittest.TestCase):
    """Test cases for AdvancedFilters class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.filters = AdvancedFilters()
        self.sample_data = create_sample_call_data(200)
        
        # Add some specific test data
        self.sample_data.loc[0, 'notes'] = 'Customer complained about billing issue'
        self.sample_data.loc[1, 'notes'] = 'Technical support resolved successfully'
        self.sample_data.loc[2, 'notes'] = 'Billing inquiry about monthly charges'
    
    def test_apply_single_filter(self):
        """Test applying a single filter criteria."""
        # Filter for connected calls only
        criteria = FilterCriteria(
            field='outcome',
            operator=FilterOperator.EQUALS,
            value='connected'
        )
        
        filtered = self.filters.apply_filter(self.sample_data, criteria)
        
        # All results should be connected
        self.assertTrue(all(filtered['outcome'] == 'connected'))
        self.assertLess(len(filtered), len(self.sample_data))
    
    def test_apply_multiple_filters(self):
        """Test applying multiple filter criteria."""
        criteria_list = [
            FilterCriteria('outcome', FilterOperator.EQUALS, 'connected'),
            FilterCriteria('duration', FilterOperator.GREATER_THAN, 120),
            FilterCriteria('campaign', FilterOperator.IN, ['sales', 'support'])
        ]
        
        filtered = self.filters.apply_filters(self.sample_data, criteria_list)
        
        # Verify all criteria are met
        self.assertTrue(all(filtered['outcome'] == 'connected'))
        self.assertTrue(all(filtered['duration'] > 120))
        self.assertTrue(all(filtered['campaign'].isin(['sales', 'support'])))
    
    def test_date_range_filter(self):
        """Test date range filtering."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 3)
        
        filtered = self.filters.apply_date_range(
            self.sample_data,
            start_date,
            end_date
        )
        
        # Check all timestamps are within range
        timestamps = pd.to_datetime(filtered['timestamp'])
        self.assertTrue(all(timestamps >= start_date))
        self.assertTrue(all(timestamps <= end_date))
    
    def test_outcome_filter(self):
        """Test outcome filtering with multiple values."""
        outcomes = ['connected', 'voicemail']
        
        filtered = self.filters.apply_outcome_filter(
            self.sample_data,
            outcomes=outcomes
        )
        
        # All outcomes should be in the specified list
        self.assertTrue(all(filtered['outcome'].isin(outcomes)))
    
    def test_duration_range_filter(self):
        """Test duration range filtering."""
        min_duration = 60
        max_duration = 300
        
        filtered = self.filters.apply_duration_range(
            self.sample_data,
            min_duration=min_duration,
            max_duration=max_duration
        )
        
        # All durations should be within range
        self.assertTrue(all(filtered['duration'] >= min_duration))
        self.assertTrue(all(filtered['duration'] <= max_duration))
    
    def test_agent_filter(self):
        """Test agent filtering."""
        agents = ['agent_001', 'agent_002', 'agent_003']
        
        filtered = self.filters.apply_agent_filter(
            self.sample_data,
            agents=agents
        )
        
        # All agents should be in the specified list
        self.assertTrue(all(filtered['agent_id'].isin(agents)))
    
    def test_campaign_filter(self):
        """Test campaign filtering."""
        campaigns = ['sales', 'support']
        
        filtered = self.filters.apply_campaign_filter(
            self.sample_data,
            campaigns=campaigns
        )
        
        # All campaigns should be in the specified list
        self.assertTrue(all(filtered['campaign'].isin(campaigns)))
    
    def test_revenue_filter(self):
        """Test revenue filtering."""
        min_revenue = 50
        max_revenue = 200
        
        filtered = self.filters.apply_revenue_filter(
            self.sample_data,
            min_revenue=min_revenue,
            max_revenue=max_revenue
        )
        
        # All revenue should be within range
        self.assertTrue(all(filtered['revenue'] >= min_revenue))
        self.assertTrue(all(filtered['revenue'] <= max_revenue))
    
    def test_text_search_filter(self):
        """Test text search in notes and transcripts."""
        # Search for 'billing'
        filtered = self.filters.apply_text_search(
            self.sample_data,
            search_query='billing',
            search_fields=['notes']
        )
        
        # Should find the billing-related calls
        self.assertEqual(len(filtered), 2)
        for _, row in filtered.iterrows():
            self.assertIn('billing', row['notes'].lower())
    
    def test_complex_filter_chain(self):
        """Test chaining multiple filters together."""
        # Apply multiple filters in sequence
        result = self.sample_data.copy()
        
        # Date range
        result = self.filters.apply_date_range(
            result,
            datetime(2024, 1, 1),
            datetime(2024, 1, 5)
        )
        
        # Outcome filter
        result = self.filters.apply_outcome_filter(
            result,
            outcomes=['connected', 'voicemail']
        )
        
        # Duration filter
        result = self.filters.apply_duration_range(
            result,
            min_duration=60
        )
        
        # Verify result is properly filtered
        self.assertLess(len(result), len(self.sample_data))
        
        # Verify all filters are applied
        timestamps = pd.to_datetime(result['timestamp'])
        self.assertTrue(all(timestamps >= datetime(2024, 1, 1)))
        self.assertTrue(all(timestamps <= datetime(2024, 1, 5)))
        self.assertTrue(all(result['outcome'].isin(['connected', 'voicemail'])))
        self.assertTrue(all(result['duration'] >= 60))
    
    def test_filter_with_null_values(self):
        """Test filtering with null values in data."""
        # Add some null values
        data_with_nulls = self.sample_data.copy()
        data_with_nulls.loc[0, 'agent_id'] = None
        data_with_nulls.loc[1, 'campaign'] = None
        
        # Apply filters
        filtered = self.filters.apply_agent_filter(
            data_with_nulls,
            agents=['agent_001', 'agent_002']
        )
        
        # Should handle nulls gracefully
        self.assertLess(len(filtered), len(data_with_nulls))
        self.assertFalse(filtered['agent_id'].isna().any())
    
    def test_empty_filter_results(self):
        """Test handling of filters that return empty results."""
        # Apply impossible filter
        filtered = self.filters.apply_outcome_filter(
            self.sample_data,
            outcomes=['nonexistent_outcome']
        )
        
        self.assertEqual(len(filtered), 0)
        self.assertIsInstance(filtered, pd.DataFrame)
        self.assertTrue(filtered.empty)


class TestFilterCriteria(unittest.TestCase):
    """Test cases for FilterCriteria class."""
    
    def test_criteria_creation(self):
        """Test creating FilterCriteria instance."""
        criteria = FilterCriteria(
            field='duration',
            operator=FilterOperator.GREATER_THAN,
            value=300
        )
        
        self.assertEqual(criteria.field, 'duration')
        self.assertEqual(criteria.operator, FilterOperator.GREATER_THAN)
        self.assertEqual(criteria.value, 300)
    
    def test_criteria_apply(self):
        """Test applying filter criteria to data."""
        data = pd.DataFrame({
            'value': [1, 2, 3, 4, 5]
        })
        
        # Test EQUALS
        criteria = FilterCriteria('value', FilterOperator.EQUALS, 3)
        result = criteria.apply(data)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['value'], 3)
        
        # Test GREATER_THAN
        criteria = FilterCriteria('value', FilterOperator.GREATER_THAN, 3)
        result = criteria.apply(data)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(result['value'] > 3))
        
        # Test LESS_THAN
        criteria = FilterCriteria('value', FilterOperator.LESS_THAN, 3)
        result = criteria.apply(data)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(result['value'] < 3))
    
    def test_criteria_in_operator(self):
        """Test IN operator for filtering."""
        data = pd.DataFrame({
            'category': ['A', 'B', 'C', 'D', 'A', 'B']
        })
        
        criteria = FilterCriteria('category', FilterOperator.IN, ['A', 'C'])
        result = criteria.apply(data)
        
        self.assertEqual(len(result), 3)
        self.assertTrue(all(result['category'].isin(['A', 'C'])))
    
    def test_criteria_contains_operator(self):
        """Test CONTAINS operator for text search."""
        data = pd.DataFrame({
            'text': ['hello world', 'goodbye world', 'hello there', 'test']
        })
        
        criteria = FilterCriteria('text', FilterOperator.CONTAINS, 'hello')
        result = criteria.apply(data)
        
        self.assertEqual(len(result), 2)
        for _, row in result.iterrows():
            self.assertIn('hello', row['text'])
    
    def test_criteria_validation(self):
        """Test filter criteria validation."""
        # Test invalid field name
        with self.assertRaises(ValueError):
            FilterCriteria('', FilterOperator.EQUALS, 'value')
        
        # Test None value for non-null operators
        with self.assertRaises(ValueError):
            FilterCriteria('field', FilterOperator.GREATER_THAN, None)


class TestDateRangeFilter(unittest.TestCase):
    """Test cases for DateRangeFilter class."""
    
    def test_date_range_creation(self):
        """Test creating DateRangeFilter instance."""
        filter = DateRangeFilter(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31)
        )
        
        self.assertEqual(filter.start_date, date(2024, 1, 1))
        self.assertEqual(filter.end_date, date(2024, 1, 31))
    
    def test_date_range_apply(self):
        """Test applying date range filter."""
        data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='D')
        })
        
        filter = DateRangeFilter(
            start_date=date(2024, 1, 3),
            end_date=date(2024, 1, 7)
        )
        
        result = filter.apply(data, date_column='date')
        
        self.assertEqual(len(result), 5)
        self.assertTrue(all(result['date'] >= pd.Timestamp('2024-01-03')))
        self.assertTrue(all(result['date'] <= pd.Timestamp('2024-01-07')))
    
    def test_date_range_validation(self):
        """Test date range validation."""
        # Test end date before start date
        with self.assertRaises(ValueError):
            DateRangeFilter(
                start_date=date(2024, 1, 31),
                end_date=date(2024, 1, 1)
            )
    
    def test_date_range_with_time(self):
        """Test date range filter with datetime objects."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01 00:00', periods=48, freq='H')
        })
        
        filter = DateRangeFilter(
            start_date=datetime(2024, 1, 1, 12, 0),
            end_date=datetime(2024, 1, 2, 12, 0)
        )
        
        result = filter.apply(data, date_column='timestamp')
        
        self.assertEqual(len(result), 25)  # 24 hours + 1


class TestNumericRangeFilter(unittest.TestCase):
    """Test cases for NumericRangeFilter class."""
    
    def test_numeric_range_creation(self):
        """Test creating NumericRangeFilter instance."""
        filter = NumericRangeFilter(
            min_value=10,
            max_value=100
        )
        
        self.assertEqual(filter.min_value, 10)
        self.assertEqual(filter.max_value, 100)
    
    def test_numeric_range_apply(self):
        """Test applying numeric range filter."""
        data = pd.DataFrame({
            'value': range(0, 200, 10)
        })
        
        filter = NumericRangeFilter(
            min_value=50,
            max_value=150
        )
        
        result = filter.apply(data, column='value')
        
        self.assertTrue(all(result['value'] >= 50))
        self.assertTrue(all(result['value'] <= 150))
    
    def test_numeric_range_with_nulls(self):
        """Test numeric range filter with null values."""
        data = pd.DataFrame({
            'value': [10, 20, None, 40, 50, None, 70]
        })
        
        filter = NumericRangeFilter(
            min_value=20,
            max_value=60
        )
        
        result = filter.apply(data, column='value')
        
        # Nulls should be excluded
        self.assertFalse(result['value'].isna().any())
        self.assertTrue(all(result['value'] >= 20))
        self.assertTrue(all(result['value'] <= 60))
    
    def test_numeric_range_validation(self):
        """Test numeric range validation."""
        # Test max less than min
        with self.assertRaises(ValueError):
            NumericRangeFilter(
                min_value=100,
                max_value=10
            )
    
    def test_open_ended_range(self):
        """Test open-ended numeric ranges."""
        data = pd.DataFrame({
            'value': range(0, 100, 10)
        })
        
        # Test min only
        filter = NumericRangeFilter(min_value=50)
        result = filter.apply(data, column='value')
        self.assertTrue(all(result['value'] >= 50))
        
        # Test max only
        filter = NumericRangeFilter(max_value=50)
        result = filter.apply(data, column='value')
        self.assertTrue(all(result['value'] <= 50))


class TestFilterOperator(unittest.TestCase):
    """Test cases for FilterOperator enum."""
    
    def test_operator_values(self):
        """Test FilterOperator enum values."""
        self.assertEqual(FilterOperator.EQUALS.value, 'equals')
        self.assertEqual(FilterOperator.NOT_EQUALS.value, 'not_equals')
        self.assertEqual(FilterOperator.GREATER_THAN.value, 'greater_than')
        self.assertEqual(FilterOperator.LESS_THAN.value, 'less_than')
        self.assertEqual(FilterOperator.IN.value, 'in')
        self.assertEqual(FilterOperator.NOT_IN.value, 'not_in')
        self.assertEqual(FilterOperator.CONTAINS.value, 'contains')
        self.assertEqual(FilterOperator.NOT_CONTAINS.value, 'not_contains')
    
    def test_operator_apply_logic(self):
        """Test operator application logic."""
        data = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        
        # Test each operator
        operators_tests = [
            (FilterOperator.EQUALS, 3, [3]),
            (FilterOperator.NOT_EQUALS, 3, [1, 2, 4, 5]),
            (FilterOperator.GREATER_THAN, 3, [4, 5]),
            (FilterOperator.LESS_THAN, 3, [1, 2]),
            (FilterOperator.GREATER_THAN_OR_EQUAL, 3, [3, 4, 5]),
            (FilterOperator.LESS_THAN_OR_EQUAL, 3, [1, 2, 3]),
        ]
        
        for operator, value, expected_results in operators_tests:
            criteria = FilterCriteria('value', operator, value)
            result = criteria.apply(data)
            self.assertEqual(result['value'].tolist(), expected_results)


if __name__ == '__main__':
    unittest.main()