"""
Test Suite for Aggregations Module

Tests for metrics calculation, time series analysis, and statistical
aggregations in the Call Analytics System.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.analysis.aggregations import (
    MetricsCalculator,
    CallMetrics,
    AgentMetrics,
    CampaignMetrics,
    TimeSeriesMetrics
)
from tests.test_analysis import create_sample_call_data, ANALYSIS_TEST_DATA_DIR


class TestMetricsCalculator(unittest.TestCase):
    """Test cases for MetricsCalculator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = MetricsCalculator()
        self.sample_data = create_sample_call_data(100)
    
    def test_calculate_basic_metrics(self):
        """Test calculation of basic call metrics."""
        metrics = self.calculator.calculate_basic_metrics(self.sample_data)
        
        self.assertIn('total_calls', metrics)
        self.assertIn('average_duration', metrics)
        self.assertIn('connection_rate', metrics)
        self.assertIn('total_revenue', metrics)
        self.assertIn('unique_callers', metrics)
        
        # Verify metric values
        self.assertEqual(metrics['total_calls'], 100)
        self.assertIsInstance(metrics['average_duration'], float)
        self.assertGreaterEqual(metrics['connection_rate'], 0)
        self.assertLessEqual(metrics['connection_rate'], 100)
        self.assertGreaterEqual(metrics['total_revenue'], 0)
    
    def test_calculate_all_metrics(self):
        """Test comprehensive metrics calculation."""
        all_metrics = self.calculator.calculate_all_metrics(self.sample_data)
        
        # Check metric categories
        expected_categories = [
            'basic_metrics',
            'outcome_distribution',
            'time_metrics',
            'agent_performance',
            'campaign_performance'
        ]
        
        for category in expected_categories:
            self.assertIn(category, all_metrics)
            self.assertIsInstance(all_metrics[category], dict)
    
    def test_empty_data_handling(self):
        """Test metrics calculation with empty data."""
        empty_data = pd.DataFrame()
        metrics = self.calculator.calculate_basic_metrics(empty_data)
        
        self.assertEqual(metrics['total_calls'], 0)
        self.assertEqual(metrics['average_duration'], 0)
        self.assertEqual(metrics['connection_rate'], 0)
    
    def test_missing_columns_handling(self):
        """Test metrics calculation with missing columns."""
        # Create data with missing revenue column
        data_no_revenue = self.sample_data.drop('revenue', axis=1)
        metrics = self.calculator.calculate_basic_metrics(data_no_revenue)
        
        # Should handle missing revenue gracefully
        self.assertEqual(metrics['total_revenue'], 0)
        self.assertIn('total_calls', metrics)


class TestCallMetrics(unittest.TestCase):
    """Test cases for CallMetrics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = CallMetrics()
        self.sample_data = create_sample_call_data(200)
    
    def test_calculate_duration_statistics(self):
        """Test duration statistics calculation."""
        duration_stats = self.metrics.calculate_duration_stats(self.sample_data)
        
        self.assertIn('mean', duration_stats)
        self.assertIn('median', duration_stats)
        self.assertIn('std', duration_stats)
        self.assertIn('min', duration_stats)
        self.assertIn('max', duration_stats)
        self.assertIn('percentile_25', duration_stats)
        self.assertIn('percentile_75', duration_stats)
        
        # Verify statistical properties
        self.assertGreaterEqual(duration_stats['min'], 0)
        self.assertLessEqual(duration_stats['min'], duration_stats['median'])
        self.assertLessEqual(duration_stats['median'], duration_stats['max'])
    
    def test_calculate_outcome_distribution(self):
        """Test outcome distribution calculation."""
        outcome_dist = self.metrics.calculate_outcome_distribution(self.sample_data)
        
        # Check all outcomes are included
        expected_outcomes = ['connected', 'no_answer', 'voicemail', 'busy', 'failed']
        for outcome in self.sample_data['outcome'].unique():
            self.assertIn(outcome, outcome_dist)
        
        # Verify percentages sum to 100
        total_percentage = sum(outcome_dist.values())
        self.assertAlmostEqual(total_percentage, 100, places=1)
    
    def test_calculate_peak_hours(self):
        """Test peak hours identification."""
        peak_hours = self.metrics.calculate_peak_hours(self.sample_data)
        
        self.assertIsInstance(peak_hours, pd.DataFrame)
        self.assertIn('hour', peak_hours.columns)
        self.assertIn('call_count', peak_hours.columns)
        
        # Should have at most 24 hours
        self.assertLessEqual(len(peak_hours), 24)
        
        # Verify sorting (highest volume first)
        if len(peak_hours) > 1:
            self.assertGreaterEqual(
                peak_hours.iloc[0]['call_count'],
                peak_hours.iloc[-1]['call_count']
            )
    
    def test_calculate_call_patterns(self):
        """Test call pattern analysis."""
        patterns = self.metrics.calculate_call_patterns(self.sample_data)
        
        self.assertIn('busiest_hour', patterns)
        self.assertIn('busiest_day', patterns)
        self.assertIn('average_calls_per_hour', patterns)
        self.assertIn('average_calls_per_day', patterns)
        
        # Verify busiest hour is valid
        self.assertGreaterEqual(patterns['busiest_hour'], 0)
        self.assertLessEqual(patterns['busiest_hour'], 23)


class TestAgentMetrics(unittest.TestCase):
    """Test cases for AgentMetrics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = AgentMetrics()
        self.sample_data = create_sample_call_data(500)
    
    def test_calculate_agent_performance(self):
        """Test agent performance metrics calculation."""
        agent_perf = self.metrics.calculate_performance(self.sample_data)
        
        self.assertIsInstance(agent_perf, pd.DataFrame)
        
        # Check required columns
        expected_columns = [
            'total_calls',
            'avg_duration',
            'connection_rate',
            'total_revenue',
            'calls_per_day'
        ]
        
        for col in expected_columns:
            self.assertIn(col, agent_perf.columns)
        
        # Verify agent count (should be 10 based on sample data generation)
        self.assertEqual(len(agent_perf), 10)
    
    def test_calculate_agent_rankings(self):
        """Test agent ranking calculation."""
        rankings = self.metrics.calculate_rankings(
            self.sample_data,
            metric='total_calls'
        )
        
        self.assertIsInstance(rankings, pd.DataFrame)
        self.assertIn('rank', rankings.columns)
        self.assertIn('total_calls', rankings.columns)
        
        # Verify rankings are sequential
        expected_ranks = list(range(1, len(rankings) + 1))
        actual_ranks = rankings['rank'].tolist()
        self.assertEqual(actual_ranks, expected_ranks)
    
    def test_calculate_agent_efficiency(self):
        """Test agent efficiency metrics."""
        efficiency = self.metrics.calculate_efficiency(self.sample_data)
        
        self.assertIsInstance(efficiency, pd.DataFrame)
        self.assertIn('calls_per_hour', efficiency.columns)
        self.assertIn('revenue_per_call', efficiency.columns)
        self.assertIn('avg_handle_time', efficiency.columns)
    
    def test_agent_comparison(self):
        """Test agent comparison functionality."""
        agent1 = 'agent_001'
        agent2 = 'agent_002'
        
        comparison = self.metrics.compare_agents(
            self.sample_data,
            agent1,
            agent2
        )
        
        self.assertIn(agent1, comparison)
        self.assertIn(agent2, comparison)
        self.assertIn('difference', comparison)
        
        # Verify difference calculation
        for metric in comparison['difference']:
            self.assertAlmostEqual(
                comparison['difference'][metric],
                comparison[agent1].get(metric, 0) - comparison[agent2].get(metric, 0),
                places=2
            )


class TestCampaignMetrics(unittest.TestCase):
    """Test cases for CampaignMetrics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = CampaignMetrics()
        self.sample_data = create_sample_call_data(300)
    
    def test_calculate_campaign_performance(self):
        """Test campaign performance metrics calculation."""
        campaign_perf = self.metrics.calculate_performance(self.sample_data)
        
        self.assertIsInstance(campaign_perf, pd.DataFrame)
        
        # Check required columns
        expected_columns = [
            'total_calls',
            'connection_rate',
            'total_revenue',
            'conversion_rate',
            'avg_call_value'
        ]
        
        for col in expected_columns:
            self.assertIn(col, campaign_perf.columns)
        
        # Verify campaign count
        unique_campaigns = self.sample_data['campaign'].nunique()
        self.assertEqual(len(campaign_perf), unique_campaigns)
    
    def test_calculate_campaign_roi(self):
        """Test campaign ROI calculation."""
        # Add cost data for ROI calculation
        campaign_costs = {
            'sales': 1000,
            'support': 500,
            'billing': 300,
            'retention': 800
        }
        
        roi = self.metrics.calculate_roi(self.sample_data, campaign_costs)
        
        self.assertIsInstance(roi, pd.DataFrame)
        self.assertIn('cost', roi.columns)
        self.assertIn('revenue', roi.columns)
        self.assertIn('roi_percentage', roi.columns)
        
        # Verify ROI calculation
        for _, row in roi.iterrows():
            if row['cost'] > 0:
                expected_roi = ((row['revenue'] - row['cost']) / row['cost']) * 100
                self.assertAlmostEqual(row['roi_percentage'], expected_roi, places=2)
    
    def test_campaign_trend_analysis(self):
        """Test campaign trend analysis over time."""
        trends = self.metrics.analyze_trends(self.sample_data)
        
        self.assertIsInstance(trends, pd.DataFrame)
        self.assertIn('date', trends.columns)
        
        # Should have columns for each campaign
        for campaign in self.sample_data['campaign'].unique():
            self.assertIn(campaign, trends.columns)


class TestTimeSeriesMetrics(unittest.TestCase):
    """Test cases for TimeSeriesMetrics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = TimeSeriesMetrics()
        # Create time series data with clear pattern
        dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
        self.time_series_data = pd.DataFrame({
            'timestamp': dates,
            'call_volume': 100 + 10 * np.sin(np.arange(90) * 2 * np.pi / 7) + np.random.randn(90) * 5,
            'duration': 300 + np.random.randn(90) * 50,
            'revenue': 1000 + np.random.randn(90) * 100
        })
    
    def test_calculate_moving_average(self):
        """Test moving average calculation."""
        ma_7 = self.metrics.calculate_moving_average(
            self.time_series_data,
            column='call_volume',
            window=7
        )
        
        self.assertEqual(len(ma_7), len(self.time_series_data))
        
        # First 6 values should be NaN
        self.assertTrue(ma_7[:6].isna().all())
        
        # Rest should have values
        self.assertFalse(ma_7[6:].isna().any())
        
        # Moving average should be smoother than original
        original_std = self.time_series_data['call_volume'].std()
        ma_std = ma_7[6:].std()
        self.assertLess(ma_std, original_std)
    
    def test_detect_trend(self):
        """Test trend detection in time series."""
        # Create data with clear upward trend
        trending_data = self.time_series_data.copy()
        trending_data['call_volume'] = range(90)
        
        trend = self.metrics.detect_trend(trending_data, 'call_volume')
        self.assertEqual(trend, 'increasing')
        
        # Create data with downward trend
        trending_data['call_volume'] = range(89, -1, -1)
        trend = self.metrics.detect_trend(trending_data, 'call_volume')
        self.assertEqual(trend, 'decreasing')
        
        # Create stable data
        trending_data['call_volume'] = 100
        trend = self.metrics.detect_trend(trending_data, 'call_volume')
        self.assertEqual(trend, 'stable')
    
    def test_calculate_growth_rate(self):
        """Test growth rate calculation."""
        growth_rate = self.metrics.calculate_growth_rate(
            self.time_series_data,
            column='revenue',
            period='monthly'
        )
        
        self.assertIsInstance(growth_rate, float)
        # Growth rate should be a reasonable percentage
        self.assertGreater(growth_rate, -100)
        self.assertLess(growth_rate, 1000)
    
    def test_forecast_simple(self):
        """Test simple forecasting."""
        forecast = self.metrics.forecast_simple(
            self.time_series_data,
            column='call_volume',
            periods=7
        )
        
        self.assertEqual(len(forecast), 7)
        # Forecast values should be reasonable
        self.assertTrue(all(forecast > 0))
        self.assertTrue(all(forecast < 1000))


if __name__ == '__main__':
    unittest.main()