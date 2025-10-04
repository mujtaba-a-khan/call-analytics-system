"""
Aggregations Module for Call Analytics System

This module provides comprehensive aggregation functions for calculating
KPIs, metrics, and statistical summaries from call data.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class AggregationResult:
    """Container for aggregation results with metadata"""
    value: Any
    count: int
    confidence: float
    metadata: Dict[str, Any]


class MetricsCalculator:
    """
    Calculate various metrics and KPIs from call data.
    Provides methods for common call center metrics.
    """
    
    def __init__(self):
        """Initialize the metrics calculator"""
        self.metrics_cache = {}
        logger.info("MetricsCalculator initialized")
    
    def calculate_basic_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate basic call metrics.
        
        Args:
            df: DataFrame with call data
            
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}
        
        # Total calls
        metrics['total_calls'] = len(df)
        
        # Connection rate
        if 'outcome' in df.columns:
            connected = df['outcome'].str.lower() == 'connected'
            metrics['connection_rate'] = (connected.sum() / len(df) * 100) if len(df) > 0 else 0
            metrics['connected_calls'] = connected.sum()
        
        # Average duration
        if 'duration' in df.columns:
            metrics['avg_duration'] = df['duration'].mean()
            metrics['total_duration'] = df['duration'].sum()
            metrics['median_duration'] = df['duration'].median()
        
        # Revenue metrics
        if 'revenue' in df.columns:
            metrics['total_revenue'] = df['revenue'].sum()
            metrics['avg_revenue'] = df['revenue'].mean()
            metrics['revenue_per_minute'] = (
                metrics['total_revenue'] / (metrics['total_duration'] / 60)
                if metrics.get('total_duration', 0) > 0 else 0
            )
        
        return metrics
    
    def calculate_agent_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate per-agent performance metrics.
        
        Args:
            df: DataFrame with call data
            
        Returns:
            DataFrame with agent metrics
        """
        if 'agent_id' not in df.columns:
            return pd.DataFrame()
        
        agent_metrics = df.groupby('agent_id').agg({
            'call_id': 'count',
            'duration': ['mean', 'sum'] if 'duration' in df.columns else [],
            'revenue': ['sum', 'mean'] if 'revenue' in df.columns else [],
            'outcome': lambda x: (x == 'connected').mean() * 100 if 'outcome' in df.columns else 0
        })
        
        agent_metrics.columns = ['_'.join(col).strip() for col in agent_metrics.columns.values]
        agent_metrics = agent_metrics.rename(columns={
            'call_id_count': 'total_calls',
            'outcome_<lambda>': 'connection_rate'
        })
        
        return agent_metrics.sort_values('total_calls', ascending=False)
    
    def calculate_hourly_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate call distribution by hour of day.
        
        Args:
            df: DataFrame with call data
            
        Returns:
            DataFrame with hourly distribution
        """
        if 'timestamp' not in df.columns:
            return pd.DataFrame()
        
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        
        hourly = df.groupby('hour').agg({
            'call_id': 'count',
            'outcome': lambda x: (x == 'connected').mean() * 100 if 'outcome' in df.columns else 0,
            'duration': 'mean' if 'duration' in df.columns else None,
            'revenue': 'sum' if 'revenue' in df.columns else None
        }).dropna(axis=1, how='all')
        
        hourly = hourly.rename(columns={
            'call_id': 'calls',
            'outcome': 'connection_rate',
            'duration': 'avg_duration',
            'revenue': 'total_revenue'
        })
        
        return hourly
    
    def calculate_campaign_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate per-campaign performance metrics.
        
        Args:
            df: DataFrame with call data
            
        Returns:
            DataFrame with campaign metrics
        """
        if 'campaign' not in df.columns:
            return pd.DataFrame()
        
        campaign_metrics = df.groupby('campaign').agg({
            'call_id': 'count',
            'outcome': lambda x: (x == 'connected').mean() * 100 if 'outcome' in df.columns else 0,
            'duration': 'mean' if 'duration' in df.columns else None,
            'revenue': ['sum', 'mean'] if 'revenue' in df.columns else [],
        }).dropna(axis=1, how='all')
        
        # Flatten column names
        campaign_metrics.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                   for col in campaign_metrics.columns.values]
        
        campaign_metrics = campaign_metrics.rename(columns={
            'call_id': 'total_calls',
            'outcome': 'connection_rate',
            'duration': 'avg_duration'
        })
        
        return campaign_metrics.sort_values('total_calls', ascending=False)
    
    def calculate_trends(self, df: pd.DataFrame, period: str = 'daily') -> pd.DataFrame:
        """
        Calculate time-based trends.
        
        Args:
            df: DataFrame with call data
            period: Aggregation period ('daily', 'weekly', 'monthly')
            
        Returns:
            DataFrame with trend data
        """
        if 'timestamp' not in df.columns:
            return pd.DataFrame()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Set grouping key based on period
        if period == 'daily':
            df['period'] = df['timestamp'].dt.date
        elif period == 'weekly':
            df['period'] = df['timestamp'].dt.to_period('W')
        elif period == 'monthly':
            df['period'] = df['timestamp'].dt.to_period('M')
        else:
            raise ValueError(f"Invalid period: {period}")
        
        trends = df.groupby('period').agg({
            'call_id': 'count',
            'outcome': lambda x: (x == 'connected').mean() * 100 if 'outcome' in df.columns else 0,
            'duration': 'mean' if 'duration' in df.columns else None,
            'revenue': 'sum' if 'revenue' in df.columns else None
        }).dropna(axis=1, how='all')
        
        trends = trends.rename(columns={
            'call_id': 'calls',
            'outcome': 'connection_rate',
            'duration': 'avg_duration',
            'revenue': 'total_revenue'
        })
        
        return trends
    
    def calculate_outcome_distribution(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate distribution of call outcomes.
        
        Args:
            df: DataFrame with call data
            
        Returns:
            Series with outcome counts
        """
        if 'outcome' not in df.columns:
            return pd.Series()
        
        return df['outcome'].value_counts()
    
    def calculate_percentiles(self, df: pd.DataFrame, column: str, 
                            percentiles: List[int] = [25, 50, 75, 90, 95]) -> Dict[str, float]:
        """
        Calculate percentiles for a numeric column.
        
        Args:
            df: DataFrame with call data
            column: Column name to calculate percentiles for
            percentiles: List of percentiles to calculate
            
        Returns:
            Dictionary of percentile values
        """
        if column not in df.columns:
            return {}
        
        result = {}
        for p in percentiles:
            result[f'p{p}'] = df[column].quantile(p / 100)
        
        return result
    
    def calculate_conversion_funnel(self, df: pd.DataFrame, 
                                   stages: List[str] = None) -> pd.DataFrame:
        """
        Calculate conversion funnel metrics.
        
        Args:
            df: DataFrame with call data
            stages: List of outcome stages in order
            
        Returns:
            DataFrame with funnel metrics
        """
        if 'outcome' not in df.columns:
            return pd.DataFrame()
        
        if stages is None:
            stages = ['attempted', 'connected', 'qualified', 'converted']
        
        funnel_data = []
        total = len(df)
        
        for stage in stages:
            count = len(df[df['outcome'] == stage])
            percentage = (count / total * 100) if total > 0 else 0
            funnel_data.append({
                'stage': stage,
                'count': count,
                'percentage': percentage,
                'conversion_rate': (count / total * 100) if total > 0 else 0
            })
            total = count  # Update total for next stage
        
        return pd.DataFrame(funnel_data)
    
    def calculate_comparative_metrics(self, current_df: pd.DataFrame, 
                                     previous_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comparative metrics between two periods.
        
        Args:
            current_df: Current period DataFrame
            previous_df: Previous period DataFrame
            
        Returns:
            Dictionary with comparative metrics
        """
        current_metrics = self.calculate_basic_metrics(current_df)
        previous_metrics = self.calculate_basic_metrics(previous_df)
        
        comparison = {}
        for key in current_metrics:
            if key in previous_metrics:
                current_val = current_metrics[key]
                previous_val = previous_metrics[key]
                
                if isinstance(current_val, (int, float)) and previous_val != 0:
                    change = current_val - previous_val
                    change_pct = (change / previous_val) * 100
                    comparison[key] = {
                        'current': current_val,
                        'previous': previous_val,
                        'change': change,
                        'change_percentage': change_pct
                    }
        
        return comparison