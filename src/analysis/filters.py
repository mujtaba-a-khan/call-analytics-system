"""
Data Filtering Module for Call Analytics System

This module provides comprehensive filtering capabilities for call data,
including date ranges, call types, outcomes, and custom criteria.
"""

import logging
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

# Configure module logger
logger = logging.getLogger(__name__)


class DataFilter:
    """
    Core filtering class for call data with chainable filter methods.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a DataFrame.
        
        Args:
            df: DataFrame to filter
        """
        self.original_df = df.copy()
        self.filtered_df = df.copy()
        self.active_filters = {}
        logger.info(f"DataFilter initialized with {len(df)} records")
    
    def reset_filters(self) -> 'DataFilter':
        """
        Reset all filters and restore original data.
        
        Returns:
            Self for method chaining
        """
        self.filtered_df = self.original_df.copy()
        self.active_filters = {}
        logger.info("Filters reset")
        return self
    
    def apply_date_range(self, 
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> 'DataFilter':
        """
        Apply date range filter to the data.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
        
        Returns:
            Self for method chaining
        """
        if 'timestamp' not in self.filtered_df.columns:
            logger.warning("No timestamp column found")
            return self
        
        # Convert timestamp column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(self.filtered_df['timestamp']):
            self.filtered_df['timestamp'] = pd.to_datetime(self.filtered_df['timestamp'])
        
        if start_date:
            self.filtered_df = self.filtered_df[self.filtered_df['timestamp'] >= start_date]
            self.active_filters['start_date'] = start_date
        
        if end_date:
            self.filtered_df = self.filtered_df[self.filtered_df['timestamp'] <= end_date]
            self.active_filters['end_date'] = end_date
        
        logger.info(f"Date filter applied: {start_date} to {end_date}")
        return self
    
    def apply_call_type_filter(self, call_types: List[str]) -> 'DataFilter':
        """
        Filter by call types.
        
        Args:
            call_types: List of call types to include
        
        Returns:
            Self for method chaining
        """
        if not call_types:
            return self
        
        if 'call_type' in self.filtered_df.columns:
            self.filtered_df = self.filtered_df[
                self.filtered_df['call_type'].isin(call_types)
            ]
            self.active_filters['call_types'] = call_types
            logger.info(f"Call type filter applied: {call_types}")
        else:
            logger.warning("No call_type column found")
        
        return self
    
    def apply_outcome_filter(self, outcomes: List[str]) -> 'DataFilter':
        """
        Filter by call outcomes.
        
        Args:
            outcomes: List of outcomes to include
        
        Returns:
            Self for method chaining
        """
        if not outcomes:
            return self
        
        if 'outcome' in self.filtered_df.columns:
            self.filtered_df = self.filtered_df[
                self.filtered_df['outcome'].isin(outcomes)
            ]
            self.active_filters['outcomes'] = outcomes
            logger.info(f"Outcome filter applied: {outcomes}")
        else:
            logger.warning("No outcome column found")
        
        return self
    
    def apply_agent_filter(self, agent_ids: List[str]) -> 'DataFilter':
        """
        Filter by agent IDs.
        
        Args:
            agent_ids: List of agent IDs to include
        
        Returns:
            Self for method chaining
        """
        if not agent_ids:
            return self
        
        if 'agent_id' in self.filtered_df.columns:
            self.filtered_df = self.filtered_df[
                self.filtered_df['agent_id'].isin(agent_ids)
            ]
            self.active_filters['agent_ids'] = agent_ids
            logger.info(f"Agent filter applied: {len(agent_ids)} agents")
        else:
            logger.warning("No agent_id column found")
        
        return self
    
    def apply_campaign_filter(self, campaigns: List[str]) -> 'DataFilter':
        """
        Filter by campaigns.
        
        Args:
            campaigns: List of campaigns to include
        
        Returns:
            Self for method chaining
        """
        if not campaigns:
            return self
        
        if 'campaign' in self.filtered_df.columns:
            self.filtered_df = self.filtered_df[
                self.filtered_df['campaign'].isin(campaigns)
            ]
            self.active_filters['campaigns'] = campaigns
            logger.info(f"Campaign filter applied: {campaigns}")
        else:
            logger.warning("No campaign column found")
        
        return self
    
    def apply_duration_filter(self,
                            min_duration: Optional[float] = None,
                            max_duration: Optional[float] = None) -> 'DataFilter':
        """
        Filter by call duration.
        
        Args:
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
        
        Returns:
            Self for method chaining
        """
        if 'duration_seconds' not in self.filtered_df.columns:
            if 'duration' in self.filtered_df.columns:
                duration_col = 'duration'
            else:
                logger.warning("No duration column found")
                return self
        else:
            duration_col = 'duration_seconds'
        
        if min_duration is not None:
            self.filtered_df = self.filtered_df[self.filtered_df[duration_col] >= min_duration]
            self.active_filters['min_duration'] = min_duration
        
        if max_duration is not None:
            self.filtered_df = self.filtered_df[self.filtered_df[duration_col] <= max_duration]
            self.active_filters['max_duration'] = max_duration
        
        logger.info(f"Duration filter applied: {min_duration} to {max_duration} seconds")
        return self
    
    def apply_amount_filter(self,
                          min_amount: Optional[float] = None,
                          max_amount: Optional[float] = None) -> 'DataFilter':
        """
        Filter by transaction amount.
        
        Args:
            min_amount: Minimum amount
            max_amount: Maximum amount
        
        Returns:
            Self for method chaining
        """
        if 'amount' not in self.filtered_df.columns:
            if 'revenue' in self.filtered_df.columns:
                amount_col = 'revenue'
            else:
                logger.warning("No amount/revenue column found")
                return self
        else:
            amount_col = 'amount'
        
        if min_amount is not None:
            self.filtered_df = self.filtered_df[self.filtered_df[amount_col] >= min_amount]
            self.active_filters['min_amount'] = min_amount
        
        if max_amount is not None:
            self.filtered_df = self.filtered_df[self.filtered_df[amount_col] <= max_amount]
            self.active_filters['max_amount'] = max_amount
        
        logger.info(f"Amount filter applied: ${min_amount} to ${max_amount}")
        return self
    
    def apply_text_search(self, 
                         search_query: str,
                         columns: Optional[List[str]] = None) -> 'DataFilter':
        """
        Apply text search across specified columns.
        
        Args:
            search_query: Search query string
            columns: Columns to search in (default: all text columns)
        
        Returns:
            Self for method chaining
        """
        if not search_query:
            return self
        
        # Default to all string columns if not specified
        if columns is None:
            columns = self.filtered_df.select_dtypes(include=['object']).columns.tolist()
        
        # Create mask for search
        mask = pd.Series([False] * len(self.filtered_df))
        
        for col in columns:
            if col in self.filtered_df.columns:
                mask |= self.filtered_df[col].astype(str).str.contains(
                    search_query, 
                    case=False, 
                    na=False,
                    regex=False
                )
        
        self.filtered_df = self.filtered_df[mask]
        self.active_filters['search_query'] = search_query
        self.active_filters['search_columns'] = columns
        
        logger.info(f"Text search applied: '{search_query}' in {columns}")
        return self
    
    def get_filtered_data(self) -> pd.DataFrame:
        """
        Get the filtered DataFrame.
        
        Returns:
            Filtered DataFrame
        """
        return self.filtered_df.copy()
    
    def get_filter_summary(self) -> Dict[str, Any]:
        """
        Get summary of applied filters.
        
        Returns:
            Dictionary containing filter summary
        """
        return {
            'original_count': len(self.original_df),
            'filtered_count': len(self.filtered_df),
            'reduction_percentage': (1 - len(self.filtered_df) / len(self.original_df)) * 100 if len(self.original_df) > 0 else 0,
            'active_filters': self.active_filters,
            'columns_affected': list(self.active_filters.keys())
        }
    
    def export_filtered_data(self, filepath: str, format: str = 'csv'):
        """
        Export filtered data to file.
        
        Args:
            filepath: Path to save the file
            format: Export format ('csv', 'excel', 'parquet')
        """
        if format == 'csv':
            self.filtered_df.to_csv(filepath, index=False)
        elif format == 'excel':
            self.filtered_df.to_excel(filepath, index=False)
        elif format == 'parquet':
            self.filtered_df.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Filtered data exported to {filepath}")


class AdvancedFilters:
    """
    Advanced filtering capabilities with complex logic and
    natural language processing support.
    """
    
    def __init__(self):
        """Initialize advanced filters with pattern matching capabilities"""
        self.query_patterns = self._compile_patterns()
        logger.info("AdvancedFilters initialized")
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for query interpretation"""
        return {
            'date_range': re.compile(r'between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})', re.IGNORECASE),
            'last_n_days': re.compile(r'last\s+(\d+)\s+days?', re.IGNORECASE),
            'call_type': re.compile(r'(inquiry|billing|sales|support|complaint)', re.IGNORECASE),
            'outcome': re.compile(r'(resolved|callback|refund|sale|connected|failed)', re.IGNORECASE),
            'duration': re.compile(r'duration\s*([<>])\s*(\d+)', re.IGNORECASE),
            'amount': re.compile(r'\$(\d+(?:\.\d{2})?)\s*(?:to|-)\s*\$(\d+(?:\.\d{2})?)', re.IGNORECASE)
        }
    
    def apply_complex_filter(self, df: pd.DataFrame, 
                            filter_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply complex filter configuration to DataFrame.
        
        Args:
            df: DataFrame to filter
            filter_config: Dictionary with filter configuration
            
        Returns:
            Filtered DataFrame
        """
        filter_obj = DataFilter(df)
        
        # Apply date range
        if 'date_range' in filter_config:
            start_date, end_date = filter_config['date_range']
            filter_obj.apply_date_range(start_date, end_date)
        
        # Apply categorical filters
        if 'call_types' in filter_config:
            filter_obj.apply_call_type_filter(filter_config['call_types'])
        
        if 'outcomes' in filter_config:
            filter_obj.apply_outcome_filter(filter_config['outcomes'])
        
        if 'agents' in filter_config:
            filter_obj.apply_agent_filter(filter_config['agents'])
        
        if 'campaigns' in filter_config:
            filter_obj.apply_campaign_filter(filter_config['campaigns'])
        
        # Apply numeric range filters
        if 'duration_range' in filter_config:
            min_dur, max_dur = filter_config['duration_range']
            filter_obj.apply_duration_filter(min_dur, max_dur)
        
        if 'amount_range' in filter_config:
            min_amt, max_amt = filter_config['amount_range']
            filter_obj.apply_amount_filter(min_amt, max_amt)
        
        # Apply text search
        if 'search_query' in filter_config:
            filter_obj.apply_text_search(
                filter_config['search_query'],
                filter_config.get('search_columns')
            )
        
        return filter_obj.get_filtered_data()
    
    def parse_natural_language_query(self, query: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse a natural language query and apply filters.
        
        Args:
            query: Natural language filter query
            df: DataFrame to filter
            
        Returns:
            Filtered DataFrame
        """
        filter_obj = DataFilter(df)
        
        # Parse date range
        date_match = self.query_patterns['date_range'].search(query)
        if date_match:
            start_date = datetime.strptime(date_match.group(1), '%Y-%m-%d')
            end_date = datetime.strptime(date_match.group(2), '%Y-%m-%d')
            filter_obj.apply_date_range(start_date, end_date)
        
        # Parse last N days
        days_match = self.query_patterns['last_n_days'].search(query)
        if days_match:
            days = int(days_match.group(1))
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            filter_obj.apply_date_range(start_date, end_date)
        
        # Parse call types
        type_matches = self.query_patterns['call_type'].findall(query)
        if type_matches:
            call_types = [match.capitalize() for match in type_matches]
            filter_obj.apply_call_type_filter(call_types)
        
        # Parse outcomes
        outcome_matches = self.query_patterns['outcome'].findall(query)
        if outcome_matches:
            outcomes = [match.capitalize() for match in outcome_matches]
            filter_obj.apply_outcome_filter(outcomes)
        
        # Parse duration
        duration_match = self.query_patterns['duration'].search(query)
        if duration_match:
            operator = duration_match.group(1)
            value = float(duration_match.group(2))
            if operator == '>':
                filter_obj.apply_duration_filter(min_duration=value)
            else:
                filter_obj.apply_duration_filter(max_duration=value)
        
        # Parse amount range
        amount_match = self.query_patterns['amount'].search(query)
        if amount_match:
            min_amount = float(amount_match.group(1))
            max_amount = float(amount_match.group(2))
            filter_obj.apply_amount_filter(min_amount, max_amount)
        
        return filter_obj.get_filtered_data()
    
    def create_smart_segments(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create smart customer segments based on behavior patterns.
        
        Args:
            df: DataFrame with call data
            
        Returns:
            Dictionary of segment DataFrames
        """
        segments = {}
        
        # High value segment
        if 'revenue' in df.columns:
            revenue_threshold = df['revenue'].quantile(0.75)
            segments['high_value'] = df[df['revenue'] > revenue_threshold]
        
        # Frequent callers
        if 'phone_number' in df.columns:
            call_counts = df['phone_number'].value_counts()
            frequent_numbers = call_counts[call_counts > 3].index
            segments['frequent_callers'] = df[df['phone_number'].isin(frequent_numbers)]
        
        # Long duration calls
        if 'duration' in df.columns:
            duration_threshold = df['duration'].quantile(0.75)
            segments['long_calls'] = df[df['duration'] > duration_threshold]
        
        # Failed outcomes
        if 'outcome' in df.columns:
            segments['failed_calls'] = df[df['outcome'].isin(['failed', 'no_answer', 'busy'])]
        
        # Recent calls
        if 'timestamp' in df.columns:
            recent_date = pd.Timestamp.now() - pd.Timedelta(days=7)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            segments['recent_calls'] = df[df['timestamp'] > recent_date]
        
        return segments