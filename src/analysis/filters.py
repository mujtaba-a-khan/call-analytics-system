"""
Data Filtering Module

Provides filtering capabilities for call data based on various criteria.
Supports both simple and complex filter combinations.
"""

import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Union
import re
import logging

logger = logging.getLogger(__name__)


class DataFilter:
    """
    Handles filtering of call data based on various criteria.
    Supports date ranges, categories, text search, and custom conditions.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a DataFrame to filter.
        
        Args:
            df: DataFrame containing call data
        """
        self.original_df = df.copy()
        self.filtered_df = df.copy()
        self.active_filters = {}
    
    def reset_filters(self):
        """Reset all filters and return to original data"""
        self.filtered_df = self.original_df.copy()
        self.active_filters = {}
        logger.info("All filters reset")
    
    def apply_date_range(self, 
                        start_date: Optional[Union[date, datetime]] = None,
                        end_date: Optional[Union[date, datetime]] = None) -> 'DataFilter':
        """
        Filter by date range.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
        
        Returns:
            Self for method chaining
        """
        if 'start_time' not in self.filtered_df.columns:
            logger.warning("No start_time column found for date filtering")
            return self
        
        # Convert to datetime if needed
        self.filtered_df['start_time'] = pd.to_datetime(self.filtered_df['start_time'])
        
        if start_date:
            start_dt = pd.to_datetime(start_date)
            self.filtered_df = self.filtered_df[self.filtered_df['start_time'] >= start_dt]
            self.active_filters['start_date'] = str(start_date)
        
        if end_date:
            # Make end date inclusive by adding 1 day
            end_dt = pd.to_datetime(end_date) + timedelta(days=1)
            self.filtered_df = self.filtered_df[self.filtered_df['start_time'] < end_dt]
            self.active_filters['end_date'] = str(end_date)
        
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
        if 'call_type' not in self.filtered_df.columns:
            logger.warning("No call_type column found")
            return self
        
        if call_types and 'All' not in call_types:
            self.filtered_df = self.filtered_df[self.filtered_df['call_type'].isin(call_types)]
            self.active_filters['call_types'] = call_types
            logger.info(f"Call type filter applied: {call_types}")
        
        return self
    
    def apply_outcome_filter(self, outcomes: List[str]) -> 'DataFilter':
        """
        Filter by call outcomes.
        
        Args:
            outcomes: List of outcomes to include
        
        Returns:
            Self for method chaining
        """
        if 'outcome' not in self.filtered_df.columns:
            logger.warning("No outcome column found")
            return self
        
        if outcomes and 'All' not in outcomes:
            self.filtered_df = self.filtered_df[self.filtered_df['outcome'].isin(outcomes)]
            self.active_filters['outcomes'] = outcomes
            logger.info(f"Outcome filter applied: {outcomes}")
        
        return self
    
    def apply_agent_filter(self, agents: List[str]) -> 'DataFilter':
        """
        Filter by agents.
        
        Args:
            agents: List of agent IDs to include
        
        Returns:
            Self for method chaining
        """
        if 'agent_id' not in self.filtered_df.columns:
            logger.warning("No agent_id column found")
            return self
        
        if agents and 'All' not in agents:
            self.filtered_df = self.filtered_df[self.filtered_df['agent_id'].isin(agents)]
            self.active_filters['agents'] = agents
            logger.info(f"Agent filter applied: {agents}")
        
        return self
    
    def apply_campaign_filter(self, campaigns: List[str]) -> 'DataFilter':
        """
        Filter by campaigns.
        
        Args:
            campaigns: List of campaign names to include
        
        Returns:
            Self for method chaining
        """
        if 'campaign' not in self.filtered_df.columns:
            logger.warning("No campaign column found")
            return self
        
        if campaigns and 'All' not in campaigns:
            self.filtered_df = self.filtered_df[self.filtered_df['campaign'].isin(campaigns)]
            self.active_filters['campaigns'] = campaigns
            logger.info(f"Campaign filter applied: {campaigns}")
        
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
            logger.warning("No duration_seconds column found")
            return self
        
        if min_duration is not None:
            self.filtered_df = self.filtered_df[self.filtered_df['duration_seconds'] >= min_duration]
            self.active_filters['min_duration'] = min_duration
        
        if max_duration is not None:
            self.filtered_df = self.filtered_df[self.filtered_df['duration_seconds'] <= max_duration]
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
            logger.warning("No amount column found")
            return self
        
        if min_amount is not None:
            self.filtered_df = self.filtered_df[self.filtered_df['amount'] >= min_amount]
            self.active_filters['min_amount'] = min_amount
        
        if max_amount is not None:
            self.filtered_df = self.filtered_df[self.filtered_df['amount'] <= max_amount]
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
            columns: List of columns to search in (default: all text columns)
        
        Returns:
            Self for method chaining
        """
        if not search_query:
            return self
        
        # Default to searching all string columns
        if columns is None:
            columns = self.filtered_df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        # Create search pattern (case-insensitive)
        pattern = re.compile(re.escape(search_query), re.IGNORECASE)
        
        # Apply search across specified columns
        mask = pd.Series(False, index=self.filtered_df.index)
        for col in columns:
            if col in self.filtered_df.columns:
                mask |= self.filtered_df[col].astype(str).str.contains(pattern, na=False)
        
        self.filtered_df = self.filtered_df[mask]
        self.active_filters['search_query'] = search_query
        logger.info(f"Text search applied: '{search_query}' in {columns}")
        
        return self
    
    def apply_connection_status_filter(self, status: str) -> 'DataFilter':
        """
        Filter by connection status.
        
        Args:
            status: Connection status ('Connected', 'Disconnected', or 'All')
        
        Returns:
            Self for method chaining
        """
        if 'connection_status' not in self.filtered_df.columns:
            logger.warning("No connection_status column found")
            return self
        
        if status and status != 'All':
            self.filtered_df = self.filtered_df[self.filtered_df['connection_status'] == status]
            self.active_filters['connection_status'] = status
            logger.info(f"Connection status filter applied: {status}")
        
        return self
    
    def apply_custom_filter(self, filter_func) -> 'DataFilter':
        """
        Apply a custom filter function.
        
        Args:
            filter_func: Function that takes a DataFrame and returns a boolean mask
        
        Returns:
            Self for method chaining
        """
        try:
            mask = filter_func(self.filtered_df)
            self.filtered_df = self.filtered_df[mask]
            self.active_filters['custom'] = 'Applied'
            logger.info("Custom filter applied")
        except Exception as e:
            logger.error(f"Failed to apply custom filter: {e}")
        
        return self
    
    def apply_filter_spec(self, spec: Dict[str, Any]) -> 'DataFilter':
        """
        Apply filters from a specification dictionary.
        
        Args:
            spec: Dictionary containing filter specifications
        
        Returns:
            Self for method chaining
        """
        # Date range
        if 'start_date' in spec or 'end_date' in spec:
            self.apply_date_range(spec.get('start_date'), spec.get('end_date'))
        
        # Call types
        if 'call_types' in spec:
            self.apply_call_type_filter(spec['call_types'])
        
        # Outcomes
        if 'outcomes' in spec:
            self.apply_outcome_filter(spec['outcomes'])
        
        # Agents
        if 'agents' in spec:
            self.apply_agent_filter(spec['agents'])
        
        # Campaigns
        if 'campaigns' in spec:
            self.apply_campaign_filter(spec['campaigns'])
        
        # Duration
        if 'min_duration' in spec or 'max_duration' in spec:
            self.apply_duration_filter(spec.get('min_duration'), spec.get('max_duration'))
        
        # Amount
        if 'min_amount' in spec or 'max_amount' in spec:
            self.apply_amount_filter(spec.get('min_amount'), spec.get('max_amount'))
        
        # Text search
        if 'search_query' in spec:
            self.apply_text_search(spec['search_query'], spec.get('search_columns'))
        
        # Connection status
        if 'connection_status' in spec:
            self.apply_connection_status_filter(spec['connection_status'])
        
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
        Get a summary of applied filters and results.
        
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


class SmartFilter:
    """
    Advanced filtering with natural language processing support.
    Interprets user queries and applies appropriate filters.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a DataFrame.
        
        Args:
            df: DataFrame to filter
        """
        self.filter = DataFilter(df)
        self.query_patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for query interpretation"""
        return {
            'date_range': re.compile(r'between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})', re.IGNORECASE),
            'last_n_days': re.compile(r'last\s+(\d+)\s+days?', re.IGNORECASE),
            'call_type': re.compile(r'(inquiry|billing|sales|support|complaint)', re.IGNORECASE),
            'outcome': re.compile(r'(resolved|callback|refund|sale)', re.IGNORECASE),
            'duration': re.compile(r'duration\s*([<>])\s*(\d+)', re.IGNORECASE),
            'amount': re.compile(r'\$(\d+(?:\.\d{2})?)\s*(?:to|-)\s*\$(\d+(?:\.\d{2})?)', re.IGNORECASE)
        }
    
    def parse_and_apply(self, query: str) -> pd.DataFrame:
        """
        Parse a natural language query and apply filters.
        
        Args:
            query: Natural language filter query
        
        Returns:
            Filtered DataFrame
        """
        # Reset filters first
        self.filter.reset_filters()
        
        # Parse date range
        date_match = self.query_patterns['date_range'].search(query)
        if date_match:
            start_date = datetime.strptime(date_match.group(1), '%Y-%m-%d')
            end_date = datetime.strptime(date_match.group(2), '%Y-%m-%d')
            self.filter.apply_date_range(start_date, end_date)
        
        # Parse last N days
        days_match = self.query_patterns['last_n_days'].search(query)
        if days_match:
            days = int(days_match.group(1))
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            self.filter.apply_date_range(start_date, end_date)
        
        # Parse call types
        type_matches = self.query_patterns['call_type'].findall(query)
        if type_matches:
            call_types = [match.capitalize() for match in type_matches]
            self.filter.apply_call_type_filter(call_types)
        
        # Parse outcomes
        outcome_matches = self.query_patterns['outcome'].findall(query)
        if outcome_matches:
            outcomes = [match.capitalize() for match in outcome_matches]
            self.filter.apply_outcome_filter(outcomes)
        
        # Parse duration
        duration_match = self.query_patterns['duration'].search(query)
        if duration_match:
            operator = duration_match.group(1)
            value = float(duration_match.group(2))
            if operator == '>':
                self.filter.apply_duration_filter(min_duration=value)
            else:
                self.filter.apply_duration_filter(max_duration=value)
        
        # Parse amount range
        amount_match = self.query_patterns['amount'].search(query)
        if amount_match:
            min_amount = float(amount_match.group(1))
            max_amount = float(amount_match.group(2))
            self.filter.apply_amount_filter(min_amount, max_amount)
        
        return self.filter.get_filtered_data()