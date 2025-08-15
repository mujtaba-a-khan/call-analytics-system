"""
Aggregations and Metrics Module

Provides analytics and aggregation functions for call data.
Generates metrics, statistics, and visualizations.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class CallMetrics:
    """
    Handles calculation of metrics and statistics for call data.
    Provides various aggregation methods and visualization generators.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a DataFrame of call data.
        
        Args:
            df: DataFrame containing call records
        """
        self.df = df.copy()
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for analysis by ensuring proper types and adding derived columns"""
        # Ensure datetime column
        if 'start_time' in self.df.columns:
            self.df['start_time'] = pd.to_datetime(self.df['start_time'], errors='coerce')
            
            # Add derived time columns
            self.df['date'] = self.df['start_time'].dt.date
            self.df['hour'] = self.df['start_time'].dt.hour
            self.df['day_of_week'] = self.df['start_time'].dt.day_name()
            self.df['week'] = self.df['start_time'].dt.isocalendar().week
            self.df['month'] = self.df['start_time'].dt.month_name()
        
        # Ensure numeric columns
        if 'duration_seconds' in self.df.columns:
            self.df['duration_seconds'] = pd.to_numeric(self.df['duration_seconds'], errors='coerce')
        
        if 'amount' in self.df.columns:
            self.df['amount'] = pd.to_numeric(self.df['amount'], errors='coerce')
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for the dataset.
        
        Returns:
            Dictionary containing various statistics
        """
        stats = {
            'total_calls': len(self.df),
            'date_range': self._get_date_range(),
            'connected_percentage': self._calculate_connection_rate(),
            'duration_statistics': self._calculate_duration_stats(),
            'type_distribution': self._get_type_distribution(),
            'outcome_distribution': self._get_outcome_distribution(),
            'unique_agents': self._count_unique_agents(),
            'unique_campaigns': self._count_unique_campaigns(),
            'revenue_stats': self._calculate_revenue_stats(),
            'peak_hours': self._identify_peak_hours(),
            'busiest_days': self._identify_busiest_days()
        }
        
        return stats
    
    def _get_date_range(self) -> Dict[str, str]:
        """Get the date range of calls"""
        if 'start_time' not in self.df.columns or self.df.empty:
            return {'start': 'N/A', 'end': 'N/A'}
        
        return {
            'start': str(self.df['start_time'].min()),
            'end': str(self.df['start_time'].max())
        }
    
    def _calculate_connection_rate(self) -> float:
        """Calculate the percentage of connected calls"""
        if 'connection_status' not in self.df.columns or self.df.empty:
            return 0.0
        
        connected_count = (self.df['connection_status'] == 'Connected').sum()
        total_count = len(self.df)
        
        return (connected_count / total_count * 100) if total_count > 0 else 0.0
    
    def _calculate_duration_stats(self) -> Dict[str, float]:
        """Calculate duration statistics"""
        if 'duration_seconds' not in self.df.columns or self.df.empty:
            return {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'total_hours': 0}
        
        duration_col = self.df['duration_seconds'].dropna()
        
        return {
            'mean': float(duration_col.mean()) if not duration_col.empty else 0,
            'median': float(duration_col.median()) if not duration_col.empty else 0,
            'min': float(duration_col.min()) if not duration_col.empty else 0,
            'max': float(duration_col.max()) if not duration_col.empty else 0,
            'total_hours': float(duration_col.sum() / 3600) if not duration_col.empty else 0
        }
    
    def _get_type_distribution(self) -> Dict[str, int]:
        """Get distribution of call types"""
        if 'call_type' not in self.df.columns:
            return {}
        
        return self.df['call_type'].value_counts().to_dict()
    
    def _get_outcome_distribution(self) -> Dict[str, int]:
        """Get distribution of call outcomes"""
        if 'outcome' not in self.df.columns:
            return {}
        
        return self.df['outcome'].value_counts().to_dict()
    
    def _count_unique_agents(self) -> int:
        """Count unique agents"""
        if 'agent_id' not in self.df.columns:
            return 0
        
        return self.df['agent_id'].nunique()
    
    def _count_unique_campaigns(self) -> int:
        """Count unique campaigns"""
        if 'campaign' not in self.df.columns:
            return 0
        
        return self.df['campaign'].nunique()
    
    def _calculate_revenue_stats(self) -> Dict[str, float]:
        """Calculate revenue statistics"""
        if 'amount' not in self.df.columns or self.df.empty:
            return {'total': 0, 'average': 0, 'max': 0}
        
        amount_col = self.df['amount'].dropna()
        
        return {
            'total': float(amount_col.sum()) if not amount_col.empty else 0,
            'average': float(amount_col.mean()) if not amount_col.empty else 0,
            'max': float(amount_col.max()) if not amount_col.empty else 0
        }
    
    def _identify_peak_hours(self) -> List[int]:
        """Identify peak call hours"""
        if 'hour' not in self.df.columns or self.df.empty:
            return []
        
        hourly_counts = self.df['hour'].value_counts()
        if hourly_counts.empty:
            return []
        
        # Get top 3 peak hours
        return hourly_counts.nlargest(3).index.tolist()
    
    def _identify_busiest_days(self) -> List[str]:
        """Identify busiest days of the week"""
        if 'day_of_week' not in self.df.columns or self.df.empty:
            return []
        
        daily_counts = self.df['day_of_week'].value_counts()
        if daily_counts.empty:
            return []
        
        # Get top 3 busiest days
        return daily_counts.nlargest(3).index.tolist()
    
    def get_type_distribution_chart(self) -> go.Figure:
        """
        Generate a pie chart for call type distribution.
        
        Returns:
            Plotly figure object
        """
        if 'call_type' not in self.df.columns:
            return go.Figure().add_annotation(text="No type data available")
        
        type_counts = self.df['call_type'].value_counts()
        
        fig = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Call Type Distribution",
            color_discrete_map={
                'Inquiry': '#3B82F6',
                'Billing/Sales': '#14B8A6',
                'Support': '#F59E0B',
                'Complaint': '#EF4444',
                'Unknown': '#6B7280'
            }
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        
        return fig
    
    def get_outcome_distribution_chart(self) -> go.Figure:
        """
        Generate a bar chart for outcome distribution.
        
        Returns:
            Plotly figure object
        """
        if 'outcome' not in self.df.columns:
            return go.Figure().add_annotation(text="No outcome data available")
        
        outcome_counts = self.df['outcome'].value_counts()
        
        fig = px.bar(
            x=outcome_counts.index,
            y=outcome_counts.values,
            title="Call Outcome Distribution",
            labels={'x': 'Outcome', 'y': 'Count'},
            color=outcome_counts.index,
            color_discrete_map={
                'Resolved': '#16A34A',
                'Callback': '#8B5CF6',
                'Refund': '#F59E0B',
                'Sale-close': '#14B8A6',
                'Unknown': '#6B7280'
            }
        )
        
        fig.update_layout(height=400, showlegend=False)
        
        return fig
    
    def get_timeline_chart(self) -> go.Figure:
        """
        Generate a timeline chart showing call volume over time.
        
        Returns:
            Plotly figure object
        """
        if 'date' not in self.df.columns or self.df.empty:
            return go.Figure().add_annotation(text="No timeline data available")
        
        daily_counts = self.df.groupby('date').size().reset_index(name='count')
        
        fig = px.line(
            daily_counts,
            x='date',
            y='count',
            title="Call Volume Over Time",
            labels={'date': 'Date', 'count': 'Number of Calls'},
            markers=True
        )
        
        fig.update_layout(height=400)
        fig.update_xaxes(rangeslider_visible=True)
        
        return fig
    
    def get_agent_performance_chart(self) -> go.Figure:
        """
        Generate a chart showing agent performance metrics.
        
        Returns:
            Plotly figure object
        """
        if 'agent_id' not in self.df.columns or self.df.empty:
            return go.Figure().add_annotation(text="No agent data available")
        
        # Calculate metrics per agent
        agent_stats = self.df.groupby('agent_id').agg({
            'call_id': 'count',
            'duration_seconds': 'mean',
            'connection_status': lambda x: (x == 'Connected').mean() * 100
        }).round(2)
        
        agent_stats.columns = ['Total Calls', 'Avg Duration (s)', 'Connection Rate (%)']
        agent_stats = agent_stats.sort_values('Total Calls', ascending=False).head(10)
        
        fig = px.bar(
            agent_stats,
            y=agent_stats.index,
            x='Total Calls',
            title="Top 10 Agents by Call Volume",
            orientation='h',
            text='Total Calls'
        )
        
        fig.update_layout(height=400)
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        
        return fig
    
    def get_hourly_distribution_chart(self) -> go.Figure:
        """
        Generate a chart showing call distribution by hour.
        
        Returns:
            Plotly figure object
        """
        if 'hour' not in self.df.columns or self.df.empty:
            return go.Figure().add_annotation(text="No hourly data available")
        
        hourly_counts = self.df['hour'].value_counts().sort_index()
        
        fig = px.bar(
            x=hourly_counts.index,
            y=hourly_counts.values,
            title="Call Distribution by Hour",
            labels={'x': 'Hour of Day', 'y': 'Number of Calls'}
        )
        
        fig.update_layout(height=400)
        fig.update_xaxes(tickmode='linear', tick0=0, dtick=1)
        
        return fig
    
    def generate_summary_report(self) -> str:
        """
        Generate a text summary report of the analytics.
        
        Returns:
            Formatted summary report string
        """
        stats = self.calculate_statistics()
        
        report = f"""
        📊 CALL ANALYTICS SUMMARY REPORT
        ================================
        
        📅 Report Period: {stats['date_range']['start']} to {stats['date_range']['end']}
        
        📞 OVERALL METRICS
        ------------------
        • Total Calls: {stats['total_calls']:,}
        • Connection Rate: {stats['connected_percentage']:.1f}%
        • Total Call Hours: {stats['duration_statistics']['total_hours']:.1f}
        
        ⏱️ DURATION STATISTICS
        ----------------------
        • Average Duration: {stats['duration_statistics']['mean']:.1f} seconds
        • Median Duration: {stats['duration_statistics']['median']:.1f} seconds
        • Min Duration: {stats['duration_statistics']['min']:.1f} seconds
        • Max Duration: {stats['duration_statistics']['max']:.1f} seconds
        
        📋 CALL TYPES
        -------------
        """
        
        for call_type, count in stats['type_distribution'].items():
            percentage = (count / stats['total_calls'] * 100) if stats['total_calls'] > 0 else 0
            report += f"• {call_type}: {count:,} ({percentage:.1f}%)\n        "
        
        report += """
        ✅ CALL OUTCOMES
        ----------------
        """
        
        for outcome, count in stats['outcome_distribution'].items():
            percentage = (count / stats['total_calls'] * 100) if stats['total_calls'] > 0 else 0
            report += f"• {outcome}: {count:,} ({percentage:.1f}%)\n        "
        
        report += f"""
        👥 WORKFORCE
        ------------
        • Unique Agents: {stats['unique_agents']}
        • Unique Campaigns: {stats['unique_campaigns']}
        
        💰 REVENUE
        ----------
        • Total Revenue: ${stats['revenue_stats']['total']:,.2f}
        • Average Transaction: ${stats['revenue_stats']['average']:,.2f}
        • Largest Transaction: ${stats['revenue_stats']['max']:,.2f}
        
        ⏰ PEAK TIMES
        -------------
        • Peak Hours: {', '.join(map(str, stats['peak_hours']))}
        • Busiest Days: {', '.join(stats['busiest_days'])}
        
        ================================
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return report