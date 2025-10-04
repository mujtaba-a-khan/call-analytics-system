"""
Metrics Display Components for Call Analytics System

This module provides reusable Streamlit components for displaying
key performance indicators (KPIs), metrics cards, and statistical
summaries with visual enhancements and real-time updates.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
import plotly.graph_objects as go

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """
    Represents a metric value with optional comparison data
    for displaying trends and changes.
    """
    
    value: Union[int, float, str]
    label: str
    delta: Optional[Union[int, float]] = None
    delta_color: str = 'normal'  # 'normal', 'inverse', 'off'
    prefix: str = ''
    suffix: str = ''
    format_type: str = 'number'  # 'number', 'percent', 'currency', 'duration'
    icon: Optional[str] = None
    help_text: Optional[str] = None
    
    def format_value(self) -> str:
        """
        Format the metric value based on its type.
        
        Returns:
            Formatted string representation of the value
        """
        if self.format_type == 'percent':
            return f"{self.prefix}{self.value:.1f}%{self.suffix}"
        elif self.format_type == 'currency':
            return f"{self.prefix}${self.value:,.2f}{self.suffix}"
        elif self.format_type == 'duration':
            # Convert seconds to human-readable format
            if isinstance(self.value, (int, float)):
                hours = int(self.value // 3600)
                minutes = int((self.value % 3600) // 60)
                seconds = int(self.value % 60)
                if hours > 0:
                    return f"{hours}h {minutes}m"
                elif minutes > 0:
                    return f"{minutes}m {seconds}s"
                else:
                    return f"{seconds}s"
            return str(self.value)
        else:
            # Default number formatting
            if isinstance(self.value, float):
                return f"{self.prefix}{self.value:,.1f}{self.suffix}"
            elif isinstance(self.value, int):
                return f"{self.prefix}{self.value:,}{self.suffix}"
            else:
                return f"{self.prefix}{self.value}{self.suffix}"
    
    def format_delta(self) -> Optional[str]:
        """
        Format the delta value for display.
        
        Returns:
            Formatted delta string or None
        """
        if self.delta is None:
            return None
        
        if self.format_type == 'percent':
            return f"{self.delta:+.1f}%"
        elif self.format_type == 'currency':
            return f"${self.delta:+,.2f}"
        else:
            if isinstance(self.delta, float):
                return f"{self.delta:+,.1f}"
            else:
                return f"{self.delta:+,}"


class MetricCard:
    """
    Component for displaying individual metric cards with
    values, trends, and visual indicators.
    """
    
    @classmethod
    def render(cls,
               metric: MetricValue,
               container: Any = None,
               use_column: bool = True,
               label_visibility: str = 'visible') -> None:
        """
        Render a single metric card.
        
        Args:
            metric: MetricValue object with metric data
            container: Streamlit container to render in
            use_column: Whether to wrap in a column
        """
        container = container or st
        
        # Create column if requested
        if use_column:
            col = container.columns(1)[0]
        else:
            col = container
        
        # Render metric with icon if provided
        metric_label = metric.label or "Metric"
        if metric.icon:
            col.markdown(f"{metric.icon} **{metric_label}**")
        
        # Use Streamlit's metric component
        display_label = metric_label
        metric_label_visibility = 'hidden' if metric.icon else label_visibility
        col.metric(
            label=display_label,
            value=metric.format_value(),
            delta=metric.format_delta(),
            delta_color=metric.delta_color,
            help=metric.help_text,
            label_visibility=metric_label_visibility
        )


class MetricsGrid:
    """
    Component for displaying multiple metrics in a responsive grid layout
    with automatic column sizing and grouping.
    """
    
    @classmethod
    def render(cls,
               metrics: List[MetricValue],
               container: Any = None,
               columns: Optional[int] = None,
               group_size: int = 4,
               label_visibility: str = 'visible') -> None:
        """
        Render metrics in a grid layout.
        
        Args:
            metrics: List of MetricValue objects
            container: Streamlit container to render in
            columns: Number of columns (auto-calculated if None)
            group_size: Default number of metrics per row
        """
        container = container or st
        
        # Determine number of columns
        if columns is None:
            columns = min(len(metrics), group_size)
        
        # Create columns
        cols = container.columns(columns)
        
        # Render metrics in columns
        for idx, metric in enumerate(metrics):
            col_idx = idx % columns
            MetricCard.render(
                metric,
                cols[col_idx],
                use_column=False,
                label_visibility=label_visibility
            )


class ProgressIndicator:
    """
    Utility component for visualizing the progress of long-running tasks.

    Provides a simple wrapper around Streamlit's progress bar with
    convenience methods for updating the progress value and status text.
    """

    def __init__(self,
                 label: str = "Processing...",
                 total: int = 100,
                 container: Any = None) -> None:
        self.container = container or st
        self.label = label
        self.total = max(total, 1)
        self.current = 0

        self._status = self.container.empty()
        self._progress_bar = self.container.progress(0)
        self._status.write(label)

    def update(self,
               step: int = 1,
               current: Optional[int] = None,
               detail: Optional[str] = None) -> None:
        """Increment or set the progress value and refresh the display."""
        if current is not None:
            self.current = max(0, min(current, self.total))
        else:
            self.current = max(0, min(self.current + step, self.total))

        percent = int((self.current / self.total) * 100)
        self._progress_bar.progress(percent)

        if detail is not None:
            self._status.write(detail)
        else:
            self._status.write(f"{self.label} ({percent}%)")

    def complete(self, detail: Optional[str] = None) -> None:
        """Mark the progress indicator as complete."""
        self.current = self.total
        self._progress_bar.progress(100)
        self._status.success(detail or f"{self.label} completed")

    def reset(self, detail: Optional[str] = None) -> None:
        """Reset the indicator back to zero progress."""
        self.current = 0
        self._progress_bar.progress(0)
        self._status.write(detail or self.label)


class SummaryStats:
    """
    Component for displaying comprehensive statistical summaries
    with mean, median, percentiles, and distribution info.
    """
    
    @classmethod
    def calculate_stats(cls, 
                       data: pd.Series,
                       percentiles: List[int] = [25, 50, 75, 95]) -> Dict[str, float]:
        """
        Calculate comprehensive statistics for a data series.
        
        Args:
            data: Pandas Series with numeric data
            percentiles: List of percentiles to calculate
            
        Returns:
            Dictionary of calculated statistics
        """
        stats = {
            'count': len(data),
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'sum': data.sum()
        }
        
        # Add percentiles
        for p in percentiles:
            stats[f'p{p}'] = data.quantile(p / 100)
        
        # Add additional metrics
        stats['range'] = stats['max'] - stats['min']
        stats['cv'] = (stats['std'] / stats['mean'] * 100) if stats['mean'] != 0 else 0
        
        return stats
    
    @classmethod
    def render(cls,
               data: pd.Series,
               title: str,
               container: Any = None,
               show_distribution: bool = True) -> None:
        """
        Render statistical summary display.
        
        Args:
            data: Pandas Series with numeric data
            title: Title for the summary
            container: Streamlit container to render in
            show_distribution: Whether to show distribution chart
        """
        container = container or st
        
        # Calculate statistics
        stats = cls.calculate_stats(data)
        
        # Display title
        container.subheader(title)
        
        # Create columns for stats display
        col1, col2, col3 = container.columns(3)
        
        # Central tendency metrics
        with col1:
            st.metric("Mean", f"{stats['mean']:.2f}")
            st.metric("Median", f"{stats['p50']:.2f}")
        
        # Spread metrics
        with col2:
            st.metric("Std Dev", f"{stats['std']:.2f}")
            st.metric("Range", f"{stats['range']:.2f}")
        
        # Summary metrics
        with col3:
            st.metric("Total", f"{stats['sum']:.0f}")
            st.metric("Count", f"{stats['count']:,}")
        
        # Show distribution if requested
        if show_distribution:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=data,
                nbinsx=30,
                name='Distribution',
                marker_color='lightblue'
            ))
            fig.update_layout(
                height=200,
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False
            )
            container.plotly_chart(fig, use_container_width=True)


class KPIDashboard:
    """
    Component for creating comprehensive KPI dashboards with
    multiple metric groups and comparative analysis.
    """
    
    @classmethod
    def render_call_metrics(cls,
                           data: pd.DataFrame,
                           container: Any = None,
                           compare_period: Optional[pd.DataFrame] = None) -> None:
        """
        Render call-related KPI metrics.
        
        Args:
            data: Current period DataFrame
            container: Streamlit container to render in
            compare_period: Optional comparison period DataFrame
        """
        container = container or st
        
        # Calculate current metrics
        total_calls = len(data)
        connected_calls = len(data[data['outcome'] == 'connected'])
        connection_rate = (connected_calls / total_calls * 100) if total_calls > 0 else 0
        avg_duration = data['duration'].mean() / 60 if 'duration' in data.columns else 0
        
        # Calculate comparison deltas if provided
        deltas = {}
        if compare_period is not None and len(compare_period) > 0:
            prev_total = len(compare_period)
            prev_connected = len(compare_period[compare_period['outcome'] == 'connected'])
            prev_rate = (prev_connected / prev_total * 100) if prev_total > 0 else 0
            prev_duration = compare_period['duration'].mean() / 60
            
            deltas['total'] = total_calls - prev_total
            deltas['connected'] = connected_calls - prev_connected
            deltas['rate'] = connection_rate - prev_rate
            deltas['duration'] = avg_duration - prev_duration
        
        # Create metrics
        metrics = [
            MetricValue(
                value=total_calls,
                label="Total Calls",
                delta=deltas.get('total'),
                icon="📞",
                help_text="Total number of calls in the period"
            ),
            MetricValue(
                value=connected_calls,
                label="Connected Calls",
                delta=deltas.get('connected'),
                icon="✅",
                help_text="Number of successfully connected calls"
            ),
            MetricValue(
                value=connection_rate,
                label="Connection Rate",
                delta=deltas.get('rate'),
                format_type='percent',
                icon="📊",
                help_text="Percentage of calls that connected"
            ),
            MetricValue(
                value=avg_duration,
                label="Avg Duration",
                delta=deltas.get('duration'),
                suffix=" min",
                icon="⏱️",
                help_text="Average call duration in minutes"
            )
        ]
        
        # Render metrics grid
        MetricsGrid.render(metrics, container, label_visibility='visible')
    
    @classmethod
    def render_revenue_metrics(cls,
                              data: pd.DataFrame,
                              container: Any = None,
                              compare_period: Optional[pd.DataFrame] = None) -> None:
        """
        Render revenue-related KPI metrics.
        
        Args:
            data: Current period DataFrame
            container: Streamlit container to render in
            compare_period: Optional comparison period DataFrame
        """
        container = container or st
        
        # Calculate revenue metrics
        total_revenue = data['revenue'].sum() if 'revenue' in data.columns else 0
        revenue_calls = len(data[data['revenue'] > 0]) if 'revenue' in data.columns else 0
        avg_revenue = total_revenue / len(data) if len(data) > 0 else 0
        conversion_rate = (revenue_calls / len(data) * 100) if len(data) > 0 else 0
        
        # Calculate comparison deltas
        deltas = {}
        if compare_period is not None and len(compare_period) > 0:
            prev_revenue = compare_period['revenue'].sum()
            prev_revenue_calls = len(compare_period[compare_period['revenue'] > 0])
            prev_avg = prev_revenue / len(compare_period)
            prev_conversion = (prev_revenue_calls / len(compare_period) * 100)
            
            deltas['total'] = total_revenue - prev_revenue
            deltas['calls'] = revenue_calls - prev_revenue_calls
            deltas['avg'] = avg_revenue - prev_avg
            deltas['conversion'] = conversion_rate - prev_conversion
        
        # Create metrics
        metrics = [
            MetricValue(
                value=total_revenue,
                label="Total Revenue",
                delta=deltas.get('total'),
                format_type='currency',
                icon="💰",
                help_text="Total revenue generated"
            ),
            MetricValue(
                value=revenue_calls,
                label="Revenue Calls",
                delta=deltas.get('calls'),
                icon="💵",
                help_text="Number of calls generating revenue"
            ),
            MetricValue(
                value=avg_revenue,
                label="Avg Revenue/Call",
                delta=deltas.get('avg'),
                format_type='currency',
                icon="📈",
                help_text="Average revenue per call"
            ),
            MetricValue(
                value=conversion_rate,
                label="Conversion Rate",
                delta=deltas.get('conversion'),
                format_type='percent',
                delta_color='normal',
                icon="🎯",
                help_text="Percentage of calls generating revenue"
            )
        ]
        
        # Render metrics grid
        MetricsGrid.render(metrics, container, label_visibility='visible')


class PerformanceIndicator:
    """
    Component for displaying performance indicators with
    visual gauges, progress bars, and target comparisons.
    """
    
    @classmethod
    def render_gauge(cls,
                    value: float,
                    target: float,
                    title: str,
                    container: Any = None,
                    ranges: Optional[List[Tuple[float, str]]] = None) -> None:
        """
        Render a gauge chart for performance indication.
        
        Args:
            value: Current value
            target: Target value
            title: Gauge title
            container: Streamlit container to render in
            ranges: Optional list of (threshold, color) tuples
        """
        container = container or st
        
        # Default ranges if not provided
        if ranges is None:
            ranges = [
                (0.5, 'red'),
                (0.8, 'yellow'),
                (1.0, 'green')
            ]
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            delta={'reference': target},
            title={'text': title},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, max(value, target) * 1.2]},
                'bar': {'color': 'darkblue'},
                'steps': [
                    {'range': [0, target * r], 'color': c}
                    for r, c in ranges
                ],
                'threshold': {
                    'line': {'color': 'red', 'width': 4},
                    'thickness': 0.75,
                    'value': target
                }
            }
        ))
        
        fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
        container.plotly_chart(fig, use_container_width=True)
    
    @classmethod
    def render_progress_bar(cls,
                           value: float,
                           max_value: float,
                           label: str,
                           container: Any = None,
                           color: str = 'blue') -> None:
        """
        Render a progress bar indicator.
        
        Args:
            value: Current value
            max_value: Maximum value
            label: Progress bar label
            container: Streamlit container to render in
            color: Bar color
        """
        container = container or st
        
        # Calculate percentage
        percentage = min((value / max_value * 100), 100) if max_value > 0 else 0
        
        # Display label and value
        container.markdown(f"**{label}**: {value:,.0f} / {max_value:,.0f}")
        
        # Display progress bar
        container.progress(percentage / 100)
        
        # Display percentage text
        container.caption(f"{percentage:.1f}% complete")
