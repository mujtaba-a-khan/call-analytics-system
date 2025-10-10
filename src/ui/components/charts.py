"""
Chart Components Module for Call Analytics System

This module provides reusable Plotly chart components for visualizing
call analytics data. Includes time series, distribution, and performance
charts with consistent styling and interactivity.
"""

import logging
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure module logger
logger = logging.getLogger(__name__)


class ChartTheme:
    """
    Consistent theme configuration for all charts in the application.
    Provides dark mode compatible color schemes and styling options.
    """

    # Color palette for dark theme
    COLORS = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'danger': '#d62728',
        'warning': '#ff9800',
        'info': '#17a2b8',
        'background': '#0e1117',
        'paper': '#262730',
        'text': '#fafafa',
        'grid': '#404040'
    }

    # Chart color sequences for multiple series
    COLOR_SEQUENCE = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    @classmethod
    def get_layout_template(cls) -> dict[str, Any]:
        """
        Get standard layout template for Plotly charts.

        Returns:
            Dictionary with layout configuration
        """
        return {
            'template': 'plotly_dark',
            'paper_bgcolor': cls.COLORS['background'],
            'plot_bgcolor': cls.COLORS['paper'],
            'font': {'color': cls.COLORS['text'], 'size': 12},
            'margin': {'l': 50, 'r': 30, 't': 40, 'b': 50},
            'hoverlabel': {
                'bgcolor': cls.COLORS['paper'],
                'font_size': 12,
                'font_family': 'Arial'
            },
            'xaxis': {
                'gridcolor': cls.COLORS['grid'],
                'zerolinecolor': cls.COLORS['grid']
            },
            'yaxis': {
                'gridcolor': cls.COLORS['grid'],
                'zerolinecolor': cls.COLORS['grid']
            }
        }


class TimeSeriesChart:
    """
    Creates interactive time series charts for call volume and metrics
    over time with multiple aggregation options.
    """

    @staticmethod
    def create_call_volume_chart(
        data: pd.DataFrame,
        date_column: str = 'timestamp',
        aggregation: str = 'daily',
        group_by: str | None = None,
        title: str = 'Call Volume Over Time'
    ) -> go.Figure:
        """
        Create a time series chart showing call volume trends.

        Args:
            data: DataFrame with call data
            date_column: Name of date column
            aggregation: Time aggregation level ('hourly', 'daily', 'weekly', 'monthly')
            group_by: Optional column to group data by
            title: Chart title

        Returns:
            Plotly Figure object
        """
        try:
            # Convert date column to datetime
            data[date_column] = pd.to_datetime(data[date_column])

            # Set aggregation frequency
            freq_map = {
                'hourly': 'H',
                'daily': 'D',
                'weekly': 'W',
                'monthly': 'M'
            }
            freq = freq_map.get(aggregation, 'D')

            # Create figure
            fig = go.Figure()

            if group_by and group_by in data.columns:
                # Group data and create traces for each group
                for group_name in data[group_by].unique():
                    group_data = data[data[group_by] == group_name]
                    aggregated = group_data.resample(freq, on=date_column).size()

                    fig.add_trace(go.Scatter(
                        x=aggregated.index,
                        y=aggregated.values,
                        mode='lines+markers',
                        name=str(group_name),
                        line=dict(width=2),
                        marker=dict(size=6)
                    ))
            else:
                # Single series aggregation
                aggregated = data.resample(freq, on=date_column).size()

                fig.add_trace(go.Scatter(
                    x=aggregated.index,
                    y=aggregated.values,
                    mode='lines+markers',
                    name='Call Volume',
                    line=dict(color=ChartTheme.COLORS['primary'], width=3),
                    marker=dict(size=8),
                    fill='tozeroy',
                    fillcolor='rgba(31, 119, 180, 0.2)'
                ))

            # Update layout
            layout = ChartTheme.get_layout_template()
            layout.update({
                'title': title,
                'xaxis_title': 'Date',
                'yaxis_title': 'Number of Calls',
                'hovermode': 'x unified',
                'showlegend': bool(group_by)
            })
            fig.update_layout(layout)

            return fig

        except Exception as e:
            logger.error(f"Error creating time series chart: {e}")
            return go.Figure()

    @staticmethod
    def create_peak_hours_heatmap(
        data: pd.DataFrame,
        timestamp_column: str = 'timestamp',
        title: str = 'Call Volume Heatmap by Hour and Day'
    ) -> go.Figure:
        """
        Create a heatmap showing call patterns by hour and day of week.

        Args:
            data: DataFrame with call data
            timestamp_column: Name of timestamp column
            title: Chart title

        Returns:
            Plotly Figure object
        """
        try:
            # Extract hour and day of week
            data['hour'] = pd.to_datetime(data[timestamp_column]).dt.hour
            data['day_of_week'] = pd.to_datetime(data[timestamp_column]).dt.day_name()

            # Create pivot table
            pivot = data.pivot_table(
                index='hour',
                columns='day_of_week',
                values=timestamp_column,
                aggfunc='count',
                fill_value=0
            )

            # Reorder days
            days_order = [
                'Monday',
                'Tuesday',
                'Wednesday',
                'Thursday',
                'Friday',
                'Saturday',
                'Sunday',
            ]
            pivot = pivot.reindex(columns=days_order, fill_value=0)

            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale='Viridis',
                colorbar=dict(title='Calls'),
                hovertemplate='%{x}<br>%{y}:00<br>Calls: %{z}<extra></extra>'
            ))

            # Update layout
            layout = ChartTheme.get_layout_template()
            layout.update({
                'title': title,
                'xaxis_title': 'Day of Week',
                'yaxis_title': 'Hour of Day',
                'yaxis': {'dtick': 1}
            })
            fig.update_layout(layout)

            return fig

        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
            return go.Figure()


class DistributionChart:
    """
    Creates distribution charts for analyzing call characteristics
    such as duration, outcomes, and agent performance.
    """

    @staticmethod
    def create_duration_distribution(
        data: pd.DataFrame,
        duration_column: str = 'duration',
        bins: int = 30,
        title: str = 'Call Duration Distribution'
    ) -> go.Figure:
        """
        Create a histogram showing distribution of call durations.

        Args:
            data: DataFrame with call data
            duration_column: Name of duration column
            bins: Number of histogram bins
            title: Chart title

        Returns:
            Plotly Figure object
        """
        try:
            # Convert duration to minutes
            durations = data[duration_column] / 60  # Assuming duration is in seconds

            # Calculate statistics
            mean_duration = durations.mean()
            median_duration = durations.median()

            # Create histogram
            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=durations,
                nbinsx=bins,
                name='Duration',
                marker_color=ChartTheme.COLORS['primary'],
                opacity=0.8
            ))

            # Add mean and median lines
            fig.add_vline(
                x=mean_duration,
                line_dash="dash",
                line_color=ChartTheme.COLORS['danger'],
                annotation_text=f"Mean: {mean_duration:.1f} min"
            )

            fig.add_vline(
                x=median_duration,
                line_dash="dash",
                line_color=ChartTheme.COLORS['success'],
                annotation_text=f"Median: {median_duration:.1f} min"
            )

            # Update layout
            layout = ChartTheme.get_layout_template()
            layout.update({
                'title': title,
                'xaxis_title': 'Duration (minutes)',
                'yaxis_title': 'Number of Calls',
                'bargap': 0.1
            })
            fig.update_layout(layout)

            return fig

        except Exception as e:
            logger.error(f"Error creating duration distribution: {e}")
            return go.Figure()

    @staticmethod
    def create_outcome_pie_chart(
        data: pd.DataFrame,
        outcome_column: str = 'outcome',
        title: str = 'Call Outcomes Distribution'
    ) -> go.Figure:
        """
        Create a pie chart showing distribution of call outcomes.

        Args:
            data: DataFrame with call data
            outcome_column: Name of outcome column
            title: Chart title

        Returns:
            Plotly Figure object
        """
        try:
            # Calculate outcome counts
            outcome_counts = data[outcome_column].value_counts()

            # Define colors for common outcomes
            outcome_colors = {
                'connected': ChartTheme.COLORS['success'],
                'no_answer': ChartTheme.COLORS['warning'],
                'voicemail': ChartTheme.COLORS['info'],
                'busy': ChartTheme.COLORS['danger'],
                'failed': ChartTheme.COLORS['danger']
            }

            colors = [outcome_colors.get(outcome.lower(), ChartTheme.COLORS['primary'])
                     for outcome in outcome_counts.index]

            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=outcome_counts.index,
                values=outcome_counts.values,
                hole=0.4,  # Donut chart
                marker_colors=colors,
                textinfo='label+percent',
                textposition='auto'
            )])

            # Update layout
            layout = ChartTheme.get_layout_template()
            layout.update({
                'title': title,
                'showlegend': True,
                'legend': {'orientation': 'v', 'yanchor': 'middle', 'y': 0.5}
            })
            fig.update_layout(layout)

            return fig

        except Exception as e:
            logger.error(f"Error creating outcome pie chart: {e}")
            return go.Figure()


class PerformanceChart:
    """
    Creates performance charts for agent and campaign analytics
    with comparative metrics and rankings.
    """

    @staticmethod
    def create_agent_performance_bar(
        data: pd.DataFrame,
        agent_column: str = 'agent_id',
        metric: str = 'calls',
        top_n: int = 10,
        title: str = 'Top Agent Performance'
    ) -> go.Figure:
        """
        Create a bar chart showing top performing agents.

        Args:
            data: DataFrame with call data
            agent_column: Name of agent column
            metric: Performance metric ('calls', 'duration', 'revenue')
            top_n: Number of top agents to show
            title: Chart title

        Returns:
            Plotly Figure object
        """
        try:
            # Calculate metric by agent
            if metric == 'calls':
                agent_stats = data.groupby(agent_column).size().sort_values(ascending=False)
                y_title = 'Number of Calls'
            elif metric == 'duration':
                duration_totals = (
                    data.groupby(agent_column)['duration']
                    .sum()
                    .sort_values(ascending=False)
                )
                agent_stats = duration_totals / 3600
                y_title = 'Total Duration (hours)'
            elif metric == 'revenue':
                agent_stats = (
                    data.groupby(agent_column)['revenue']
                    .sum()
                    .sort_values(ascending=False)
                )
                y_title = 'Total Revenue ($)'
            else:
                agent_stats = data.groupby(agent_column).size().sort_values(ascending=False)
                y_title = 'Count'

            # Get top N agents
            top_agents = agent_stats.head(top_n)

            # Create bar chart
            fig = go.Figure(data=[go.Bar(
                x=top_agents.index,
                y=top_agents.values,
                marker_color=ChartTheme.COLORS['primary'],
                text=top_agents.values.round(1),
                textposition='outside'
            )])

            # Update layout
            layout = ChartTheme.get_layout_template()
            layout.update({
                'title': title,
                'xaxis_title': 'Agent',
                'yaxis_title': y_title,
                'xaxis': {'tickangle': -45}
            })
            fig.update_layout(layout)

            return fig

        except Exception as e:
            logger.error(f"Error creating agent performance chart: {e}")
            return go.Figure()

    @staticmethod
    def create_campaign_comparison(
        data: pd.DataFrame,
        campaign_column: str = 'campaign',
        metrics: list[str] | None = None,
        title: str = 'Campaign Performance Comparison'
    ) -> go.Figure:
        """
        Create a multi-metric comparison chart for campaigns.

        Args:
            data: DataFrame with call data
            campaign_column: Name of campaign column
            metrics: List of metrics to compare
            title: Chart title

        Returns:
            Plotly Figure object
        """
        try:
            metrics = metrics or ['calls', 'connection_rate', 'avg_duration']
            # Calculate metrics for each campaign
            campaign_stats = {}
            campaigns = data[campaign_column].unique()

            for campaign in campaigns:
                campaign_data = data[data[campaign_column] == campaign]
                stats = {
                    'calls': len(campaign_data),
                    'connection_rate': (campaign_data['outcome'] == 'connected').mean() * 100,
                    'avg_duration': campaign_data['duration'].mean() / 60,
                    'total_revenue': campaign_data['revenue'].sum()
                }
                campaign_stats[campaign] = stats

            # Create subplot figure
            fig = make_subplots(
                rows=1,
                cols=len(metrics),
                subplot_titles=metrics,
                specs=[[{'type': 'bar'} for _ in metrics]]
            )

            # Add traces for each metric
            for idx, metric in enumerate(metrics, 1):
                values = [campaign_stats[c].get(metric, 0) for c in campaigns]

                fig.add_trace(
                    go.Bar(
                        x=campaigns,
                        y=values,
                        name=metric,
                        marker_color=ChartTheme.COLOR_SEQUENCE[idx-1],
                        showlegend=False
                    ),
                    row=1,
                    col=idx
                )

            # Update layout
            layout = ChartTheme.get_layout_template()
            layout.update({
                'title': title,
                'showlegend': False,
                'height': 400
            })
            fig.update_layout(layout)

            # Update x-axis for all subplots
            for i in range(1, len(metrics) + 1):
                fig.update_xaxes(tickangle=-45, row=1, col=i)

            return fig

        except Exception as e:
            logger.error(f"Error creating campaign comparison: {e}")
            return go.Figure()


class TrendChart:
    """
    Creates trend analysis charts for identifying patterns
    and forecasting future metrics.
    """

    @staticmethod
    def create_moving_average_chart(
        data: pd.DataFrame,
        date_column: str = 'timestamp',
        value_column: str = 'calls',
        window_sizes: list[int] | None = None,
        title: str = 'Trend Analysis with Moving Averages'
    ) -> go.Figure:
        """
        Create a chart with moving averages for trend analysis.

        Args:
            data: DataFrame with time series data
            date_column: Name of date column
            value_column: Name of value column to analyze
            window_sizes: List of moving average window sizes
            title: Chart title

        Returns:
            Plotly Figure object
        """
        try:
            window_sizes = window_sizes or [7, 30]
            # Prepare data
            data[date_column] = pd.to_datetime(data[date_column])
            data = data.sort_values(date_column)

            # Aggregate daily values
            if value_column == 'calls':
                daily_data = data.resample('D', on=date_column).size()
            else:
                daily_data = data.resample('D', on=date_column)[value_column].mean()

            # Create figure
            fig = go.Figure()

            # Add actual values
            fig.add_trace(go.Scatter(
                x=daily_data.index,
                y=daily_data.values,
                mode='lines',
                name='Actual',
                line=dict(color=ChartTheme.COLORS['info'], width=1),
                opacity=0.5
            ))

            # Add moving averages
            colors = [
                ChartTheme.COLORS['primary'],
                ChartTheme.COLORS['secondary'],
                ChartTheme.COLORS['success'],
            ]

            for window, color in zip(window_sizes, colors, strict=False):
                ma = daily_data.rolling(window=window, min_periods=1).mean()

                fig.add_trace(go.Scatter(
                    x=ma.index,
                    y=ma.values,
                    mode='lines',
                    name=f'{window}-day MA',
                    line=dict(color=color, width=2)
                ))

            # Update layout
            layout = ChartTheme.get_layout_template()
            layout.update({
                'title': title,
                'xaxis_title': 'Date',
                'yaxis_title': 'Value',
                'hovermode': 'x unified',
                'showlegend': True
            })
            fig.update_layout(layout)

            return fig

        except Exception as e:
            logger.error(f"Error creating moving average chart: {e}")
            return go.Figure()


def render_chart_in_streamlit(
    chart_function: callable,
    container: Any,
    **kwargs
) -> None:
    """
    Helper function to render a chart in a Streamlit container.

    Args:
        chart_function: Chart creation function
        container: Streamlit container (column, expander, etc.)
        **kwargs: Arguments to pass to chart function
    """
    try:
        fig = chart_function(**kwargs)
        container.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        logger.error(f"Error rendering chart: {e}")
        container.error(f"Failed to render chart: {str(e)}")
