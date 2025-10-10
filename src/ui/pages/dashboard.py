"""
Dashboard Page Module for Call Analytics System

This module implements the main dashboard page of the application,
providing an overview of call metrics, performance indicators, and
real-time analytics with interactive visualizations.
"""

import logging
import sys
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.analysis.aggregations import MetricsCalculator
from src.core.storage_manager import StorageManager
from src.ui.components import (
    AgentPerformanceTable,
    CallRecordsTable,
    DateRangeFilter,
    DistributionChart,
    FilterState,
    KPIDashboard,
    PerformanceChart,
    TimeSeriesChart,
    TrendChart,
)

# Configure module logger
logger = logging.getLogger(__name__)


class DashboardPage:
    """
    Main dashboard page providing comprehensive overview of call analytics
    with real-time metrics, charts, and performance indicators.
    """

    def __init__(self, storage_manager: StorageManager):
        """
        Initialize dashboard page with storage manager.

        Args:
            storage_manager: Storage manager instance for data access
        """
        self.storage_manager = storage_manager
        self.metrics_calculator = MetricsCalculator()

    def render(self) -> None:
        """
        Render the complete dashboard page with all components.
        """
        try:
            # Page header
            st.title("📊 Call Analytics Dashboard")
            st.markdown(
                "Real-time insights and performance metrics for your call center operations"
            )

            # Date range filter in sidebar
            with st.sidebar:
                st.header("Dashboard Filters")
                start_date, end_date = DateRangeFilter.render(
                    key_prefix="dashboard", default_range="Last 7 Days"
                )

                # Additional filters
                st.subheader("Advanced Filters")
                filter_state = self._render_sidebar_filters(start_date, end_date)

            # Load and filter data
            data = self._load_dashboard_data(filter_state)

            if data.empty:
                st.warning(
                    "No data available for the selected period. Adjust your filters or "
                    "upload call data."
                )
                self._render_empty_state()
                return

            # Calculate comparison period for trends
            comparison_data = self._load_comparison_data(filter_state)

            # Render dashboard sections
            self._render_kpi_section(data, comparison_data)
            st.divider()

            self._render_charts_section(data)
            st.divider()

            self._render_performance_section(data)
            st.divider()

            self._render_recent_calls_section(data)

            # Auto-refresh option
            self._setup_auto_refresh()

        except Exception as e:
            logger.error(f"Error rendering dashboard: {e}")
            st.error(f"Failed to load dashboard: {str(e)}")

    def _render_sidebar_filters(self, start_date, end_date) -> FilterState:
        """
        Render additional filters in the sidebar.

        Args:
            start_date: Filter start date
            end_date: Filter end date

        Returns:
            FilterState with all filter selections
        """
        filter_state = FilterState(date_range=(start_date, end_date))

        # Campaign filter
        campaigns = self.storage_manager.get_unique_values("campaign")
        if campaigns:
            filter_state.selected_campaigns = st.multiselect(
                "Campaigns", options=campaigns, default=[], key="dashboard_campaigns"
            )

        # Agent filter
        agents = self.storage_manager.get_unique_values("agent_id")
        if agents:
            filter_state.selected_agents = st.multiselect(
                "Agents", options=agents, default=[], key="dashboard_agents"
            )

        # Outcome filter
        outcomes = self.storage_manager.get_unique_values("outcome")
        if outcomes:
            filter_state.selected_outcomes = st.multiselect(
                "Call Outcomes", options=outcomes, default=[], key="dashboard_outcomes"
            )

        return filter_state

    def _load_dashboard_data(self, filter_state: FilterState) -> pd.DataFrame:
        """
        Load and filter data for the dashboard.

        Args:
            filter_state: Current filter state

        Returns:
            Filtered DataFrame
        """
        try:
            # Load data from storage
            data = self.storage_manager.load_call_records(
                start_date=filter_state.date_range[0], end_date=filter_state.date_range[1]
            )

            # Apply additional filters
            if data is not None and not data.empty:
                data = filter_state.apply_to_dataframe(data)

            return data if data is not None else pd.DataFrame()

        except Exception as e:
            logger.error(f"Error loading dashboard data: {e}")
            return pd.DataFrame()

    def _load_comparison_data(self, filter_state: FilterState) -> pd.DataFrame:
        """
        Load data for the comparison period (previous period of same length).

        Args:
            filter_state: Current filter state

        Returns:
            Comparison period DataFrame
        """
        try:
            # Calculate previous period dates
            period_length = (filter_state.date_range[1] - filter_state.date_range[0]).days
            prev_end = filter_state.date_range[0] - timedelta(days=1)
            prev_start = prev_end - timedelta(days=period_length)

            # Create comparison filter state
            comparison_filter = FilterState(
                date_range=(prev_start, prev_end),
                selected_agents=filter_state.selected_agents,
                selected_campaigns=filter_state.selected_campaigns,
                selected_outcomes=filter_state.selected_outcomes,
            )

            # Load comparison data
            return self._load_dashboard_data(comparison_filter)

        except Exception as e:
            logger.error(f"Error loading comparison data: {e}")
            return pd.DataFrame()

    def _render_kpi_section(self, data: pd.DataFrame, comparison_data: pd.DataFrame) -> None:
        """
        Render the KPI metrics section at the top of the dashboard.

        Args:
            data: Current period data
            comparison_data: Previous period data for comparison
        """
        st.header("📈 Key Performance Indicators")

        # Create two rows of KPIs
        col1, col2 = st.columns(2)

        with col1:
            KPIDashboard.render_call_metrics(data, st.container(), comparison_data)

        with col2:
            KPIDashboard.render_revenue_metrics(data, st.container(), comparison_data)

    def _render_charts_section(self, data: pd.DataFrame) -> None:
        """
        Render the charts section with time series and distribution visualizations.

        Args:
            data: DataFrame with call data
        """
        st.header("📊 Analytics Overview")

        # Create tabs for different chart types
        tab1, tab2, tab3, tab4 = st.tabs(["Time Series", "Distribution", "Peak Hours", "Trends"])

        with tab1:
            # Time series chart
            col1, col2 = st.columns([3, 1])
            with col2:
                aggregation = st.radio(
                    "Aggregation", ["hourly", "daily", "weekly"], index=1, key="ts_aggregation"
                )
                group_by = st.selectbox(
                    "Group By", ["None", "outcome", "campaign", "agent_id"], key="ts_groupby"
                )

            with col1:
                fig = TimeSeriesChart.create_call_volume_chart(
                    data, aggregation=aggregation, group_by=None if group_by == "None" else group_by
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # Distribution charts
            col1, col2 = st.columns(2)

            with col1:
                if "duration" in data.columns:
                    fig = DistributionChart.create_duration_distribution(data)
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                if "outcome" in data.columns:
                    fig = DistributionChart.create_outcome_pie_chart(data)
                    st.plotly_chart(fig, use_container_width=True)

        with tab3:
            # Peak hours heatmap
            fig = TimeSeriesChart.create_peak_hours_heatmap(data)
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            # Trend analysis
            window_sizes = st.multiselect(
                "Moving Average Windows", [3, 7, 14, 30], default=[7, 30], key="ma_windows"
            )

            fig = TrendChart.create_moving_average_chart(data, window_sizes=window_sizes)
            st.plotly_chart(fig, use_container_width=True)

    def _render_performance_section(self, data: pd.DataFrame) -> None:
        """
        Render the performance analysis section.

        Args:
            data: DataFrame with call data
        """
        st.header("🎯 Performance Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Top Agents")
            if "agent_id" in data.columns:
                fig = PerformanceChart.create_agent_performance_bar(data, metric="calls", top_n=5)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Campaign Performance")
            if "campaign" in data.columns:
                fig = PerformanceChart.create_campaign_comparison(
                    data, metrics=["calls", "connection_rate"]
                )
                st.plotly_chart(fig, use_container_width=True)

        # Agent performance table
        with st.expander("Detailed Agent Performance", expanded=False):
            AgentPerformanceTable.render(data, st.container())

    def _render_recent_calls_section(self, data: pd.DataFrame) -> None:
        """
        Render the recent calls section.

        Args:
            data: DataFrame with call data
        """
        st.header("📞 Recent Calls")

        # Get most recent calls
        if "timestamp" in data.columns:
            recent_calls = data.nlargest(10, "timestamp")
        else:
            recent_calls = data.head(10)

        # Display calls table
        selected_call = CallRecordsTable.render(
            recent_calls, st.container(), show_actions=True, show_transcript=False
        )

        # Show call details if selected
        if selected_call:
            with st.expander("Call Details", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    for key in ["call_id", "phone_number", "timestamp", "duration"]:
                        if key in selected_call:
                            st.write(f"**{key.replace('_', ' ').title()}:** {selected_call[key]}")

                with col2:
                    for key in ["outcome", "agent_id", "campaign", "revenue"]:
                        if key in selected_call:
                            st.write(f"**{key.replace('_', ' ').title()}:** {selected_call[key]}")

                if "notes" in selected_call and selected_call["notes"]:
                    st.write("**Notes:**")
                    st.write(selected_call["notes"])

    def _render_empty_state(self) -> None:
        """
        Render empty state when no data is available.
        """
        st.info(
            """
        ### No Data Available

        To get started with the dashboard:
        1. Upload call data using the Upload page
        2. Process audio files for transcription
        3. Configure your analytics settings

        Once data is available, you'll see:
        - Real-time KPI metrics
        - Interactive charts and visualizations
        - Agent performance analytics
        - Call outcome distributions
        """
        )

    def _setup_auto_refresh(self) -> None:
        """
        Setup auto-refresh functionality for the dashboard.
        """
        with st.sidebar:
            st.divider()
            auto_refresh = st.checkbox("Auto-refresh", key="dashboard_auto_refresh")

            if auto_refresh:
                refresh_interval = st.slider(
                    "Refresh interval (seconds)",
                    min_value=5,
                    max_value=60,
                    value=30,
                    key="dashboard_refresh_interval",
                )

                # Use Streamlit's automatic rerun
                st.info(f"Dashboard will refresh every {refresh_interval} seconds")
                time.sleep(refresh_interval)
                st.rerun()


def render_dashboard_page(storage_manager: StorageManager) -> None:
    """
    Main entry point for rendering the dashboard page.

    Args:
        storage_manager: Storage manager instance
    """
    dashboard = DashboardPage(storage_manager)
    dashboard.render()
