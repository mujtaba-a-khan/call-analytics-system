"""
Analysis Page Module for Call Analytics System

This module implements the analysis page for deep-dive analytics,
custom queries, semantic search, and advanced filtering capabilities
with export functionality.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Import components
from ..components import (
    FilterState, DateRangeFilter, MultiSelectFilter,
    SearchFilter, RangeSliderFilter, QuickFilters,
    DataTable, ComparisonTable,
    TimeSeriesChart, DistributionChart,
    save_filter_preset, load_filter_preset
)

# Import analysis modules
from ...analysis.semantic_search import SemanticSearchEngine
from ...analysis.query_interpreter import QueryInterpreter
from ...analysis.aggregations import MetricsCalculator
from ...analysis.filters import AdvancedFilters
from ...core.storage_manager import StorageManager

# Configure module logger
logger = logging.getLogger(__name__)


class AnalysisPage:
    """
    Advanced analysis page for deep-dive exploration of call data
    with semantic search, custom queries, and detailed analytics.
    """
    
    def __init__(self, storage_manager: StorageManager, vector_store=None):
        """
        Initialize analysis page with required components.
        
        Args:
            storage_manager: Storage manager instance
            vector_store: Optional vector store for semantic search
        """
        self.storage_manager = storage_manager
        self.vector_store = vector_store
        self.metrics_calculator = MetricsCalculator()
        self.advanced_filters = AdvancedFilters()
        
        # Initialize semantic search if vector store available
        if vector_store:
            self.search_engine = SemanticSearchEngine(vector_store)
            self.query_interpreter = QueryInterpreter()
        else:
            self.search_engine = None
            self.query_interpreter = None
    
    def render(self) -> None:
        """
        Render the complete analysis page with all components.
        """
        try:
            # Page header
            st.title("🔍 Advanced Analysis")
            st.markdown("Explore your call data with powerful search, filtering, and analytics tools")
            
            # Create tabs for different analysis modes
            tab1, tab2, tab3, tab4 = st.tabs([
                "🔎 Semantic Search",
                "📊 Custom Analysis",
                "🆚 Comparison",
                "📈 Cohort Analysis"
            ])
            
            with tab1:
                self._render_semantic_search_tab()
            
            with tab2:
                self._render_custom_analysis_tab()
            
            with tab3:
                self._render_comparison_tab()
            
            with tab4:
                self._render_cohort_analysis_tab()
            
        except Exception as e:
            logger.error(f"Error rendering analysis page: {e}")
            st.error(f"Failed to load analysis page: {str(e)}")
    
    def _render_semantic_search_tab(self) -> None:
        """
        Render the semantic search tab for natural language queries.
        """
        st.header("Semantic Search")
        st.markdown("Search your call transcripts and notes using natural language")
        
        if not self.search_engine:
            st.warning("Semantic search requires vector database setup. Please configure the vector store.")
            return
        
        # Search interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_area(
                "Enter your search query",
                placeholder="e.g., 'Find calls where customers complained about billing issues'",
                height=100,
                key="semantic_query"
            )
        
        with col2:
            st.write("**Search Options**")
            
            top_k = st.number_input(
                "Results to return",
                min_value=1,
                max_value=100,
                value=10,
                key="search_top_k"
            )
            
            similarity_threshold = st.slider(
                "Similarity threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                key="search_threshold"
            )
        
        # Search filters
        with st.expander("Advanced Filters", expanded=False):
            filter_state = self._render_search_filters()
        
        # Execute search
        if st.button("🔍 Search", type="primary", use_container_width=True):
            if query:
                self._execute_semantic_search(query, top_k, similarity_threshold, filter_state)
            else:
                st.warning("Please enter a search query")
        
        # Show example queries
        with st.expander("Example Queries"):
            st.markdown("""
            - "Find calls where customers were frustrated or angry"
            - "Show me successful sales calls with high revenue"
            - "Calls mentioning technical issues with the product"
            - "Customer requesting refund or cancellation"
            - "Positive feedback about customer service"
            """)
    
    def _render_custom_analysis_tab(self) -> None:
        """
        Render the custom analysis tab for detailed data exploration.
        """
        st.header("Custom Analysis")
        st.markdown("Build custom queries and explore data with advanced filters")
        
        # Filter configuration
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Configure Filters")
            filter_state = self._render_comprehensive_filters()
        
        with col2:
            st.subheader("Filter Presets")
            self._render_filter_presets(filter_state)
        
        # Apply filters and load data
        if st.button("Apply Filters", type="primary", use_container_width=True):
            data = self._apply_custom_filters(filter_state)
            
            if not data.empty:
                # Display results
                st.success(f"Found {len(data)} records matching your criteria")
                
                # Analysis options
                analysis_type = st.selectbox(
                    "Select Analysis Type",
                    ["Summary Statistics", "Time Analysis", "Agent Analysis", 
                     "Outcome Analysis", "Custom Aggregation"]
                )
                
                self._render_analysis_results(data, analysis_type)
                
                # Export options
                self._render_export_options(data)
            else:
                st.warning("No records found matching your filters")
    
    def _render_comparison_tab(self) -> None:
        """
        Render the comparison tab for period-over-period analysis.
        """
        st.header("Period Comparison")
        st.markdown("Compare metrics between different time periods or segments")
        
        # Period selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Period 1")
            period1_start, period1_end = DateRangeFilter.render(
                container=st.container(),
                key_prefix="period1",
                default_range="Last 30 Days"
            )
        
        with col2:
            st.subheader("Period 2")
            period2_start, period2_end = DateRangeFilter.render(
                container=st.container(),
                key_prefix="period2",
                default_range="Last 7 Days"
            )
        
        # Comparison dimensions
        st.subheader("Comparison Settings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            group_by = st.selectbox(
                "Group By",
                ["agent_id", "campaign", "outcome", "call_type"],
                key="comparison_group"
            )
        
        with col2:
            metrics = st.multiselect(
                "Metrics to Compare",
                ["count", "duration", "revenue", "connection_rate"],
                default=["count", "revenue"],
                key="comparison_metrics"
            )
        
        with col3:
            chart_type = st.selectbox(
                "Visualization",
                ["Bar Chart", "Line Chart", "Table"],
                key="comparison_chart"
            )
        
        # Execute comparison
        if st.button("Compare Periods", type="primary", use_container_width=True):
            self._execute_period_comparison(
                (period1_start, period1_end),
                (period2_start, period2_end),
                group_by, metrics, chart_type
            )
    
    def _render_cohort_analysis_tab(self) -> None:
        """
        Render the cohort analysis tab for retention and behavior analysis.
        """
        st.header("Cohort Analysis")
        st.markdown("Analyze customer behavior patterns over time")
        
        # Cohort configuration
        col1, col2 = st.columns(2)
        
        with col1:
            cohort_type = st.selectbox(
                "Cohort Type",
                ["First Call Date", "Campaign Start", "Agent Assignment"],
                key="cohort_type"
            )
            
            cohort_period = st.selectbox(
                "Cohort Period",
                ["Daily", "Weekly", "Monthly"],
                key="cohort_period"
            )
        
        with col2:
            metric = st.selectbox(
                "Metric to Track",
                ["Retention Rate", "Average Duration", "Revenue per Cohort", "Call Frequency"],
                key="cohort_metric"
            )
            
            periods_to_analyze = st.slider(
                "Periods to Analyze",
                min_value=3,
                max_value=12,
                value=6,
                key="cohort_periods"
            )
        
        # Date range for cohort analysis
        st.subheader("Analysis Period")
        start_date, end_date = DateRangeFilter.render(
            key_prefix="cohort",
            default_range="Last 90 Days"
        )
        
        # Execute cohort analysis
        if st.button("Generate Cohort Analysis", type="primary", use_container_width=True):
            self._execute_cohort_analysis(
                cohort_type, cohort_period, metric,
                periods_to_analyze, start_date, end_date
            )
    
    def _render_search_filters(self) -> FilterState:
        """
        Render filters for semantic search.
        
        Returns:
            FilterState with search filters
        """
        filter_state = FilterState()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Date range filter
            filter_state.date_range = DateRangeFilter.render(
                container=st.container(),
                key_prefix="search_date"
            )
        
        with col2:
            # Campaign filter
            campaigns = self.storage_manager.get_unique_values('campaign')
            if campaigns:
                filter_state.selected_campaigns = st.multiselect(
                    "Campaigns",
                    options=campaigns,
                    key="search_campaigns"
                )
        
        with col3:
            # Agent filter
            agents = self.storage_manager.get_unique_values('agent_id')
            if agents:
                filter_state.selected_agents = st.multiselect(
                    "Agents",
                    options=agents,
                    key="search_agents"
                )
        
        return filter_state
    
    def _render_comprehensive_filters(self) -> FilterState:
        """
        Render comprehensive filter options for custom analysis.
        
        Returns:
            FilterState with all filter selections
        """
        filter_state = FilterState()
        
        # Date range
        filter_state.date_range = DateRangeFilter.render(
            key_prefix="custom_date",
            default_range="Last 30 Days"
        )
        
        # Multi-select filters
        col1, col2 = st.columns(2)
        
        with col1:
            # Agent filter
            agents = self.storage_manager.get_unique_values('agent_id')
            if agents:
                filter_state.selected_agents = MultiSelectFilter.render(
                    label="Agents",
                    options=agents,
                    key="custom_agents"
                )
            
            # Campaign filter
            campaigns = self.storage_manager.get_unique_values('campaign')
            if campaigns:
                filter_state.selected_campaigns = MultiSelectFilter.render(
                    label="Campaigns",
                    options=campaigns,
                    key="custom_campaigns"
                )
        
        with col2:
            # Outcome filter
            outcomes = self.storage_manager.get_unique_values('outcome')
            if outcomes:
                filter_state.selected_outcomes = MultiSelectFilter.render(
                    label="Outcomes",
                    options=outcomes,
                    key="custom_outcomes"
                )
            
            # Call type filter
            call_types = self.storage_manager.get_unique_values('call_type')
            if call_types:
                filter_state.selected_types = MultiSelectFilter.render(
                    label="Call Types",
                    options=call_types,
                    key="custom_types"
                )
        
        # Range filters
        st.subheader("Range Filters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Duration filter
            filter_state.duration_range = RangeSliderFilter.render(
                label="Call Duration (seconds)",
                min_value=0.0,
                max_value=3600.0,
                default_range=(0.0, 3600.0),
                key="custom_duration",
                step=60.0
            )
        
        with col2:
            # Revenue filter
            filter_state.revenue_range = RangeSliderFilter.render(
                label="Revenue ($)",
                min_value=0.0,
                max_value=1000.0,
                default_range=(0.0, 1000.0),
                key="custom_revenue",
                step=10.0
            )
        
        # Text search
        filter_state.search_query = SearchFilter.render(
            key="custom_search",
            placeholder="Search in notes and transcripts..."
        )
        
        return filter_state
    
    def _render_filter_presets(self, current_state: FilterState) -> None:
        """
        Render filter preset management interface.
        
        Args:
            current_state: Current filter state
        """
        # Load preset
        presets = st.session_state.get('filter_presets', {})
        if presets:
            selected_preset = st.selectbox(
                "Load Preset",
                [""] + list(presets.keys()),
                key="load_preset"
            )
            
            if selected_preset and st.button("Load", key="load_preset_btn"):
                loaded_state = load_filter_preset(selected_preset)
                if loaded_state:
                    st.success(f"Loaded preset: {selected_preset}")
                    st.rerun()
        
        # Save preset
        st.divider()
        preset_name = st.text_input(
            "Save Current Filters",
            placeholder="Enter preset name",
            key="save_preset_name"
        )
        
        if st.button("Save Preset", key="save_preset_btn"):
            if preset_name:
                save_filter_preset(preset_name, current_state)
            else:
                st.warning("Please enter a preset name")
    
    def _execute_semantic_search(self, 
                                 query: str,
                                 top_k: int,
                                 threshold: float,
                                 filter_state: FilterState) -> None:
        """
        Execute semantic search and display results.
        
        Args:
            query: Search query
            top_k: Number of results
            threshold: Similarity threshold
            filter_state: Additional filters
        """
        try:
            with st.spinner("Searching..."):
                # Perform semantic search
                results = self.search_engine.search(
                    query=query,
                    top_k=top_k,
                    threshold=threshold,
                    filters=filter_state.to_dict()
                )
                
                if results:
                    st.success(f"Found {len(results)} relevant calls")
                    
                    # Display results
                    for idx, result in enumerate(results, 1):
                        with st.expander(f"Result {idx} - Score: {result['score']:.3f}"):
                            # Display call details
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Call ID:** {result['call_id']}")
                                st.write(f"**Date:** {result['timestamp']}")
                                st.write(f"**Agent:** {result['agent_id']}")
                            
                            with col2:
                                st.write(f"**Duration:** {result['duration']} seconds")
                                st.write(f"**Outcome:** {result['outcome']}")
                                st.write(f"**Campaign:** {result['campaign']}")
                            
                            # Display matched content
                            st.write("**Matched Content:**")
                            st.write(result.get('matched_text', 'N/A'))
                else:
                    st.warning("No results found matching your query")
                    
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            st.error(f"Search failed: {str(e)}")
    
    def _apply_custom_filters(self, filter_state: FilterState) -> pd.DataFrame:
        """
        Apply custom filters and return filtered data.
        
        Args:
            filter_state: Filter configuration
            
        Returns:
            Filtered DataFrame
        """
        try:
            # Load base data
            data = self.storage_manager.load_call_records(
                start_date=filter_state.date_range[0],
                end_date=filter_state.date_range[1]
            )
            
            if data is not None and not data.empty:
                # Apply filters
                data = filter_state.apply_to_dataframe(data)
            
            return data if data is not None else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            return pd.DataFrame()
    
    def _render_analysis_results(self, data: pd.DataFrame, analysis_type: str) -> None:
        """
        Render analysis results based on selected type.
        
        Args:
            data: Filtered data
            analysis_type: Type of analysis to perform
        """
        if analysis_type == "Summary Statistics":
            # Calculate and display summary stats
            metrics = self.metrics_calculator.calculate_all_metrics(data)
            
            # Display metrics in expandable sections
            for category, values in metrics.items():
                with st.expander(f"{category.replace('_', ' ').title()}", expanded=True):
                    cols = st.columns(3)
                    for idx, (key, value) in enumerate(values.items()):
                        col_idx = idx % 3
                        cols[col_idx].metric(
                            label=key.replace('_', ' ').title(),
                            value=f"{value:.2f}" if isinstance(value, float) else value
                        )
        
        elif analysis_type == "Time Analysis":
            # Time-based analysis
            fig = TimeSeriesChart.create_call_volume_chart(
                data,
                aggregation='daily',
                title="Call Volume Trend"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Agent Analysis":
            # Agent performance analysis
            AgentPerformanceTable.render(data, st.container())
        
        elif analysis_type == "Outcome Analysis":
            # Outcome distribution
            fig = DistributionChart.create_outcome_pie_chart(data)
            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Custom Aggregation":
            # Custom aggregation interface
            self._render_custom_aggregation(data)
    
    def _render_custom_aggregation(self, data: pd.DataFrame) -> None:
        """
        Render custom aggregation interface.
        
        Args:
            data: Data to aggregate
        """
        col1, col2, col3 = st.columns(3)
        
        with col1:
            group_by = st.selectbox(
                "Group By",
                data.columns.tolist(),
                key="agg_group"
            )
        
        with col2:
            agg_column = st.selectbox(
                "Aggregate Column",
                data.select_dtypes(include=[np.number]).columns.tolist(),
                key="agg_column"
            )
        
        with col3:
            agg_function = st.selectbox(
                "Function",
                ["sum", "mean", "median", "count", "min", "max"],
                key="agg_function"
            )
        
        # Perform aggregation
        result = data.groupby(group_by)[agg_column].agg(agg_function).reset_index()
        
        # Display result
        st.dataframe(result, use_container_width=True)
        
        # Visualization
        fig = px.bar(
            result,
            x=group_by,
            y=agg_column,
            title=f"{agg_function.title()} of {agg_column} by {group_by}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_export_options(self, data: pd.DataFrame) -> None:
        """
        Render export options for analysis results.
        
        Args:
            data: Data to export
        """
        st.divider()
        st.subheader("Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Export to CSV"):
                csv = data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"analysis_results_{datetime.now():%Y%m%d_%H%M%S}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📊 Export to Excel"):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    data.to_excel(writer, index=False, sheet_name='Analysis')
                excel_data = output.getvalue()
                
                st.download_button(
                    label="Download Excel",
                    data=excel_data,
                    file_name=f"analysis_results_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col3:
            if st.button("📋 Copy to Clipboard"):
                # Note: This is a placeholder as clipboard functionality
                # requires JavaScript integration
                st.info("Data ready for copying")
                st.code(data.to_string(), language=None)
    
    def _execute_period_comparison(self,
                                   period1: Tuple[datetime, datetime],
                                   period2: Tuple[datetime, datetime],
                                   group_by: str,
                                   metrics: List[str],
                                   chart_type: str) -> None:
        """
        Execute period comparison analysis.
        
        Args:
            period1: First period (start, end)
            period2: Second period (start, end)
            group_by: Grouping column
            metrics: Metrics to compare
            chart_type: Visualization type
        """
        try:
            # Load data for both periods
            data1 = self.storage_manager.load_call_records(
                start_date=period1[0],
                end_date=period1[1]
            )
            
            data2 = self.storage_manager.load_call_records(
                start_date=period2[0],
                end_date=period2[1]
            )
            
            if data1.empty or data2.empty:
                st.warning("Insufficient data for comparison")
                return
            
            # Display comparison
            ComparisonTable.render(
                current_data=data2,
                previous_data=data1,
                group_column=group_by,
                metrics=metrics,
                container=st.container()
            )
            
        except Exception as e:
            logger.error(f"Error in period comparison: {e}")
            st.error(f"Comparison failed: {str(e)}")
    
    def _execute_cohort_analysis(self,
                                 cohort_type: str,
                                 cohort_period: str,
                                 metric: str,
                                 periods: int,
                                 start_date: datetime,
                                 end_date: datetime) -> None:
        """
        Execute cohort analysis.
        
        Args:
            cohort_type: Type of cohort
            cohort_period: Cohort period granularity
            metric: Metric to track
            periods: Number of periods to analyze
            start_date: Analysis start date
            end_date: Analysis end date
        """
        try:
            # Load data
            data = self.storage_manager.load_call_records(
                start_date=start_date,
                end_date=end_date
            )
            
            if data.empty:
                st.warning("No data available for cohort analysis")
                return
            
            # Perform cohort analysis (simplified version)
            st.info(f"Cohort analysis for {metric} by {cohort_type}")
            
            # Create cohort visualization
            # This is a placeholder for actual cohort analysis logic
            st.write("Cohort analysis results would appear here")
            
        except Exception as e:
            logger.error(f"Error in cohort analysis: {e}")
            st.error(f"Cohort analysis failed: {str(e)}")


def render_analysis_page(storage_manager: StorageManager, vector_store=None) -> None:
    """
    Main entry point for rendering the analysis page.
    
    Args:
        storage_manager: Storage manager instance
        vector_store: Optional vector store for semantic search
    """
    analysis = AnalysisPage(storage_manager, vector_store)
    analysis.render()