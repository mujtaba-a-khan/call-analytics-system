"""
Table Display Components for Call Analytics System

This module provides reusable Streamlit table components for displaying
call records, analytics results, and agent performance data with sorting,
filtering, pagination, and export capabilities.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from urllib.parse import urlencode
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import base64
from io import BytesIO
import xlsxwriter

# Configure module logger
logger = logging.getLogger(__name__)


class DataTable:
    """
    Enhanced data table component with sorting, filtering, and
    pagination capabilities for large datasets.
    """
    
    @classmethod
    def render(cls,
               data: pd.DataFrame,
               container: Any = None,
               page_size: int = 20,
               key: str = 'data_table',
               show_index: bool = False,
               column_config: Optional[Dict[str, Any]] = None,
               height: Optional[int] = None,
               enable_search: bool = True,
               enable_export: bool = True) -> pd.DataFrame:
        """
        Render an interactive data table with advanced features.
        
        Args:
            data: DataFrame to display
            container: Streamlit container to render in
            page_size: Number of rows per page
            key: Component key for state management
            show_index: Whether to show row index
            column_config: Column configuration dictionary
            height: Fixed height for the table
            enable_search: Whether to enable search functionality
            enable_export: Whether to enable export functionality
            
        Returns:
            Filtered/sorted DataFrame based on user interactions
        """
        container = container or st
        
        # Initialize session state for pagination
        if f'{key}_page' not in st.session_state:
            st.session_state[f'{key}_page'] = 0
        
        # Search functionality
        filtered_data = data.copy()
        if enable_search:
            search_query = container.text_input(
                "🔍 Search table",
                key=f"{key}_search",
                placeholder="Type to search..."
            )
            
            if search_query:
                # Search across all string columns
                mask = pd.Series([False] * len(filtered_data))
                for col in filtered_data.select_dtypes(include=['object']).columns:
                    mask |= filtered_data[col].astype(str).str.contains(
                        search_query, case=False, na=False
                    )
                filtered_data = filtered_data[mask]
        
        # Calculate pagination
        total_rows = len(filtered_data)
        total_pages = (total_rows - 1) // page_size + 1 if total_rows > 0 else 1
        current_page = st.session_state[f'{key}_page']
        
        # Ensure current page is valid
        if current_page >= total_pages:
            current_page = total_pages - 1
            st.session_state[f'{key}_page'] = current_page
        
        # Slice data for current page
        start_idx = current_page * page_size
        end_idx = min(start_idx + page_size, total_rows)
        page_data = filtered_data.iloc[start_idx:end_idx]
        
        # Display table info
        col1, col2, col3 = container.columns([2, 3, 2])
        with col1:
            st.caption(f"Showing {start_idx + 1}-{end_idx} of {total_rows} rows")
        
        # Pagination controls
        with col2:
            subcol1, subcol2, subcol3, subcol4, subcol5 = st.columns(5)
            
            with subcol1:
                if st.button("⏮️", key=f"{key}_first", disabled=(current_page == 0)):
                    st.session_state[f'{key}_page'] = 0
                    st.rerun()
            
            with subcol2:
                if st.button("◀️", key=f"{key}_prev", disabled=(current_page == 0)):
                    st.session_state[f'{key}_page'] = current_page - 1
                    st.rerun()
            
            with subcol3:
                st.caption(f"Page {current_page + 1}/{total_pages}")
            
            with subcol4:
                if st.button("▶️", key=f"{key}_next", disabled=(current_page >= total_pages - 1)):
                    st.session_state[f'{key}_page'] = current_page + 1
                    st.rerun()
            
            with subcol5:
                if st.button("⏭️", key=f"{key}_last", disabled=(current_page >= total_pages - 1)):
                    st.session_state[f'{key}_page'] = total_pages - 1
                    st.rerun()
        
        # Export controls
        if enable_export:
            with col3:
                export_format = st.selectbox(
                    "Export",
                    ["", "CSV", "Excel", "JSON"],
                    key=f"{key}_export_format"
                )
                
                if export_format:
                    cls._handle_export(filtered_data, export_format, key)
        
        # Display the table
        st.dataframe(
            page_data,
            use_container_width=True,
            hide_index=not show_index,
            column_config=column_config,
            height=height
        )
        
        return filtered_data
    
    @staticmethod
    def _handle_export(data: pd.DataFrame, format: str, key: str) -> None:
        """
        Handle data export in various formats.
        
        Args:
            data: DataFrame to export
            format: Export format ('CSV', 'Excel', 'JSON')
            key: Component key for unique download button
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "CSV":
            csv = data.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="export_{timestamp}.csv">Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
            
        elif format == "Excel":
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                data.to_excel(writer, index=False, sheet_name='Data')
            excel_data = output.getvalue()
            b64 = base64.b64encode(excel_data).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="export_{timestamp}.xlsx">Download Excel</a>'
            st.markdown(href, unsafe_allow_html=True)
            
        elif format == "JSON":
            json_str = data.to_json(orient='records', indent=2)
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="export_{timestamp}.json">Download JSON</a>'
            st.markdown(href, unsafe_allow_html=True)


class CallRecordsTable:
    """
    Specialized table component for displaying call records with
    custom formatting and action buttons.
    """
    
    @classmethod
    def render(cls,
               records: pd.DataFrame,
               container: Any = None,
               show_actions: bool = True,
               show_transcript: bool = False) -> Optional[Dict[str, Any]]:
        """
        Render call records table with specialized formatting.
        
        Args:
            records: DataFrame with call records
            container: Streamlit container to render in
            show_actions: Whether to show action buttons
            show_transcript: Whether to show transcript column
            
        Returns:
            Selected record details if action clicked
        """
        container = container or st

        record_session_key = 'recent_calls_selected_record'

        if records is None or records.empty:
            if 'view_call' in st.query_params.keys():
                st.query_params.pop('view_call', None)
            st.session_state.pop(record_session_key, None)
            return None

        # Normalize indices to ensure consistent row mapping
        records = records.reset_index(drop=True)

        # Prepare display columns
        display_columns = ['call_id', 'phone_number', 'timestamp', 'duration', 
                          'outcome', 'agent_id', 'campaign']
        
        if show_transcript and 'transcript' in records.columns:
            display_columns.append('transcript')
        
        if 'revenue' in records.columns:
            display_columns.append('revenue')
        
        # Filter to available columns
        display_columns = [col for col in display_columns if col in records.columns]
        display_data = records[display_columns].copy()
        
        # Format columns for display
        if 'timestamp' in display_data.columns:
            display_data['timestamp'] = pd.to_datetime(display_data['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        
        if 'duration' in display_data.columns:
            display_data['duration'] = display_data['duration'].apply(cls._format_duration)
        
        if 'revenue' in display_data.columns:
            display_data['revenue'] = display_data['revenue'].apply(lambda x: f"${x:.2f}" if x > 0 else "-")
        
        if 'transcript' in display_data.columns:
            display_data['transcript'] = display_data['transcript'].str[:100] + "..."
        
        # Column configuration
        column_config = {
            'call_id': st.column_config.TextColumn('Call ID', width='small'),
            'phone_number': st.column_config.TextColumn('Phone', width='medium'),
            'timestamp': st.column_config.TextColumn('Date/Time', width='medium'),
            'duration': st.column_config.TextColumn('Duration', width='small'),
            'outcome': st.column_config.TextColumn('Outcome', width='small'),
            'agent_id': st.column_config.TextColumn('Agent', width='small'),
            'campaign': st.column_config.TextColumn('Campaign', width='medium'),
            'revenue': st.column_config.TextColumn('Revenue', width='small'),
            'transcript': st.column_config.TextColumn('Transcript Preview', width='large')
        }

        def _first_value(value: Any) -> Optional[str]:
            if value is None:
                return None
            if isinstance(value, list):
                return value[0] if value else None
            return str(value)

        def _sync_selection(record: Dict[str, Any]) -> None:
            st.session_state[record_session_key] = record
            call_id_value = record.get('call_id')
            if not call_id_value:
                return
            current_param = _first_value(st.query_params.get('view_call'))
            if current_param != call_id_value:
                st.query_params['view_call'] = call_id_value

        selected_record: Optional[Dict[str, Any]] = None

        view_call_param = _first_value(st.query_params.get('view_call'))
        if view_call_param and 'call_id' in records.columns:
            matches = records.index[records['call_id'] == view_call_param].tolist()
            if matches:
                selected_record = records.iloc[matches[0]].to_dict()
                _sync_selection(selected_record)
            else:
                st.query_params.pop('view_call', None)
                st.session_state.pop(record_session_key, None)

        if selected_record is None:
            cached_record = st.session_state.get(record_session_key)
            if cached_record and cached_record.get('call_id') in records['call_id'].values:
                selected_record = cached_record
                _sync_selection(selected_record)
            else:
                st.session_state.pop(record_session_key, None)

        if show_actions and 'call_id' in records.columns:
            def _build_view_url(call_id: Any) -> str:
                call_id_str = str(call_id) if call_id is not None else ''
                if not call_id_str:
                    return ''
                query_pairs: List[Tuple[str, str]] = []
                for key in st.query_params.keys():
                    if key == 'view_call':
                        continue
                    value = st.query_params.get(key)
                    if isinstance(value, list):
                        query_pairs.extend((key, str(item)) for item in value if item is not None)
                    elif value is not None:
                        query_pairs.append((key, str(value)))
                query_pairs.append(('view_call', call_id_str))
                query_string = urlencode(query_pairs)
                return f'?{query_string}' if query_string else ''

            display_data['Actions'] = records['call_id'].apply(_build_view_url)
            column_config['Actions'] = st.column_config.LinkColumn(
                'Actions',
                help='Open call details',
                width='small',
                display_text='View'
            )

            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True,
                column_config=column_config,
                key='recent_calls_table'
            )
        else:
            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True,
                column_config=column_config
            )

        return selected_record
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """
        Format duration from seconds to human-readable format.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string
        """
        if pd.isna(seconds):
            return "-"
        
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        
        if minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"


class AgentPerformanceTable:
    """
    Specialized table for displaying agent performance metrics
    with rankings and visual indicators.
    """
    
    @classmethod
    def render(cls,
               data: pd.DataFrame,
               container: Any = None,
               metrics: List[str] = ['calls', 'connection_rate', 'avg_duration', 'revenue'],
               show_rankings: bool = True) -> None:
        """
        Render agent performance table with metrics and rankings.
        
        Args:
            data: DataFrame with call data
            container: Streamlit container to render in
            metrics: List of metrics to calculate and display
            show_rankings: Whether to show ranking column
        """
        container = container or st
        
        # Calculate agent metrics
        agent_stats = cls._calculate_agent_metrics(data, metrics)
        
        if agent_stats.empty:
            container.warning("No agent data available")
            return
        
        # Add rankings if requested
        if show_rankings:
            # Rank by total calls (or first metric)
            agent_stats = agent_stats.sort_values('Total Calls', ascending=False)
            agent_stats.insert(0, 'Rank', range(1, len(agent_stats) + 1))
        
        # Format columns
        for col in agent_stats.columns:
            if 'Rate' in col or 'Percentage' in col:
                agent_stats[col] = agent_stats[col].apply(lambda x: f"{x:.1f}%")
            elif 'Duration' in col:
                agent_stats[col] = agent_stats[col].apply(lambda x: f"{x:.1f} min")
            elif 'Revenue' in col:
                agent_stats[col] = agent_stats[col].apply(lambda x: f"${x:,.2f}")
            elif col not in ['Agent', 'Rank']:
                agent_stats[col] = agent_stats[col].apply(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
        
        # Display table
        st.dataframe(
            agent_stats,
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        # Add summary statistics
        with container.expander("Summary Statistics"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Agents", len(agent_stats))
            with col2:
                st.metric("Avg Calls/Agent", f"{data.groupby('agent_id').size().mean():.1f}")
            with col3:
                if 'revenue' in data.columns:
                    st.metric("Total Revenue", f"${data['revenue'].sum():,.2f}")
    
    @staticmethod
    def _calculate_agent_metrics(data: pd.DataFrame, 
                                 metrics: List[str]) -> pd.DataFrame:
        """
        Calculate performance metrics for each agent.
        
        Args:
            data: DataFrame with call data
            metrics: List of metrics to calculate
            
        Returns:
            DataFrame with agent metrics
        """
        if 'agent_id' not in data.columns:
            return pd.DataFrame()
        
        agent_groups = data.groupby('agent_id')
        agent_stats = pd.DataFrame()
        agent_stats['Agent'] = agent_groups.size().index
        
        # Calculate each requested metric
        if 'calls' in metrics:
            agent_stats['Total Calls'] = agent_groups.size().values
        
        if 'connection_rate' in metrics and 'outcome' in data.columns:
            agent_stats['Connection Rate'] = agent_groups['outcome'].agg(
                lambda outcomes: (outcomes == 'connected').mean() * 100
            ).values
        
        if 'avg_duration' in metrics and 'duration' in data.columns:
            agent_stats['Avg Duration'] = agent_groups['duration'].mean().values / 60
        
        if 'revenue' in metrics and 'revenue' in data.columns:
            agent_stats['Total Revenue'] = agent_groups['revenue'].sum().values
            agent_stats['Avg Revenue'] = agent_groups['revenue'].mean().values
        
        return agent_stats


class ComparisonTable:
    """
    Component for displaying comparison tables with highlighting
    of differences and trends between periods or groups.
    """
    
    @classmethod
    def render(cls,
               current_data: pd.DataFrame,
               previous_data: pd.DataFrame,
               group_column: str,
               metrics: List[str],
               container: Any = None,
               title: str = "Period Comparison") -> None:
        """
        Render a comparison table between two datasets.
        
        Args:
            current_data: Current period data
            previous_data: Previous period data
            group_column: Column to group by
            metrics: List of metrics to compare
            container: Streamlit container to render in
            title: Table title
        """
        container = container or st
        
        container.subheader(title)
        
        # Calculate metrics for both periods
        current_metrics = cls._calculate_group_metrics(current_data, group_column, metrics)
        previous_metrics = cls._calculate_group_metrics(previous_data, group_column, metrics)
        
        # Merge and calculate changes
        comparison = current_metrics.merge(
            previous_metrics,
            on=group_column,
            suffixes=('_Current', '_Previous'),
            how='outer'
        ).fillna(0)
        
        # Calculate percentage changes
        for metric in metrics:
            curr_col = f"{metric}_Current"
            prev_col = f"{metric}_Previous"
            
            if curr_col in comparison.columns and prev_col in comparison.columns:
                change_col = f"{metric}_Change"
                comparison[change_col] = (
                    (comparison[curr_col] - comparison[prev_col]) / 
                    comparison[prev_col].replace(0, 1) * 100
                )
                
                # Format change column with arrows
                comparison[change_col] = comparison[change_col].apply(
                    lambda x: f"{'↑' if x > 0 else '↓'} {abs(x):.1f}%" if not pd.isna(x) else "-"
                )
        
        # Display the comparison table
        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
            column_config={
                col: st.column_config.TextColumn(
                    col.replace('_', ' '),
                    help=f"Comparison of {col}"
                )
                for col in comparison.columns
            }
        )
    
    @staticmethod
    def _calculate_group_metrics(data: pd.DataFrame,
                                 group_column: str,
                                 metrics: List[str]) -> pd.DataFrame:
        """
        Calculate metrics for grouped data.
        
        Args:
            data: Input DataFrame
            group_column: Column to group by
            metrics: List of metrics to calculate
            
        Returns:
            DataFrame with calculated metrics
        """
        if group_column not in data.columns:
            return pd.DataFrame()
        
        result = pd.DataFrame()
        result[group_column] = data[group_column].unique()
        
        for group in result[group_column]:
            group_data = data[data[group_column] == group]
            
            for metric in metrics:
                if metric == 'count':
                    result.loc[result[group_column] == group, metric] = len(group_data)
                elif metric in data.columns:
                    result.loc[result[group_column] == group, metric] = group_data[metric].sum()
        
        return result
