"""
Table Display Components for Call Analytics System

This module provides reusable Streamlit table components for displaying
call records, analytics results, and agent performance data with sorting,
filtering, pagination, and export capabilities.
"""

import base64
import logging
from datetime import datetime
from io import BytesIO
from typing import Any, NamedTuple
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import streamlit as st

# Configure module logger
logger = logging.getLogger(__name__)


class PaginationContext(NamedTuple):
    page_data: pd.DataFrame
    total_rows: int
    total_pages: int
    current_page: int
    start_idx: int
    end_idx: int


class DataTable:
    """
    Enhanced data table component with sorting, filtering, and
    pagination capabilities for large datasets.
    """

    @classmethod
    def render(
        cls,
        data: pd.DataFrame,
        container: Any = None,
        page_size: int = 20,
        key: str = "data_table",
        show_index: bool = False,
        column_config: dict[str, Any] | None = None,
        height: int | None = None,
        enable_search: bool = True,
        enable_export: bool = True,
    ) -> pd.DataFrame:
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
        cls._ensure_pagination_state(key)
        filtered_data = cls._apply_search(container, data.copy(), enable_search, key)
        pagination = cls._paginate(filtered_data, page_size, key)

        col1, col2, col3 = container.columns([2, 3, 2])
        cls._render_table_summary(
            col1,
            pagination.start_idx,
            pagination.end_idx,
            pagination.total_rows,
        )
        cls._render_pagination_controls(col2, key, pagination.current_page, pagination.total_pages)
        cls._render_export_controls(col3, enable_export, filtered_data, key)

        st.dataframe(
            pagination.page_data,
            width="stretch",
            hide_index=not show_index,
            column_config=column_config,
            height=height,
        )

        return filtered_data

    @staticmethod
    def _ensure_pagination_state(key: str) -> None:
        if f"{key}_page" not in st.session_state:
            st.session_state[f"{key}_page"] = 0

    @staticmethod
    def _apply_search(
        container: Any, data: pd.DataFrame, enable_search: bool, key: str
    ) -> pd.DataFrame:
        if not enable_search:
            return data

        search_query = container.text_input(
            "🔍 Search table", key=f"{key}_search", placeholder="Type to search..."
        )
        if not search_query:
            return data

        mask = pd.Series([False] * len(data))
        for col in data.select_dtypes(include=["object"]).columns:
            mask |= data[col].astype(str).str.contains(search_query, case=False, na=False)

        return data[mask]

    @staticmethod
    def _paginate(data: pd.DataFrame, page_size: int, key: str) -> "PaginationContext":
        total_rows = len(data)
        total_pages = max(1, (total_rows - 1) // page_size + 1) if page_size else 1
        current_page = min(st.session_state.get(f"{key}_page", 0), total_pages - 1)
        st.session_state[f"{key}_page"] = current_page

        start_idx = current_page * page_size
        end_idx = min(start_idx + page_size, total_rows)
        page_data = data.iloc[start_idx:end_idx]

        return PaginationContext(
            page_data=page_data,
            total_rows=total_rows,
            total_pages=total_pages,
            current_page=current_page,
            start_idx=start_idx,
            end_idx=end_idx,
        )

    @staticmethod
    def _render_table_summary(col: Any, start_idx: int, end_idx: int, total_rows: int) -> None:
        with col:
            first_row = start_idx + 1 if total_rows else 0
            st.caption(f"Showing {first_row}-{end_idx} of {total_rows} rows")

    @classmethod
    def _render_pagination_controls(
        cls, col: Any, key: str, current_page: int, total_pages: int
    ) -> None:
        with col:
            buttons = st.columns(5)
            cls._pagination_button(buttons[0], "⏮️", f"{key}_first", key, 0, current_page == 0)
            cls._pagination_button(
                buttons[1], "◀️", f"{key}_prev", key, current_page - 1, current_page == 0
            )
            with buttons[2]:
                st.caption(f"Page {current_page + 1}/{total_pages}")
            cls._pagination_button(
                buttons[3],
                "▶️",
                f"{key}_next",
                key,
                current_page + 1,
                current_page >= total_pages - 1,
            )
            cls._pagination_button(
                buttons[4],
                "⏭️",
                f"{key}_last",
                key,
                total_pages - 1,
                current_page >= total_pages - 1,
            )

    @staticmethod
    def _render_export_controls(
        col: Any, enable_export: bool, data: pd.DataFrame, key: str
    ) -> None:
        if not enable_export:
            return

        with col:
            export_format = st.selectbox(
                "Export", ["", "CSV", "Excel", "JSON"], key=f"{key}_export_format"
            )
            if export_format:
                DataTable._handle_export(data, export_format, key)

    @staticmethod
    def _pagination_button(
        col: Any,
        label: str,
        state_key: str,
        base_key: str,
        target_page: int,
        disabled: bool,
    ) -> None:
        with col:
            if st.button(label, key=state_key, disabled=disabled):
                st.session_state[f"{base_key}_page"] = target_page
                st.rerun()

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
            href = (
                '<a href="data:file/csv;base64,'
                f'{b64}" download="export_{timestamp}.csv">Download CSV</a>'
            )
            st.markdown(href, unsafe_allow_html=True)

        elif format == "Excel":
            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                data.to_excel(writer, index=False, sheet_name="Data")
            excel_data = output.getvalue()
            b64 = base64.b64encode(excel_data).decode()
            href = (
                '<a href="data:application/vnd.openxmlformats-officedocument.'
                "spreadsheetml.sheet;base64,"
                f'{b64}" download="export_{timestamp}.xlsx">Download Excel</a>'
            )
            st.markdown(href, unsafe_allow_html=True)

        elif format == "JSON":
            json_str = data.to_json(orient="records", indent=2)
            b64 = base64.b64encode(json_str.encode()).decode()
            href = (
                '<a href="data:file/json;base64,'
                f'{b64}" download="export_{timestamp}.json">Download JSON</a>'
            )
            st.markdown(href, unsafe_allow_html=True)


class CallRecordsTable:
    """
    Specialized table component for displaying call records with
    custom formatting and action buttons.
    """

    @classmethod
    def render(
        cls,
        records: pd.DataFrame,
        container: Any = None,
        show_actions: bool = True,
        show_transcript: bool = False,
    ) -> dict[str, Any] | None:
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
        record_session_key = "recent_calls_selected_record"

        records = cls._prepare_records(records, record_session_key)
        if records is None:
            return None

        display_data, column_config = cls._build_display_table(records, show_transcript)
        selected_record = cls._restore_selection(records, record_session_key)

        if show_actions and "call_id" in records.columns:
            display_data, column_config = cls._attach_actions_column(
                records, display_data, column_config
            )
            cls._render_dataframe(
                display_data, column_config, container, table_key="recent_calls_table"
            )
        else:
            cls._render_dataframe(display_data, column_config, container)

        return selected_record

    @staticmethod
    def _prepare_records(records: pd.DataFrame | None, session_key: str) -> pd.DataFrame | None:
        if records is None or records.empty:
            if "view_call" in st.query_params:
                st.query_params.pop("view_call", None)
            st.session_state.pop(session_key, None)
            return None
        return records.reset_index(drop=True)

    @classmethod
    def _build_display_table(
        cls, records: pd.DataFrame, show_transcript: bool
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        display_columns = [
            "call_id",
            "phone_number",
            "timestamp",
            "duration",
            "outcome",
            "agent_id",
            "campaign",
        ]

        if show_transcript and "transcript" in records.columns:
            display_columns.append("transcript")

        if "revenue" in records.columns:
            display_columns.append("revenue")

        available_columns = [col for col in display_columns if col in records.columns]
        display_data = records[available_columns].copy()

        if "timestamp" in display_data.columns:
            display_data["timestamp"] = pd.to_datetime(display_data["timestamp"]).dt.strftime(
                "%Y-%m-%d %H:%M"
            )

        if "duration" in display_data.columns:
            display_data["duration"] = display_data["duration"].apply(cls._format_duration)

        if "revenue" in display_data.columns:
            display_data["revenue"] = display_data["revenue"].apply(
                lambda x: f"${x:.2f}" if x > 0 else "-"
            )

        if "transcript" in display_data.columns:
            display_data["transcript"] = display_data["transcript"].str[:100] + "..."

        base_config: dict[str, Any] = {
            "call_id": st.column_config.TextColumn("Call ID", width="small"),
            "phone_number": st.column_config.TextColumn("Phone", width="medium"),
            "timestamp": st.column_config.TextColumn("Date/Time", width="medium"),
            "duration": st.column_config.TextColumn("Duration", width="small"),
            "outcome": st.column_config.TextColumn("Outcome", width="small"),
            "agent_id": st.column_config.TextColumn("Agent", width="small"),
            "campaign": st.column_config.TextColumn("Campaign", width="medium"),
            "revenue": st.column_config.TextColumn("Revenue", width="small"),
            "transcript": st.column_config.TextColumn("Transcript Preview", width="large"),
        }

        column_config = {
            key: value for key, value in base_config.items() if key in display_data.columns
        }

        return display_data, column_config

    @staticmethod
    def _first_value(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, list):
            return value[0] if value else None
        return str(value)

    @classmethod
    def _restore_selection(cls, records: pd.DataFrame, session_key: str) -> dict[str, Any] | None:
        selected = cls._selection_from_query(records, session_key)
        if selected is not None:
            return selected
        return cls._selection_from_session(records, session_key)

    @classmethod
    def _selection_from_query(
        cls, records: pd.DataFrame, session_key: str
    ) -> dict[str, Any] | None:
        view_call_param = cls._first_value(st.query_params.get("view_call"))
        if not view_call_param or "call_id" not in records.columns:
            return None

        matches = records.index[records["call_id"] == view_call_param].tolist()
        if not matches:
            st.query_params.pop("view_call", None)
            st.session_state.pop(session_key, None)
            return None

        selected_record = records.iloc[matches[0]].to_dict()
        cls._sync_selection(selected_record, session_key)
        return selected_record

    @classmethod
    def _selection_from_session(
        cls, records: pd.DataFrame, session_key: str
    ) -> dict[str, Any] | None:
        cached_record = st.session_state.get(session_key)
        call_ids = records["call_id"].values if "call_id" in records.columns else np.array([])
        if cached_record and cached_record.get("call_id") in call_ids:
            cls._sync_selection(cached_record, session_key)
            return cached_record

        st.session_state.pop(session_key, None)
        return None

    @classmethod
    def _sync_selection(cls, record: dict[str, Any], session_key: str) -> None:
        st.session_state[session_key] = record
        call_id_value = record.get("call_id")
        if not call_id_value:
            return

        current_param = cls._first_value(st.query_params.get("view_call"))
        if current_param != call_id_value:
            st.query_params["view_call"] = call_id_value

    @classmethod
    def _attach_actions_column(
        cls,
        records: pd.DataFrame,
        display_data: pd.DataFrame,
        column_config: dict[str, Any],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        display_with_actions = display_data.copy()
        display_with_actions["Actions"] = records["call_id"].apply(cls._build_view_url)
        column_config["Actions"] = st.column_config.LinkColumn(
            "Actions", help="Open call details", width="small", display_text="View"
        )
        return display_with_actions, column_config

    @staticmethod
    def _build_view_url(call_id: Any) -> str:
        call_id_str = str(call_id) if call_id is not None else ""
        if not call_id_str:
            return ""

        query_pairs: list[tuple[str, str]] = []
        for key in st.query_params:
            if key == "view_call":
                continue
            value = st.query_params.get(key)
            if isinstance(value, list):
                query_pairs.extend((key, str(item)) for item in value if item is not None)
            elif value is not None:
                query_pairs.append((key, str(value)))

        query_pairs.append(("view_call", call_id_str))
        query_string = urlencode(query_pairs)
        return f"?{query_string}" if query_string else ""

    @staticmethod
    def _render_dataframe(
        data: pd.DataFrame,
        column_config: dict[str, Any],
        container: Any,
        table_key: str | None = None,
    ) -> None:
        render_kwargs = {
            "width": "stretch",
            "hide_index": True,
            "column_config": column_config,
        }
        if table_key:
            render_kwargs["key"] = table_key

        container.dataframe(data, **render_kwargs)

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
    def render(
        cls,
        data: pd.DataFrame,
        container: Any = None,
        metrics: list[str] | None = None,
        show_rankings: bool = True,
    ) -> None:
        """
        Render agent performance table with metrics and rankings.

        Args:
            data: DataFrame with call data
            container: Streamlit container to render in
            metrics: List of metrics to calculate and display
            show_rankings: Whether to show ranking column
        """
        container = container or st
        metrics = metrics or ["calls", "connection_rate", "avg_duration", "revenue"]

        # Calculate agent metrics
        agent_stats = cls._calculate_agent_metrics(data, metrics)

        if agent_stats.empty:
            container.warning("No agent data available")
            return

        # Add rankings if requested
        if show_rankings:
            # Rank by total calls (or first metric)
            agent_stats = agent_stats.sort_values("Total Calls", ascending=False)
            agent_stats.insert(0, "Rank", range(1, len(agent_stats) + 1))

        # Format columns
        for col in agent_stats.columns:
            if "Rate" in col or "Percentage" in col:
                agent_stats[col] = agent_stats[col].apply(lambda x: f"{x:.1f}%")
            elif "Duration" in col:
                agent_stats[col] = agent_stats[col].apply(lambda x: f"{x:.1f} min")
            elif "Revenue" in col:
                agent_stats[col] = agent_stats[col].apply(lambda x: f"${x:,.2f}")
            elif col not in ["Agent", "Rank"]:
                agent_stats[col] = agent_stats[col].apply(
                    lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x
                )

        # Display table
        st.dataframe(agent_stats, width="stretch", hide_index=True, height=400)

        # Add summary statistics
        with container.expander("Summary Statistics"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Agents", len(agent_stats))
            with col2:
                st.metric("Avg Calls/Agent", f"{data.groupby('agent_id').size().mean():.1f}")
            with col3:
                if "revenue" in data.columns:
                    st.metric("Total Revenue", f"${data['revenue'].sum():,.2f}")

    @staticmethod
    def _calculate_agent_metrics(data: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
        """
        Calculate performance metrics for each agent.

        Args:
            data: DataFrame with call data
            metrics: List of metrics to calculate

        Returns:
            DataFrame with agent metrics
        """
        if "agent_id" not in data.columns:
            return pd.DataFrame()

        agent_groups = data.groupby("agent_id")
        agent_stats = pd.DataFrame()
        agent_stats["Agent"] = agent_groups.size().index

        # Calculate each requested metric
        if "calls" in metrics:
            agent_stats["Total Calls"] = agent_groups.size().values

        if "connection_rate" in metrics and "outcome" in data.columns:
            agent_stats["Connection Rate"] = (
                agent_groups["outcome"]
                .agg(lambda outcomes: (outcomes == "connected").mean() * 100)
                .values
            )

        if "avg_duration" in metrics and "duration" in data.columns:
            agent_stats["Avg Duration"] = agent_groups["duration"].mean().values / 60

        if "revenue" in metrics and "revenue" in data.columns:
            agent_stats["Total Revenue"] = agent_groups["revenue"].sum().values
            agent_stats["Avg Revenue"] = agent_groups["revenue"].mean().values

        return agent_stats


class ComparisonTable:
    """
    Component for displaying comparison tables with highlighting
    of differences and trends between periods or groups.
    """

    @classmethod
    def render(
        cls,
        current_data: pd.DataFrame,
        previous_data: pd.DataFrame,
        group_column: str,
        metrics: list[str],
        container: Any = None,
        title: str = "Period Comparison",
    ) -> None:
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

        comparison = cls.prepare_comparison_data(
            current_data=current_data,
            previous_data=previous_data,
            group_column=group_column,
            metrics=metrics,
        )

        if comparison.empty:
            container.warning("No comparison data available")
            return

        container.subheader(title)

        display_df = cls._build_display_frame(comparison, metrics)

        container.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
            column_config={
                col: st.column_config.TextColumn(col.replace("_", " "), help=f"Comparison of {col}")
                for col in display_df.columns
            },
        )

    @classmethod
    def _build_display_frame(cls, comparison: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
        display_df = comparison.copy()

        for metric in metrics:
            change_col = f"{metric}_Change"
            delta_col = f"{metric}_Delta"

            if change_col in display_df.columns:
                display_df[change_col] = display_df.apply(
                    lambda row, c=change_col, d=delta_col: cls._format_change_value(row, c, d),
                    axis=1,
                )

            if delta_col in display_df.columns:
                display_df.drop(columns=[delta_col], inplace=True)

        return display_df

    @staticmethod
    def _format_change_value(row: pd.Series, change_column: str, delta_column: str) -> str:
        value = row[change_column]
        delta_value = row.get(delta_column)

        if pd.isna(value):
            return "-"

        if value > 0:
            arrow = "↑"
        elif value < 0:
            arrow = "↓"
        else:
            arrow = "→"

        pct_text = f"{abs(value):.1f}%"
        delta_text = ComparisonTable._format_delta_text(delta_value)

        return f"{arrow} {pct_text}{delta_text}"

    @staticmethod
    def _format_delta_text(delta_value: Any) -> str:
        if delta_value is None or pd.isna(delta_value) or delta_value == 0:
            return ""

        if float(delta_value).is_integer():
            formatted_delta = f"{delta_value:+,.0f}"
        else:
            formatted_delta = f"{delta_value:+,.2f}"

        return f" (Δ {formatted_delta})"

    @staticmethod
    def prepare_comparison_data(
        current_data: pd.DataFrame,
        previous_data: pd.DataFrame,
        group_column: str,
        metrics: list[str],
    ) -> pd.DataFrame:
        """Compute comparison metrics between periods for reuse in tables and charts."""
        if current_data is None or previous_data is None:
            return pd.DataFrame()

        if group_column not in current_data.columns or group_column not in previous_data.columns:
            return pd.DataFrame()

        current_metrics = ComparisonTable._calculate_group_metrics(
            current_data, group_column, metrics
        )
        previous_metrics = ComparisonTable._calculate_group_metrics(
            previous_data, group_column, metrics
        )

        if current_metrics.empty and previous_metrics.empty:
            return pd.DataFrame()

        comparison = current_metrics.merge(
            previous_metrics, on=group_column, suffixes=("_Current", "_Previous"), how="outer"
        ).fillna(0)

        for metric in metrics:
            curr_col = f"{metric}_Current"
            prev_col = f"{metric}_Previous"

            if curr_col in comparison.columns and prev_col in comparison.columns:
                change_col = f"{metric}_Change"
                delta_col = f"{metric}_Delta"

                delta_values = comparison[curr_col] - comparison[prev_col]
                comparison[delta_col] = delta_values

                prev_values = comparison[prev_col]
                max_reference = max(
                    comparison[curr_col].abs().max(), comparison[prev_col].abs().max(), 1e-9
                )

                with np.errstate(divide="ignore", invalid="ignore"):
                    zero_prev = prev_values == 0
                    zero_delta = delta_values == 0

                    base_change = np.where(
                        zero_prev,
                        np.where(
                            zero_delta,
                            0.0,
                            np.sign(delta_values)
                            * np.minimum(
                                100.0,
                                (np.abs(delta_values) / max_reference) * 100.0,
                            ),
                        ),
                        (delta_values / np.abs(prev_values)) * 100.0,
                    )

                    raw_change = np.where(
                        zero_prev, base_change, np.clip(base_change, -100.0, 100.0)
                    )

                comparison[change_col] = raw_change

        return comparison

    @staticmethod
    def _calculate_group_metrics(
        data: pd.DataFrame, group_column: str, metrics: list[str]
    ) -> pd.DataFrame:
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
                if metric == "count":
                    result.loc[result[group_column] == group, metric] = len(group_data)
                elif metric in data.columns:
                    result.loc[result[group_column] == group, metric] = group_data[metric].sum()

        return result
