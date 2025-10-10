"""
Filter Components Module for Call Analytics System

This module provides reusable Streamlit filter components for data
filtering and selection. Includes date ranges, multi-select filters,
and advanced search capabilities with state management.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd
import streamlit as st

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class FilterState:
    """
    Manages the state of all filters in the application.
    Provides serialization and restoration of filter selections.
    """

    date_range: tuple[date, date] = field(
        default_factory=lambda: (date.today() - timedelta(days=30), date.today())
    )
    selected_agents: list[str] = field(default_factory=list)
    selected_campaigns: list[str] = field(default_factory=list)
    selected_outcomes: list[str] = field(default_factory=list)
    selected_types: list[str] = field(default_factory=list)
    duration_range: tuple[float, float] = field(default=(0.0, float("inf")))
    revenue_range: tuple[float, float] = field(default=(0.0, float("inf")))
    search_query: str = ""

    def to_dict(self) -> dict[str, Any]:
        """
        Convert filter state to dictionary for serialization.

        Returns:
            Dictionary representation of filter state
        """
        return {
            "date_range": [self.date_range[0].isoformat(), self.date_range[1].isoformat()],
            "selected_agents": self.selected_agents,
            "selected_campaigns": self.selected_campaigns,
            "selected_outcomes": self.selected_outcomes,
            "selected_types": self.selected_types,
            "duration_range": list(self.duration_range),
            "revenue_range": list(self.revenue_range),
            "search_query": self.search_query,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FilterState":
        """
        Create FilterState from dictionary.

        Args:
            data: Dictionary with filter state data

        Returns:
            FilterState instance
        """
        return cls(
            date_range=(
                date.fromisoformat(data["date_range"][0]),
                date.fromisoformat(data["date_range"][1]),
            ),
            selected_agents=data.get("selected_agents", []),
            selected_campaigns=data.get("selected_campaigns", []),
            selected_outcomes=data.get("selected_outcomes", []),
            selected_types=data.get("selected_types", []),
            duration_range=tuple(data.get("duration_range", [0.0, float("inf")])),
            revenue_range=tuple(data.get("revenue_range", [0.0, float("inf")])),
            search_query=data.get("search_query", ""),
        )

    def apply_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all filters to a DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()

        # Apply date range filter
        if "timestamp" in filtered_df.columns:
            filtered_df["timestamp"] = pd.to_datetime(filtered_df["timestamp"])
            start_datetime = pd.Timestamp(self.date_range[0])
            end_datetime = pd.Timestamp(self.date_range[1]) + pd.Timedelta(days=1)
            filtered_df = filtered_df[
                (filtered_df["timestamp"] >= start_datetime)
                & (filtered_df["timestamp"] < end_datetime)
            ]

        # Apply agent filter
        if self.selected_agents and "agent_id" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["agent_id"].isin(self.selected_agents)]

        # Apply campaign filter
        if self.selected_campaigns and "campaign" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["campaign"].isin(self.selected_campaigns)]

        # Apply outcome filter
        if self.selected_outcomes and "outcome" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["outcome"].isin(self.selected_outcomes)]

        # Apply type filter
        if self.selected_types and "call_type" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["call_type"].isin(self.selected_types)]

        # Apply duration range filter
        if "duration" in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df["duration"] >= self.duration_range[0])
                & (filtered_df["duration"] <= self.duration_range[1])
            ]

        # Apply revenue range filter
        if "revenue" in filtered_df.columns:
            filtered_df = filtered_df[
                (filtered_df["revenue"] >= self.revenue_range[0])
                & (filtered_df["revenue"] <= self.revenue_range[1])
            ]

        # Apply search query to notes/transcript
        if self.search_query:
            search_cols = ["notes", "transcript"]
            mask = pd.Series([False] * len(filtered_df))
            for col in search_cols:
                if col in filtered_df.columns:
                    mask |= filtered_df[col].str.contains(self.search_query, case=False, na=False)
            filtered_df = filtered_df[mask]

        return filtered_df


class DateRangeFilter:
    """
    Component for selecting date ranges with preset options
    and custom range selection.
    """

    PRESET_RANGES = {
        "Today": lambda: (date.today(), date.today()),
        "Yesterday": lambda: (date.today() - timedelta(days=1), date.today() - timedelta(days=1)),
        "Last 7 Days": lambda: (date.today() - timedelta(days=6), date.today()),
        "Last 30 Days": lambda: (date.today() - timedelta(days=29), date.today()),
        "Last 90 Days": lambda: (date.today() - timedelta(days=89), date.today()),
        "This Month": lambda: (date.today().replace(day=1), date.today()),
        "Last Month": lambda: (
            (date.today().replace(day=1) - timedelta(days=1)).replace(day=1),
            date.today().replace(day=1) - timedelta(days=1),
        ),
        "This Year": lambda: (date(date.today().year, 1, 1), date.today()),
        "Custom": None,
    }

    @staticmethod
    def _resolve_preset_range(preset: str) -> tuple[date, date]:
        resolver = DateRangeFilter.PRESET_RANGES.get(preset)
        if callable(resolver):
            start_date, end_date = resolver()
        else:
            today = date.today()
            start_date, end_date = today, today

        if start_date > end_date:
            start_date, end_date = end_date, start_date

        return start_date, end_date

    @staticmethod
    def _normalize_selection(
        selection: Any,
        current_range: tuple[date, date] | None = None,
    ) -> tuple[date, date] | None:
        if selection is None:
            return None

        if isinstance(selection, (tuple, list)):
            cleaned = tuple(d for d in selection if d is not None)
        else:
            cleaned = (selection,) if selection else ()

        if not cleaned:
            return None

        if len(cleaned) == 1:
            selected = cleaned[0]

            base_start, base_end = None, None
            if isinstance(current_range, tuple) and len(current_range) == 2:
                base_start, base_end = current_range

            if base_start is None and base_end is None:
                start_date = end_date = selected
            elif base_start is not None and base_end is not None and base_start == base_end:
                if selected >= base_start:
                    start_date, end_date = base_start, selected
                else:
                    start_date, end_date = selected, base_start
            else:
                # Determine which boundary the user is adjusting based on proximity
                if base_start is None:
                    base_start = selected
                if base_end is None:
                    base_end = selected

                adjust_start = abs((selected - base_start).days) <= abs((selected - base_end).days)
                if adjust_start:
                    start_date, end_date = selected, base_end
                else:
                    start_date, end_date = base_start, selected
        else:
            start_date, end_date = cleaned[:2]

        if start_date > end_date:
            start_date, end_date = end_date, start_date

        return start_date, end_date

    @classmethod
    def render(
        cls,
        container: Any = None,
        key_prefix: str = "date_filter",
        default_range: str = "Last 30 Days",
    ) -> tuple[date, date]:
        """
        Render date range filter component.

        Args:
            container: Streamlit container to render in
            key_prefix: Prefix for component keys
            default_range: Default preset range selection

        Returns:
            Tuple of (start_date, end_date)
        """
        container = container or st

        preset_key = f"{key_prefix}_preset"
        picker_key = f"{key_prefix}_date_picker"
        state_key = f"{key_prefix}_date_value"

        preset_options = list(cls.PRESET_RANGES.keys())
        default_index = (
            preset_options.index(default_range) if default_range in preset_options else 0
        )

        if state_key not in st.session_state:
            st.session_state[state_key] = cls._resolve_preset_range(preset_options[default_index])

        preset = container.selectbox(
            "Date Range", options=preset_options, index=default_index, key=preset_key
        )

        if preset == "Custom":
            default_value = cls._resolve_preset_range(preset_options[default_index])
            current_value = st.session_state.get(state_key, default_value)

            if not isinstance(current_value, tuple) or len(current_value) != 2:
                current_value = default_value

            date_selection = container.date_input(
                "Select date range",
                value=current_value,
                key=picker_key,
                label_visibility="collapsed",
                disabled=False,
            )

            normalized = cls._normalize_selection(date_selection, current_value)

            if normalized is None:
                start_date, end_date = current_value
            else:
                start_date, end_date = normalized
                st.session_state[state_key] = (start_date, end_date)
        else:
            start_date, end_date = cls._resolve_preset_range(preset)
            st.session_state[state_key] = (start_date, end_date)
            container.date_input(
                "Select date range",
                value=(start_date, end_date),
                key=picker_key,
                label_visibility="collapsed",
                disabled=True,
            )

        container.caption(f"📅 {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")

        return start_date, end_date


class MultiSelectFilter:
    """
    Component for multi-select filtering with search and
    select all/none functionality.
    """

    @classmethod
    def render(
        cls,
        label: str,
        options: list[str],
        default: list[str] | None = None,
        container: Any = None,
        key: str = "multiselect",
        help_text: str | None = None,
        max_selections: int | None = None,
    ) -> list[str]:
        """
        Render multi-select filter component with enhanced features.

        Args:
            label: Label for the filter
            options: List of available options
            default: Default selected options
            container: Streamlit container to render in
            key: Component key
            help_text: Optional help text
            max_selections: Maximum number of selections allowed

        Returns:
            List of selected options
        """
        container = container or st

        # Add select all/none buttons
        col1, col2, col3 = container.columns([2, 1, 1])

        with col2:
            if st.button("Select All", key=f"{key}_all", use_container_width=True):
                st.session_state[f"{key}_selected"] = (
                    options[:max_selections] if max_selections else options
                )

        with col3:
            if st.button("Clear", key=f"{key}_clear", use_container_width=True):
                st.session_state[f"{key}_selected"] = []

        # Get current selection from session state
        if f"{key}_selected" not in st.session_state:
            st.session_state[f"{key}_selected"] = default or []

        # Render multiselect
        with col1:
            selected = st.multiselect(
                label,
                options=options,
                default=st.session_state[f"{key}_selected"],
                key=f"{key}_widget",
                help=help_text,
            )

        # Update session state
        st.session_state[f"{key}_selected"] = selected

        # Show selection count
        if selected:
            container.caption(f"Selected: {len(selected)} of {len(options)}")

        return selected


class RangeSliderFilter:
    """
    Component for numeric range filtering with min/max sliders.
    """

    @classmethod
    def render(
        cls,
        label: str,
        min_value: float,
        max_value: float,
        default_range: tuple[float, float] | None = None,
        container: Any = None,
        key: str = "range_slider",
        step: float = 1.0,
        format_func: callable | None = None,
    ) -> tuple[float, float]:
        """
        Render range slider filter component.

        Args:
            label: Label for the filter
            min_value: Minimum possible value
            max_value: Maximum possible value
            default_range: Default selected range
            container: Streamlit container to render in
            key: Component key
            step: Slider step size
            format_func: Optional function to format display values

        Returns:
            Tuple of (min_selected, max_selected)
        """
        container = container or st

        # Set default range
        if default_range is None:
            default_range = (min_value, max_value)

        # Create slider
        selected_range = container.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            value=default_range,
            step=step,
            key=key,
        )

        # Display formatted range
        if format_func:
            display_min = format_func(selected_range[0])
            display_max = format_func(selected_range[1])
            container.caption(f"Range: {display_min} - {display_max}")

        return selected_range


class SearchFilter:
    """
    Component for text-based search filtering with
    advanced search operators and suggestions.
    """

    @classmethod
    def render(
        cls,
        container: Any = None,
        key: str = "search",
        placeholder: str = "Search calls...",
        suggestions: list[str] | None = None,
        help_text: str = "Search in notes and transcripts",
    ) -> str:
        """
        Render search filter component with autocomplete.

        Args:
            container: Streamlit container to render in
            key: Component key
            placeholder: Placeholder text
            suggestions: Optional list of search suggestions
            help_text: Help text for the search field

        Returns:
            Search query string
        """
        container = container or st

        # Search input
        search_query = container.text_input(
            "Search", placeholder=placeholder, key=key, help=help_text
        )

        # Show search operators help
        with container.expander("Search Tips"):
            st.markdown(
                """
            **Search Operators:**
            - Use quotes for exact phrases: `"customer complaint"`
            - Use AND/OR for multiple terms: `billing AND payment`
            - Use - to exclude terms: `support -technical`
            - Use wildcards: `call*` matches 'calls', 'calling', etc.
            """
            )

        # Show suggestions if query is being typed
        if suggestions and search_query and len(search_query) >= 2:
            matching_suggestions = [s for s in suggestions if search_query.lower() in s.lower()][:5]

            if matching_suggestions:
                st.caption("Suggestions:")
                for suggestion in matching_suggestions:
                    if st.button(suggestion, key=f"{key}_sug_{suggestion}"):
                        st.session_state[key] = suggestion
                        st.experimental_rerun()

        return search_query


class QuickFilters:
    """
    Component for rendering a set of quick filter buttons
    for common filtering scenarios.
    """

    @classmethod
    def render(
        cls, data: pd.DataFrame, container: Any = None, key_prefix: str = "quick"
    ) -> dict[str, bool]:
        """
        Render quick filter buttons.

        Args:
            data: DataFrame to derive quick filter options from
            container: Streamlit container to render in
            key_prefix: Prefix for component keys

        Returns:
            Dictionary of filter states
        """
        container = container or st

        # Define quick filters
        quick_filters = {
            "✅ Connected Calls": lambda df: df[df["outcome"] == "connected"],
            "❌ Failed Calls": lambda df: df[df["outcome"].isin(["failed", "no_answer"])],
            "⏱️ Long Calls (>5 min)": lambda df: df[df["duration"] > 300],
            "💰 Revenue Calls": lambda df: df[df["revenue"] > 0],
            "📅 Today's Calls": lambda df: df[
                pd.to_datetime(df["timestamp"]).dt.date == date.today()
            ],
        }

        # Render filter buttons
        container.write("**Quick Filters:**")
        cols = container.columns(len(quick_filters))

        filter_states = {}
        for idx, (label, _filter_func) in enumerate(quick_filters.items()):
            with cols[idx]:
                filter_states[label] = st.checkbox(
                    label, key=f"{key_prefix}_{label}", help=f"Apply {label} filter"
                )

        return filter_states


def save_filter_preset(name: str, filter_state: FilterState) -> None:
    """
    Save current filter configuration as a preset.

    Args:
        name: Name for the preset
        filter_state: Current filter state to save
    """
    try:
        # Load existing presets
        presets = st.session_state.get("filter_presets", {})

        # Add new preset
        presets[name] = filter_state.to_dict()

        # Save to session state
        st.session_state["filter_presets"] = presets

        logger.info(f"Saved filter preset: {name}")
        st.success(f"Filter preset '{name}' saved successfully!")

    except Exception as e:
        logger.error(f"Error saving filter preset: {e}")
        st.error(f"Failed to save preset: {str(e)}")


def load_filter_preset(name: str) -> FilterState | None:
    """
    Load a saved filter preset.

    Args:
        name: Name of the preset to load

    Returns:
        FilterState if preset exists, None otherwise
    """
    try:
        presets = st.session_state.get("filter_presets", {})

        if name in presets:
            filter_state = FilterState.from_dict(presets[name])
            logger.info(f"Loaded filter preset: {name}")
            return filter_state
        else:
            logger.warning(f"Filter preset not found: {name}")
            return None

    except Exception as e:
        logger.error(f"Error loading filter preset: {e}")
        return None
