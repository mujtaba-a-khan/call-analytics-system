# Functional Programming

This document outlines how the system applies functional programming principles, including final data structures, higher-order functions, functions as parameters or return values, and closures/anonymous functions.

## Table of Contents
- [Functional Programming Principles](#functional-programming-principles)
  - [Only Final Data Structures](#only-final-data-structures)
  - [Functions](#functions)
  - [Higher-Order Functions](#higher-order-functions)
  - [Functions as Parameters and Return Values](#functions-as-parameters-and-return-values)
  - [Closures / Anonymous Functions](#closures--anonymous-functions)
- [Core Functions](#core-functions)
  - [1. `ChartTheme.get_layout_template`](#1-chartthemeget_layout_template)
  - [2. `render_chart_in_streamlit`](#2-render_chart_in_streamlit)
  - [3. `FilterState.apply_to_dataframe`](#3-filterstateapply_to_dataframe)
  - [4. `EmbeddingManager.generate_embeddings`](#4-embeddingmanagergenerate_embeddings)
  - [5. `QuickFilters.render`](#5-quickfiltersrender)
- [Conclusion](#conclusion)

## Functional Programming Principles

### Only Final Data Structures
- The codebase relies on dataclasses such as `FilterState` and `TranscriptionResult` to manage state cleanly. Filters copy DataFrames before applying transformations, but some visualization helpers (for example `TimeSeriesChart.create_peak_hours_heatmap`) add helper columns in place, so immutability is not enforced everywhere.
- Example: `FilterState.apply_to_dataframe` returns a new filtered DataFrame without modifying the input.

### Functions
- Core logic functions are designed, producing outputs solely based on inputs. Rendering functions introduce controlled side effects for UI updates.
- Example: `ChartTheme.get_layout_template` returns a configuration dictionary without external state changes.

### Higher-Order Functions
- Functions accept other functions as arguments, enabling modular and reusable code.
- Example: `render_chart_in_streamlit` takes a chart creation function as a parameter.

### Functions as Parameters and Return Values
- Functions are passed as parameters, and the design supports potential function return values in extensible components.
- Example: `DateRangeFilter.PRESET_RANGES` uses lambda functions as values.

### Closures / Anonymous Functions
- Lambda functions and closures encapsulate logic, particularly in filtering and preset configurations.
- Example: `QuickFilters.render` uses lambda functions to define filter logic dynamically.

## Core Functions

### 1. `ChartTheme.get_layout_template`

*From: `src/ui/components/charts.py`, lines 58-75*

**Description**: Returns a dictionary with standardized Plotly chart layout configurations, ensuring consistent styling across visualizations.

**Functional Aspects**:

- **Function**: Returns a new dictionary without modifying external state.
- **Final Data Structure**: Provides a fresh dictionary for each call; charts typically update this copy to set titles and axes without touching shared state.
- **Reusability**: Used across multiple chart types, demonstrating functional composition.

```python
@classmethod
def get_layout_template(cls) -> dict[str, Any]:
    """
    Get standard layout template for Plotly charts.

    Returns:
        Dictionary with layout configuration
    """
    return {
        "template": "plotly_dark",
        "paper_bgcolor": cls.COLORS["background"],
        "plot_bgcolor": cls.COLORS["paper"],
        "font": {"color": cls.COLORS["text"], "size": 12},
        "margin": {"l": 50, "r": 30, "t": 40, "b": 50},
        "hoverlabel": {"bgcolor": cls.COLORS["paper"], "font_size": 12, "font_family": "Arial"},
        "xaxis": {"gridcolor": cls.COLORS["grid"], "zerolinecolor": cls.COLORS["grid"]},
        "yaxis": {"gridcolor": cls.COLORS["grid"], "zerolinecolor": cls.COLORS["grid"]},
    }
```

### 2. `render_chart_in_streamlit`

*From: `src/ui/components/charts.py`, lines 648-662*

**Description**: A higher-order function that renders a Plotly chart in a Streamlit container by accepting a chart creation function and its arguments.

**Functional Aspects**:

- **Higher-Order Function**: Takes a `chart_function` (callable) as a parameter, enabling flexible chart composition.
- **Core Logic**: The chart creation function, side effects are isolated to Streamlit rendering.
- **Modularity**: Allows different chart types to be rendered with consistent configuration.

```python
def render_chart_in_streamlit(chart_function: callable, container: Any, **kwargs) -> None:
    """
    Helper function to render a chart in a Streamlit container.

    Args:
        chart_function: Chart creation function
        container: Streamlit container (column, expander, etc.)
        **kwargs: Arguments to pass to chart function
    """
    try:
        fig = chart_function(**kwargs)
        container.plotly_chart(fig, config=ChartTheme.get_plotly_config())
    except Exception as e:
        logger.error(f"Error rendering chart: {e}")
        container.error(f"Failed to render chart: {str(e)}")
```

### 3. `FilterState.apply_to_dataframe`

*From: `src/ui/components/filters.py`, lines 83-106*

**Description**: Applies a series of filters to a Pandas DataFrame, returning a new filtered DataFrame.

**Functional Aspects**:

- **Function**: Returns a new DataFrame without modifying the input.
- **Final Data Structure**: Uses a copy of the input DataFrame to ensure immutability.
- **Composition**: Composes multiple filter operations (`_apply_date_filter`, `_filter_by_selection`, `_apply_range_filter`) in a functional pipeline.

```python
def apply_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all filters to a DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    filtered_df = self._apply_date_filter(filtered_df)
    filtered_df = self._filter_by_selection(filtered_df, "agent_id", self.selected_agents)
    filtered_df = self._filter_by_selection(filtered_df, "campaign", self.selected_campaigns)
    filtered_df = self._filter_by_selection(filtered_df, "outcome", self.selected_outcomes)
    filtered_df = self._filter_by_selection(filtered_df, "call_type", self.selected_types)
    filtered_df = self._apply_range_filter(
        filtered_df, "duration", self.duration_range, inclusive_upper=True
    )
    filtered_df = self._apply_range_filter(
        filtered_df, "revenue", self.revenue_range, inclusive_upper=True
    )
    filtered_df = self._apply_search_filter(filtered_df)
    return filtered_df
```

### 4. `EmbeddingManager.generate_embeddings`

*From: `src/ml/embeddings.py`, lines 273-319*

**Description**: Generates text embeddings with caching support, using a specified embedding provider.

**Functional Aspects**:

- **Core Logic**: The embedding generation; caching is an optional, controlled side effect.
- **Switchable Provider**: Internally calls whatever embedding service is configured, so you can swap providers without touching the rest of the code.
- **Final Data Structure**: Returns a NumPy array of embeddings, treated as immutable in downstream operations.

```python
def generate_embeddings(
    self,
    texts: list[str],
    use_cache: bool = True,
) -> np.ndarray:
    """
    Generate embeddings for texts with caching support.

    Args:
        texts: List of texts to embed
        use_cache: Whether to use cached embeddings

    Returns:
        Array of embeddings
    """
    if not texts:
        return np.array([])

    cache_active = use_cache and self.cache_enabled

    # Check cache
    if cache_active:
        cached_embeddings, missing_indices, texts_to_generate = self._split_cache_hits(texts)

        if not texts_to_generate:
            return self._assemble_cached_results(len(texts), cached_embeddings)
    else:
        cached_embeddings = {}
        missing_indices = list(range(len(texts)))
        texts_to_generate = list(texts)

    # Generate new embeddings
    new_embeddings = self.provider.generate(texts_to_generate)

    # Add to cache
    if self.cache_enabled:
        self._store_in_cache(texts_to_generate, new_embeddings)

    if cache_active:
        return self._assemble_results(
            len(texts),
            cached_embeddings,
            missing_indices,
            new_embeddings,
        )

    return new_embeddings
```

### 5. `QuickFilters.render`

*From: `src/ui/components/filters.py`, lines 528-567*

**Description**: Renders quick filter buttons for common scenarios while defining lambda helpers that callers can apply when a checkbox is enabled.

**Functional Aspects**:

- **Closures / Anonymous Functions**: Stores lambda predicates in `quick_filters` so downstream code can reuse the same logic when a quick filter is active.
- **Core Logic**: Returns checkbox state only; applying the lambdas (which operate on new DataFrame views) is left to the caller.
- **Modularity**: Each predicate is independent, making it easy to extend the preset catalogue.

```python
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
```

The method surfaces UI state; to filter data, calling code combines the returned `filter_states` with matching predicates (for example by rebuilding the same `quick_filters` mapping and applying the selected entries to a copy of the DataFrame).

## Conclusion

The Call Analytics System incorporates functional programming principles through immutable data structures, core logic, higher-order functions, functions as parameters, and closures/anonymous functions. While Streamlit interactions introduce some side effects, these are isolated to rendering components, preserving the functional purity of core logic. The selected functions demonstrate these principles in practice, ensuring modularity, reusability, and maintainability.
