"""
UI Components Package for Call Analytics System

This package exports all reusable UI components for the Streamlit application.
Components are organized into charts, filters, metrics, and tables modules.
"""

# Package version
__version__ = '1.0.0'

# Core components that are frequently used can be imported directly
# Everything else uses lazy loading

# Define available exports
__all__ = [
    # Chart components
    'ChartTheme',
    'TimeSeriesChart',
    'DistributionChart',
    'PerformanceChart',
    'TrendChart',
    'render_chart_in_streamlit',

    # Filter components
    'FilterState',
    'DateRangeFilter',
    'MultiSelectFilter',
    'RangeSliderFilter',
    'SearchFilter',
    'QuickFilters',
    'save_filter_preset',
    'load_filter_preset',

    # Metrics components
    'MetricValue',
    'MetricCard',
    'MetricsGrid',
    'SummaryStats',
    'KPIDashboard',
    'PerformanceIndicator',
    'ProgressIndicator',

    # Table components
    'DataTable',
    'CallRecordsTable',
    'AgentPerformanceTable',
    'ComparisonTable'
]

def __getattr__(name):
    """
    Lazy loading of UI components.
    Components are only imported when accessed.
    """
    # Chart components
    if name in ['ChartTheme', 'TimeSeriesChart', 'DistributionChart',
                'PerformanceChart', 'TrendChart', 'render_chart_in_streamlit']:
        from . import charts
        return getattr(charts, name)

    # Filter components
    elif name in ['FilterState', 'DateRangeFilter', 'MultiSelectFilter',
                  'RangeSliderFilter', 'SearchFilter', 'QuickFilters',
                  'save_filter_preset', 'load_filter_preset']:
        from . import filters
        return getattr(filters, name)

    # Metrics components
    elif name in ['MetricValue', 'MetricCard', 'MetricsGrid', 'SummaryStats',
                  'KPIDashboard', 'PerformanceIndicator', 'ProgressIndicator']:
        from . import metrics
        return getattr(metrics, name)

    # Table components
    elif name in ['DataTable', 'CallRecordsTable', 'AgentPerformanceTable',
                  'ComparisonTable']:
        from . import tables
        return getattr(tables, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
