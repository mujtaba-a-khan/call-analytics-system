"""
UI Components Package for Call Analytics System

This package exports all reusable UI components for the Streamlit application.
Components are organized into charts, filters, metrics, and tables modules.
"""

# Import chart components
from .charts import (
    ChartTheme,
    TimeSeriesChart,
    DistributionChart,
    PerformanceChart,
    TrendChart,
    render_chart_in_streamlit
)

# Import filter components
from .filters import (
    FilterState,
    DateRangeFilter,
    MultiSelectFilter,
    RangeSliderFilter,
    SearchFilter,
    QuickFilters,
    save_filter_preset,
    load_filter_preset
)

# Import metrics components
from .metrics import (
    MetricValue,
    MetricCard,
    MetricsGrid,
    SummaryStats,
    KPIDashboard,
    PerformanceIndicator
)

# Import table components
from .tables import (
    DataTable,
    CallRecordsTable,
    AgentPerformanceTable,
    ComparisonTable
)

# Define package exports
__all__ = [
    # Charts
    'ChartTheme',
    'TimeSeriesChart',
    'DistributionChart',
    'PerformanceChart',
    'TrendChart',
    'render_chart_in_streamlit',
    
    # Filters
    'FilterState',
    'DateRangeFilter',
    'MultiSelectFilter',
    'RangeSliderFilter',
    'SearchFilter',
    'QuickFilters',
    'save_filter_preset',
    'load_filter_preset',
    
    # Metrics
    'MetricValue',
    'MetricCard',
    'MetricsGrid',
    'SummaryStats',
    'KPIDashboard',
    'PerformanceIndicator',
    
    # Tables
    'DataTable',
    'CallRecordsTable',
    'AgentPerformanceTable',
    'ComparisonTable'
]

# Package version
__version__ = '1.0.0'