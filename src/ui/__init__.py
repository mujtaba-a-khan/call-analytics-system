"""
UI Package for Call Analytics System

This package contains the Streamlit user interface components including
the main application, pages, and reusable UI components.
"""

# Import main app
from .app import main as run_app

# Import pages
from .pages import (
    DashboardPage,
    AnalysisPage,
    QAInterface,
    UploadPage,
    render_dashboard_page,
    render_analysis_page,
    render_qa_interface,
    render_upload_page
)

# Import components
from .components import (
    # Charts
    ChartTheme,
    TimeSeriesChart,
    DistributionChart,
    PerformanceChart,
    TrendChart,
    
    # Filters
    FilterState,
    DateRangeFilter,
    MultiSelectFilter,
    RangeSliderFilter,
    SearchFilter,
    QuickFilters,
    
    # Metrics
    MetricValue,
    MetricCard,
    MetricsGrid,
    SummaryStats,
    KPIDashboard,
    PerformanceIndicator,
    
    # Tables
    DataTable,
    CallRecordsTable,
    AgentPerformanceTable,
    ComparisonTable
)

# Define package exports
__all__ = [
    # Main app
    'run_app',
    
    # Pages
    'DashboardPage',
    'AnalysisPage',
    'QAInterface',
    'UploadPage',
    'render_dashboard_page',
    'render_analysis_page',
    'render_qa_interface',
    'render_upload_page',
    
    # Chart components
    'ChartTheme',
    'TimeSeriesChart',
    'DistributionChart',
    'PerformanceChart',
    'TrendChart',
    
    # Filter components
    'FilterState',
    'DateRangeFilter',
    'MultiSelectFilter',
    'RangeSliderFilter',
    'SearchFilter',
    'QuickFilters',
    
    # Metric components
    'MetricValue',
    'MetricCard',
    'MetricsGrid',
    'SummaryStats',
    'KPIDashboard',
    'PerformanceIndicator',
    
    # Table components
    'DataTable',
    'CallRecordsTable',
    'AgentPerformanceTable',
    'ComparisonTable'
]

# Package version
__version__ = '1.0.0'