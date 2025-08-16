"""
Analysis Package for Call Analytics System

This package provides advanced analytics, search, and filtering capabilities
for call data including metrics calculation, semantic search, and natural
language query interpretation.
"""

# Import aggregations module
from .aggregations import (
    MetricsCalculator,
    CallMetrics,
    AgentMetrics,
    CampaignMetrics,
    TimeSeriesMetrics
)

# Import filters module
from .filters import (
    AdvancedFilters,
    FilterCriteria,
    FilterOperator,
    DateRangeFilter,
    NumericRangeFilter
)

# Import semantic search module
from .semantic_search import (
    SemanticSearchEngine,
    SearchConfig,
    SearchResult,
    SimilarityScorer
)

# Import query interpreter module
from .query_interpreter import (
    QueryInterpreter,
    QueryIntent,
    EntityExtractor,
    NaturalLanguageProcessor
)

# Define package exports
__all__ = [
    # Aggregations
    'MetricsCalculator',
    'CallMetrics',
    'AgentMetrics',
    'CampaignMetrics',
    'TimeSeriesMetrics',
    
    # Filters
    'AdvancedFilters',
    'FilterCriteria',
    'FilterOperator',
    'DateRangeFilter',
    'NumericRangeFilter',
    
    # Semantic Search
    'SemanticSearchEngine',
    'SearchConfig',
    'SearchResult',
    'SimilarityScorer',
    
    # Query Interpreter
    'QueryInterpreter',
    'QueryIntent',
    'EntityExtractor',
    'NaturalLanguageProcessor'
]

# Package version
__version__ = '1.0.0'

def get_analysis_capabilities():
    """
    Get information about available analysis capabilities.
    
    Returns:
        Dictionary of capability information
    """
    capabilities = {
        'metrics_calculation': True,
        'advanced_filtering': True,
        'semantic_search': True,
        'natural_language_queries': True,
        'time_series_analysis': True,
        'cohort_analysis': True,
        'predictive_analytics': False  # Future feature
    }
    
    return capabilities