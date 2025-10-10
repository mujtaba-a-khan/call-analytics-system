"""
Analysis Package for Call Analytics System

This package contains analysis, aggregation, and search functionality.
"""

__version__ = '1.0.0'

__all__ = [
    'MetricsCalculator',
    'SemanticSearchEngine',
    'QueryInterpreter',
    'AdvancedFilters'
]

def __getattr__(name):
    """Lazy loading for analysis modules"""
    if name == 'MetricsCalculator':
        from .aggregations import MetricsCalculator
        return MetricsCalculator
    elif name == 'SemanticSearchEngine':
        from .semantic_search import SemanticSearchEngine
        return SemanticSearchEngine
    elif name == 'QueryInterpreter':
        from .query_interpreter import QueryInterpreter
        return QueryInterpreter
    elif name == 'AdvancedFilters':
        from .filters import AdvancedFilters
        return AdvancedFilters

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
