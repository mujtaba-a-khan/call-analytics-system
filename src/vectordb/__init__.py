"""
Vector Database Package for Call Analytics System

This package handles vector storage and retrieval for semantic search.
"""

__version__ = "1.0.0"

from typing import Any, NamedTuple

__all__ = [
    "ChromaClient",
    "ChromaDBClient",
    "DocumentIndexer",
    "DocumentRetriever",
    "VectorDBError",
]


class _Export(NamedTuple):
    module: str
    attr: str


_EXPORTS: dict[str, _Export] = {
    "ChromaClient": _Export("chroma_client", "ChromaClient"),
    "ChromaDBClient": _Export("chroma_client", "ChromaClient"),
    "VectorDBError": _Export("chroma_client", "VectorDBError"),
    "DocumentIndexer": _Export("indexer", "DocumentIndexer"),
    "DocumentRetriever": _Export("retriever", "DocumentRetriever"),
}


def __getattr__(name: str) -> Any:
    """Lazy loading for vector database modules."""
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc

    module = __import__(f"{__name__}.{module_name}", fromlist=[attr])
    return getattr(module, attr)
