"""
Machine Learning Package for Call Analytics System

This package contains ML models and inference engines.
"""

import importlib.util

__version__ = '1.0.0'

def get_ml_capabilities():
    """
    Check available ML capabilities.

    Returns:
        Dict[str, bool]: Available capabilities
    """
    capabilities = {}

    # Check Whisper
    capabilities['whisper'] = importlib.util.find_spec('whisper') is not None

    # Check sentence transformers
    capabilities['sentence_transformers'] = (
        importlib.util.find_spec('sentence_transformers') is not None
    )

    # Check Torch
    try:
        import torch
        capabilities['torch'] = True
        capabilities['cuda'] = torch.cuda.is_available()
    except ImportError:
        capabilities['torch'] = False
        capabilities['cuda'] = False

    # Check Ollama
    capabilities['ollama'] = importlib.util.find_spec('ollama') is not None

    return capabilities

__all__ = ['get_ml_capabilities']

def __getattr__(name):
    """Lazy loading for ML modules"""
    if name == 'WhisperSTT':
        from .whisper_stt import WhisperSTT
        return WhisperSTT
    elif name == 'EmbeddingManager':
        from .embeddings import EmbeddingManager
        return EmbeddingManager
    elif name == 'LLMInterface':
        from .llm_interface import LLMInterface
        return LLMInterface

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
