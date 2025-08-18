"""
Machine Learning Package for Call Analytics System

This package contains ML models and inference engines.
"""

__version__ = '1.0.0'

def get_ml_capabilities():
    """
    Check available ML capabilities.
    
    Returns:
        Dict[str, bool]: Available capabilities
    """
    capabilities = {}
    
    # Check Whisper
    try:
        import whisper
        capabilities['whisper'] = True
    except ImportError:
        capabilities['whisper'] = False
    
    # Check sentence transformers
    try:
        import sentence_transformers
        capabilities['sentence_transformers'] = True
    except ImportError:
        capabilities['sentence_transformers'] = False
    
    # Check Torch
    try:
        import torch
        capabilities['torch'] = True
        capabilities['cuda'] = torch.cuda.is_available()
    except ImportError:
        capabilities['torch'] = False
        capabilities['cuda'] = False
    
    # Check Ollama
    try:
        import ollama
        capabilities['ollama'] = True
    except ImportError:
        capabilities['ollama'] = False
    
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