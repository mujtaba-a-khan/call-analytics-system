"""
Machine Learning Package for Call Analytics System

This package contains machine learning models and processing modules
including speech-to-text, embeddings generation, and LLM integration.
"""

# Import Whisper STT
from .whisper_stt import (
    WhisperSTT,
    TranscriptionConfig,
    TranscriptionResult
)

# Import embeddings
from .embeddings import (
    EmbeddingStrategy,
    SentenceTransformerEmbedding,
    OllamaEmbedding,
    HashEmbedding,
    EmbeddingManager
)

# Import LLM client (if available)
try:
    from .llm_client import (
        LLMClient,
        OllamaClient,
        LLMConfig,
        LLMResponse
    )
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLMClient = None
    OllamaClient = None
    LLMConfig = None
    LLMResponse = None

# Define package exports
__all__ = [
    # STT
    'WhisperSTT',
    'TranscriptionConfig',
    'TranscriptionResult',
    
    # Embeddings
    'EmbeddingStrategy',
    'SentenceTransformerEmbedding',
    'OllamaEmbedding',
    'HashEmbedding',
    'EmbeddingManager',
]

# Add LLM exports if available
if LLM_AVAILABLE:
    __all__.extend([
        'LLMClient',
        'OllamaClient',
        'LLMConfig',
        'LLMResponse'
    ])

# Package version
__version__ = '1.0.0'

# Package info
def get_ml_capabilities():
    """
    Get information about available ML capabilities.
    
    Returns:
        Dictionary of capability status
    """
    capabilities = {
        'whisper_stt': True,
        'embeddings': True,
        'llm': LLM_AVAILABLE
    }
    
    # Check for specific models
    try:
        import torch
        capabilities['pytorch'] = True
    except ImportError:
        capabilities['pytorch'] = False
    
    try:
        import sentence_transformers
        capabilities['sentence_transformers'] = True
    except ImportError:
        capabilities['sentence_transformers'] = False
    
    try:
        import ollama
        capabilities['ollama'] = True
    except ImportError:
        capabilities['ollama'] = False
    
    return capabilities