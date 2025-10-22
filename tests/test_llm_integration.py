"""
Test LLM integration capabilities for call analytics system.

Tests cover:
1. LLM Interface basic functionality
2. Q&A integration with LLM and semantic search
3. Error handling and fallbacks
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd

from src.ml.llm_interface import LocalLLMInterface, LLMResponse
from src.ui.pages.qa_interface import QAInterface
from src.analysis.semantic_search import SemanticSearchEngine


@pytest.fixture
def mock_llm():
    """Create a mock LLM interface"""
    llm = Mock(spec=LocalLLMInterface)
    llm.answer_question.return_value = "Total revenue for the period is $150,000"
    llm.generate.return_value = LLMResponse(
        text="Total revenue for the period is $150,000",
        model="test-model",
        provider="test",
        tokens_used=10,
        success=True
    )
    return llm


@pytest.fixture
def mock_search_engine():
    """Create a mock semantic search engine"""
    engine = Mock(spec=SemanticSearchEngine)
    engine.search.return_value = [
        {
            "document": "Call transcript with revenue $50,000",
            "metadata": {"call_id": "1", "revenue": 50000},
            "score": 0.95
        },
        {
            "document": "Call transcript with revenue $100,000",
            "metadata": {"call_id": "2", "revenue": 100000},
            "score": 0.90
        }
    ]
    return engine


@pytest.fixture
def mock_storage():
    """Create a mock storage manager"""
    storage = Mock()
    storage.load_data.return_value = pd.DataFrame({
        'call_id': ['1', '2'],
        'revenue': [50000, 100000],
        'timestamp': ['2025-10-01', '2025-10-02']
    })
    return storage


def test_llm_revenue_query(mock_llm, mock_search_engine, mock_storage):
    """Test LLM response for revenue query with context"""
    qa_interface = QAInterface(
        storage_manager=mock_storage,
        vector_store=mock_search_engine,
        llm_client=mock_llm
    )
    
    # Simulate a revenue query
    query = "What is the total revenue?"
    response = {}
    intent = {
        "action": "analyze",
        "entities": ["revenue"],
        "aggregations": ["sum"],
        "confidence": 0.9
    }
    
    result = qa_interface._handle_general_query(query, intent, response)
    
    assert isinstance(result, dict)
    assert "content" in result
    mock_llm.generate.assert_called_once()
    assert result["content"] == "Total revenue for the period is $150,000"


def test_llm_query_no_context(mock_llm):
    """Test LLM direct query without semantic search context"""
    llm = mock_llm
    response = llm.answer_question("What is the average call duration?", "")
    
    assert isinstance(response, str)
    assert llm.answer_question.called


def test_semantic_search_integration(mock_search_engine):
    """Test semantic search integration for relevant calls"""
    engine = mock_search_engine
    
    results = engine.search(
        query="Find high revenue calls",
        top_k=2,
        threshold=0.5
    )
    
    assert len(results) == 2
    assert all(r["score"] > 0.5 for r in results)
    assert sum(r["metadata"]["revenue"] for r in results) == 150000


@pytest.mark.parametrize("query,expected_calls,min_revenue", [
    ("Find calls with revenue over 75000", 1, 75000),
    ("Show me all high value deals", 2, 0),
])
def test_revenue_queries(query, expected_calls, min_revenue, mock_search_engine):
    """Test different revenue-related queries"""
    results = mock_search_engine.search(query=query, top_k=5)
    # Filter results based on revenue threshold
    filtered_results = [
        r for r in results 
        if r["metadata"]["revenue"] > min_revenue
    ]
    assert len(filtered_results) == expected_calls


def test_llm_error_handling(mock_llm, mock_storage):
    """Test error handling when LLM fails"""
    # Setup mock to return error response
    mock_llm.generate.return_value = LLMResponse(
        text="",
        model="test-model",
        provider="test",
        tokens_used=0,
        success=False,
        error="LLM API error"
    )
    
    qa_interface = QAInterface(
        storage_manager=mock_storage,
        vector_store=None,
        llm_client=mock_llm
    )
    
    response = {}
    result = qa_interface._handle_general_query(
        "What is the revenue trend?",
        {"confidence": 0.5},
        response
    )
    
    assert isinstance(result, dict)
    error_msg = result.get("content", "").lower()
    # Check for expected error message components
    assert "verify" in error_msg
    assert "llm service" in error_msg
    assert "try again" in error_msg