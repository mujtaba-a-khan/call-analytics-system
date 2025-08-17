"""
Test Suite for Query Interpreter Module

Tests for natural language query interpretation, intent detection,
and entity extraction in the Call Analytics System.
"""

import unittest
from datetime import datetime, date, timedelta

from src.analysis.query_interpreter import (
    QueryInterpreter,
    QueryIntent,
    EntityExtractor,
    NaturalLanguageProcessor
)
from tests.test_analysis import ANALYSIS_TEST_DATA_DIR


class TestQueryInterpreter(unittest.TestCase):
    """Test cases for QueryInterpreter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.interpreter = QueryInterpreter()
    
    def test_interpret_metric_queries(self):
        """Test interpreting metric-related queries."""
        test_cases = [
            ("What is the average call duration?", {
                'type': 'metric',
                'metric': 'average_duration',
                'time_range': 'all'
            }),
            ("How many calls did we have today?", {
                'type': 'metric',
                'metric': 'total_calls',
                'time_range': 'today'
            }),
            ("Show me the connection rate last week", {
                'type': 'metric',
                'metric': 'connection_rate',
                'time_range': 'last_week'
            }),
            ("What's the total revenue this month?", {
                'type': 'metric',
                'metric': 'total_revenue',
                'time_range': 'this_month'
            })
        ]
        
        for query, expected in test_cases:
            intent = self.interpreter.interpret(query)
            self.assertEqual(intent['type'], expected['type'])
            self.assertEqual(intent['metric'], expected['metric'])
            self.assertEqual(intent['time_range'], expected['time_range'])
    
    def test_interpret_search_queries(self):
        """Test interpreting search queries."""
        test_cases = [
            ("Find calls about billing issues", {
                'type': 'search',
                'search_terms': ['billing', 'issues']
            }),
            ("Show me customer complaints", {
                'type': 'search',
                'search_terms': ['customer', 'complaints']
            }),
            ("Search for technical support calls", {
                'type': 'search',
                'search_terms': ['technical', 'support', 'calls']
            })
        ]
        
        for query, expected in test_cases:
            intent = self.interpreter.interpret(query)
            self.assertEqual(intent['type'], expected['type'])
            for term in expected['search_terms']:
                self.assertIn(term, intent['search_terms'])
    
    def test_interpret_comparison_queries(self):
        """Test interpreting comparison queries."""
        test_cases = [
            ("Compare this week to last week", {
                'type': 'comparison',
                'compare': 'periods',
                'periods': ['this_week', 'last_week']
            }),
            ("Compare agent performance", {
                'type': 'comparison',
                'compare': 'agents'
            }),
            ("Compare sales and support campaigns", {
                'type': 'comparison',
                'compare': 'campaigns',
                'entities': ['sales', 'support']
            })
        ]
        
        for query, expected in test_cases:
            intent = self.interpreter.interpret(query)
            self.assertEqual(intent['type'], expected['type'])
            self.assertEqual(intent['compare'], expected['compare'])
            if 'periods' in expected:
                for period in expected['periods']:
                    self.assertIn(period, intent['periods'])
    
    def test_interpret_analysis_queries(self):
        """Test interpreting analysis queries."""
        test_cases = [
            ("Analyze call trends", {
                'type': 'analysis',
                'analysis_type': 'trend'
            }),
            ("Show agent performance", {
                'type': 'analysis',
                'analysis_type': 'performance',
                'entity': 'agent'
            }),
            ("Analyze peak hours", {
                'type': 'analysis',
                'analysis_type': 'peak_hours'
            })
        ]
        
        for query, expected in test_cases:
            intent = self.interpreter.interpret(query)
            self.assertEqual(intent['type'], expected['type'])
            self.assertEqual(intent['analysis_type'], expected['analysis_type'])
            if 'entity' in expected:
                self.assertEqual(intent['entity'], expected['entity'])
    
    def test_interpret_filter_queries(self):
        """Test interpreting filter-based queries."""
        test_cases = [
            ("Show calls longer than 5 minutes", {
                'type': 'filter',
                'filters': {
                    'duration': {'operator': '>', 'value': 300}
                }
            }),
            ("Show connected calls from agent_001", {
                'type': 'filter',
                'filters': {
                    'outcome': 'connected',
                    'agent_id': 'agent_001'
                }
            })
        ]
        
        for query, expected in test_cases:
            intent = self.interpreter.interpret(query)
            self.assertEqual(intent['type'], expected['type'])
            self.assertIn('filters', intent)
    
    def test_interpret_ambiguous_queries(self):
        """Test handling ambiguous queries."""
        ambiguous_queries = [
            "Show me the data",
            "What's happening?",
            "Give me information",
            "Help"
        ]
        
        for query in ambiguous_queries:
            intent = self.interpreter.interpret(query)
            self.assertEqual(intent['type'], 'general')
            self.assertIn('clarification_needed', intent)
            self.assertTrue(intent['clarification_needed'])
    
    def test_extract_time_ranges(self):
        """Test extracting time ranges from queries."""
        test_cases = [
            ("Show me data from yesterday", 'yesterday'),
            ("What happened today?", 'today'),
            ("Last week's performance", 'last_week'),
            ("This month's revenue", 'this_month'),
            ("Calls from last 7 days", 'last_7_days'),
            ("Previous quarter results", 'last_quarter')
        ]
        
        for query, expected_range in test_cases:
            time_range = self.interpreter.extract_time_range(query)
            self.assertEqual(time_range, expected_range)
    
    def test_extract_metrics(self):
        """Test extracting metrics from queries."""
        test_cases = [
            ("average duration", 'average_duration'),
            ("total calls", 'total_calls'),
            ("connection rate", 'connection_rate'),
            ("revenue", 'revenue'),
            ("conversion rate", 'conversion_rate')
        ]
        
        for query, expected_metric in test_cases:
            metric = self.interpreter.extract_metric(query)
            self.assertEqual(metric, expected_metric)


class TestEntityExtractor(unittest.TestCase):
    """Test cases for EntityExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = EntityExtractor()
    
    def test_extract_agent_ids(self):
        """Test extracting agent IDs from text."""
        test_cases = [
            ("Show me calls from agent_001", ['agent_001']),
            ("Compare agent_001 and agent_002", ['agent_001', 'agent_002']),
            ("Agent performance for agent_123", ['agent_123'])
        ]
        
        for text, expected_agents in test_cases:
            agents = self.extractor.extract_agents(text)
            self.assertEqual(agents, expected_agents)
    
    def test_extract_campaigns(self):
        """Test extracting campaign names from text."""
        test_cases = [
            ("Show sales campaign results", ['sales']),
            ("Compare sales and support campaigns", ['sales', 'support']),
            ("Billing campaign performance", ['billing'])
        ]
        
        for text, expected_campaigns in test_cases:
            campaigns = self.extractor.extract_campaigns(text)
            for campaign in expected_campaigns:
                self.assertIn(campaign, campaigns)
    
    def test_extract_phone_numbers(self):
        """Test extracting phone numbers from text."""
        test_cases = [
            ("Find calls from +1234567890", ['+1234567890']),
            ("Calls to 555-1234", ['555-1234']),
            ("Number (555) 123-4567", ['(555) 123-4567'])
        ]
        
        for text, expected_numbers in test_cases:
            numbers = self.extractor.extract_phone_numbers(text)
            self.assertEqual(len(numbers), len(expected_numbers))
            for number in expected_numbers:
                self.assertIn(number, numbers)
    
    def test_extract_dates(self):
        """Test extracting dates from text."""
        test_cases = [
            ("Show calls from January 15, 2024", [date(2024, 1, 15)]),
            ("Data from 2024-01-15 to 2024-01-20", [date(2024, 1, 15), date(2024, 1, 20)]),
            ("Calls on 01/15/2024", [date(2024, 1, 15)])
        ]
        
        for text, expected_dates in test_cases:
            dates = self.extractor.extract_dates(text)
            self.assertEqual(len(dates), len(expected_dates))
            for expected_date in expected_dates:
                self.assertIn(expected_date, dates)
    
    def test_extract_durations(self):
        """Test extracting durations from text."""
        test_cases = [
            ("Calls longer than 5 minutes", 300),  # 5 minutes in seconds
            ("Shorter than 30 seconds", 30),
            ("Between 1 and 10 minutes", (60, 600))
        ]
        
        for text, expected_duration in test_cases:
            duration = self.extractor.extract_duration(text)
            if isinstance(expected_duration, tuple):
                self.assertIsInstance(duration, tuple)
                self.assertEqual(duration[0], expected_duration[0])
                self.assertEqual(duration[1], expected_duration[1])
            else:
                self.assertEqual(duration, expected_duration)
    
    def test_extract_outcomes(self):
        """Test extracting call outcomes from text."""
        test_cases = [
            ("Show connected calls", ['connected']),
            ("Failed and no answer calls", ['failed', 'no_answer']),
            ("Voicemail outcomes", ['voicemail'])
        ]
        
        for text, expected_outcomes in test_cases:
            outcomes = self.extractor.extract_outcomes(text)
            for outcome in expected_outcomes:
                self.assertIn(outcome, outcomes)
    
    def test_extract_multiple_entities(self):
        """Test extracting multiple entity types from a single query."""
        query = "Show connected calls from agent_001 in the sales campaign longer than 5 minutes"
        
        entities = self.extractor.extract_all(query)
        
        self.assertIn('agents', entities)
        self.assertIn('agent_001', entities['agents'])
        
        self.assertIn('outcomes', entities)
        self.assertIn('connected', entities['outcomes'])
        
        self.assertIn('campaigns', entities)
        self.assertIn('sales', entities['campaigns'])
        
        self.assertIn('duration', entities)
        self.assertEqual(entities['duration'], 300)  # 5 minutes


class TestQueryIntent(unittest.TestCase):
    """Test cases for QueryIntent class."""
    
    def test_intent_creation(self):
        """Test creating QueryIntent instance."""
        intent = QueryIntent(
            type='metric',
            confidence=0.95,
            entities={'metric': 'average_duration', 'time_range': 'today'}
        )
        
        self.assertEqual(intent.type, 'metric')
        self.assertAlmostEqual(intent.confidence, 0.95)
        self.assertEqual(intent.entities['metric'], 'average_duration')
        self.assertEqual(intent.entities['time_range'], 'today')
    
    def test_intent_confidence_validation(self):
        """Test intent confidence validation."""
        # Test valid confidence
        intent = QueryIntent('search', 0.8)
        self.assertAlmostEqual(intent.confidence, 0.8)
        
        # Test invalid confidence
        with self.assertRaises(ValueError):
            QueryIntent('search', 1.5)
        
        with self.assertRaises(ValueError):
            QueryIntent('search', -0.1)
    
    def test_intent_to_dict(self):
        """Test converting intent to dictionary."""
        intent = QueryIntent(
            type='comparison',
            confidence=0.9,
            entities={'compare': 'periods', 'periods': ['this_week', 'last_week']}
        )
        
        intent_dict = intent.to_dict()
        
        self.assertIsInstance(intent_dict, dict)
        self.assertEqual(intent_dict['type'], 'comparison')
        self.assertEqual(intent_dict['confidence'], 0.9)
        self.assertEqual(intent_dict['entities']['compare'], 'periods')
    
    def test_intent_types(self):
        """Test valid intent types."""
        valid_types = [
            'metric', 'search', 'comparison', 'analysis',
            'filter', 'general', 'export', 'help'
        ]
        
        for intent_type in valid_types:
            intent = QueryIntent(intent_type)
            self.assertEqual(intent.type, intent_type)


class TestNaturalLanguageProcessor(unittest.TestCase):
    """Test cases for NaturalLanguageProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.nlp = NaturalLanguageProcessor()
    
    def test_tokenize(self):
        """Test text tokenization."""
        text = "Show me the average call duration"
        tokens = self.nlp.tokenize(text)
        
        expected_tokens = ['show', 'me', 'the', 'average', 'call', 'duration']
        self.assertEqual(tokens, expected_tokens)
    
    def test_remove_stopwords(self):
        """Test stopword removal."""
        tokens = ['show', 'me', 'the', 'average', 'call', 'duration']
        filtered = self.nlp.remove_stopwords(tokens)
        
        # 'show', 'me', 'the' should be removed as stopwords
        self.assertIn('average', filtered)
        self.assertIn('call', filtered)
        self.assertIn('duration', filtered)
        self.assertNotIn('the', filtered)
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        text = "What is the average call duration for connected calls in the sales campaign?"
        keywords = self.nlp.extract_keywords(text)
        
        important_keywords = ['average', 'call', 'duration', 'connected', 'sales', 'campaign']
        for keyword in important_keywords:
            self.assertIn(keyword, keywords)
    
    def test_detect_question_type(self):
        """Test detecting question type."""
        test_cases = [
            ("What is the average duration?", 'what'),
            ("How many calls were made?", 'how_many'),
            ("When did the call happen?", 'when'),
            ("Where are the calls from?", 'where'),
            ("Who made the most calls?", 'who'),
            ("Why did calls fail?", 'why')
        ]
        
        for question, expected_type in test_cases:
            question_type = self.nlp.detect_question_type(question)
            self.assertEqual(question_type, expected_type)
    
    def test_normalize_text(self):
        """Test text normalization."""
        text = "SHOW ME THE CALLS!!! From Agent_001... TODAY???"
        normalized = self.nlp.normalize(text)
        
        # Should be lowercase and cleaned
        self.assertEqual(normalized, "show me the calls from agent_001 today")
    
    def test_extract_numeric_values(self):
        """Test extracting numeric values from text."""
        test_cases = [
            ("Show calls longer than 5 minutes", [5]),
            ("Between 100 and 200 calls", [100, 200]),
            ("Top 10 agents", [10]),
            ("Greater than 50%", [50])
        ]
        
        for text, expected_numbers in test_cases:
            numbers = self.nlp.extract_numbers(text)
            self.assertEqual(numbers, expected_numbers)
    
    def test_similarity_scoring(self):
        """Test text similarity scoring."""
        text1 = "average call duration"
        text2 = "mean call length"
        text3 = "total revenue"
        
        # Similar texts should have high score
        similarity_12 = self.nlp.calculate_similarity(text1, text2)
        self.assertGreater(similarity_12, 0.5)
        
        # Different texts should have low score
        similarity_13 = self.nlp.calculate_similarity(text1, text3)
        self.assertLess(similarity_13, 0.5)


if __name__ == '__main__':
    unittest.main()