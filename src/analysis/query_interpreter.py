"""
Query Interpreter Module

Interprets natural language queries and converts them
into structured filter specifications for data retrieval.
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import dateparser
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QueryIntent:
    """Container for parsed query intent"""
    action: str  # search, filter, aggregate, compare
    entities: Dict[str, List[str]]
    time_range: Optional[Tuple[datetime, datetime]]
    filters: Dict[str, Any]
    aggregations: List[str]
    confidence: float


class QueryInterpreter:
    """
    Interprets natural language queries and extracts structured information
    for filtering and searching call data.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize the query interpreter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.patterns = self._compile_patterns()
        self.entity_extractors = self._initialize_extractors()
        
        logger.info("QueryInterpreter initialized")
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """
        Compile regex patterns for query parsing.
        
        Returns:
            Dictionary of compiled patterns
        """
        patterns = {
            # Time patterns
            'last_n_days': re.compile(r'last\s+(\d+)\s+days?', re.IGNORECASE),
            'last_n_hours': re.compile(r'last\s+(\d+)\s+hours?', re.IGNORECASE),
            'yesterday': re.compile(r'\byesterday\b', re.IGNORECASE),
            'today': re.compile(r'\btoday\b', re.IGNORECASE),
            'this_week': re.compile(r'this\s+week', re.IGNORECASE),
            'last_week': re.compile(r'last\s+week', re.IGNORECASE),
            'this_month': re.compile(r'this\s+month', re.IGNORECASE),
            'last_month': re.compile(r'last\s+month', re.IGNORECASE),
            
            # Entity patterns
            'agent': re.compile(r'agent\s+(\w+)', re.IGNORECASE),
            'campaign': re.compile(r'campaign\s+"([^"]+)"', re.IGNORECASE),
            'customer': re.compile(r'customer\s+(\w+)', re.IGNORECASE),
            
            # Type patterns
            'call_type': re.compile(r'(inquiry|support|complaint|billing|sales)', re.IGNORECASE),
            'outcome': re.compile(r'(resolved|callback|refund|sale)', re.IGNORECASE),
            
            # Aggregation patterns
            'count': re.compile(r'\b(count|number|how many)\b', re.IGNORECASE),
            'average': re.compile(r'\b(average|avg|mean)\b', re.IGNORECASE),
            'sum': re.compile(r'\b(sum|total)\b', re.IGNORECASE),
            'max': re.compile(r'\b(max|maximum|highest)\b', re.IGNORECASE),
            'min': re.compile(r'\b(min|minimum|lowest)\b', re.IGNORECASE),
            
            # Comparison patterns
            'greater_than': re.compile(r'(greater than|more than|over|above|>)\s*(\d+)', re.IGNORECASE),
            'less_than': re.compile(r'(less than|fewer than|under|below|<)\s*(\d+)', re.IGNORECASE),
            'between': re.compile(r'between\s+(\d+)\s+and\s+(\d+)', re.IGNORECASE),
            
            # Action patterns
            'show': re.compile(r'\b(show|display|list|get)\b', re.IGNORECASE),
            'find': re.compile(r'\b(find|search|look for)\b', re.IGNORECASE),
            'compare': re.compile(r'\b(compare|versus|vs)\b', re.IGNORECASE),
            'analyze': re.compile(r'\b(analyze|analysis|breakdown)\b', re.IGNORECASE),
        }
        
        return patterns
    
    def _initialize_extractors(self) -> Dict[str, Any]:
        """
        Initialize entity extractors.
        
        Returns:
            Dictionary of extractor functions
        """
        return {
            'time': self._extract_time_range,
            'entities': self._extract_entities,
            'filters': self._extract_filters,
            'aggregations': self._extract_aggregations,
            'action': self._extract_action
        }
    
    def interpret(self, query: str) -> QueryIntent:
        """
        Interpret a natural language query.
        
        Args:
            query: Natural language query string
        
        Returns:
            QueryIntent object with parsed information
        """
        query_lower = query.lower()
        
        # Extract components
        action = self._extract_action(query_lower)
        entities = self._extract_entities(query_lower)
        time_range = self._extract_time_range(query_lower)
        filters = self._extract_filters(query_lower)
        aggregations = self._extract_aggregations(query_lower)
        
        # Calculate confidence based on successful extractions
        confidence = self._calculate_confidence(
            action, entities, time_range, filters, aggregations
        )
        
        intent = QueryIntent(
            action=action,
            entities=entities,
            time_range=time_range,
            filters=filters,
            aggregations=aggregations,
            confidence=confidence
        )
        
        logger.debug(f"Interpreted query: '{query[:50]}...' with confidence {confidence:.2f}")
        
        return intent
    
    def _extract_action(self, query: str) -> str:
        """
        Extract the primary action from the query.
        
        Args:
            query: Query string
        
        Returns:
            Action type
        """
        if self.patterns['compare'].search(query):
            return 'compare'
        elif self.patterns['analyze'].search(query):
            return 'analyze'
        elif self.patterns['find'].search(query):
            return 'search'
        elif any(self.patterns[agg].search(query) for agg in ['count', 'average', 'sum']):
            return 'aggregate'
        else:
            return 'filter'
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract entities from the query.
        
        Args:
            query: Query string
        
        Returns:
            Dictionary of entity types to values
        """
        entities = {
            'agents': [],
            'campaigns': [],
            'customers': [],
            'call_types': [],
            'outcomes': []
        }
        
        # Extract agents
        agent_matches = self.patterns['agent'].findall(query)
        entities['agents'] = agent_matches
        
        # Extract campaigns
        campaign_matches = self.patterns['campaign'].findall(query)
        entities['campaigns'] = campaign_matches
        
        # Extract call types
        type_matches = self.patterns['call_type'].findall(query)
        entities['call_types'] = [t.capitalize() for t in type_matches]
        
        # Extract outcomes
        outcome_matches = self.patterns['outcome'].findall(query)
        entities['outcomes'] = [o.capitalize() for o in outcome_matches]
        
        return entities
    
    def _extract_time_range(self, query: str) -> Optional[Tuple[datetime, datetime]]:
        """
        Extract time range from the query.
        
        Args:
            query: Query string
        
        Returns:
            Tuple of (start_date, end_date) or None
        """
        now = datetime.now()
        
        # Check for last N days
        match = self.patterns['last_n_days'].search(query)
        if match:
            days = int(match.group(1))
            start_date = now - timedelta(days=days)
            return (start_date, now)
        
        # Check for last N hours
        match = self.patterns['last_n_hours'].search(query)
        if match:
            hours = int(match.group(1))
            start_date = now - timedelta(hours=hours)
            return (start_date, now)
        
        # Check for yesterday
        if self.patterns['yesterday'].search(query):
            yesterday = now - timedelta(days=1)
            start_date = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
            return (start_date, end_date)
        
        # Check for today
        if self.patterns['today'].search(query):
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return (start_date, now)
        
        # Check for this week
        if self.patterns['this_week'].search(query):
            start_date = now - timedelta(days=now.weekday())
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
            return (start_date, now)
        
        # Check for last week
        if self.patterns['last_week'].search(query):
            start_date = now - timedelta(days=now.weekday() + 7)
            end_date = start_date + timedelta(days=6)
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
            return (start_date, end_date)
        
        # Try to parse explicit numeric day (e.g., 2025-11-11 or 11/11/2025)
        numeric_day_patterns = [
            (re.compile(r'\b(\d{4})[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12][0-9]|3[01])\b'), 'ymd'),
            (re.compile(r'\b(0?[1-9]|1[0-2])[-/](0?[1-9]|[12][0-9]|3[01])[-/](\d{4})\b'), 'mdy')
        ]
        for pattern, fmt in numeric_day_patterns:
            match = pattern.search(query)
            if match:
                groups = list(map(int, match.groups()))
                if fmt == 'ymd':
                    year, month, day = groups
                else:
                    month, day, year = groups
                try:
                    start_date = datetime(year, month, day, 0, 0, 0)
                    end_date = datetime(year, month, day, 23, 59, 59, 999999)
                    return (start_date, end_date)
                except ValueError:
                    pass

        # Month name (full or abbreviated) + year (e.g., "november 2025" or "nov 2025")
        month_name_map = {
            'jan': 1, 'january': 1,
            'feb': 2, 'february': 2,
            'mar': 3, 'march': 3,
            'apr': 4, 'april': 4,
            'may': 5,
            'jun': 6, 'june': 6,
            'jul': 7, 'july': 7,
            'aug': 8, 'august': 8,
            'sep': 9, 'sept': 9, 'september': 9,
            'oct': 10, 'october': 10,
            'nov': 11, 'november': 11,
            'dec': 12, 'december': 12
        }
        month_words_pattern = re.compile(r'\b(' + '|'.join(month_name_map.keys()) + r')\b\s*(\d{4})', re.IGNORECASE)
        month_words_match = month_words_pattern.search(query)
        if month_words_match:
            month_word, year = month_words_match.groups()
            month_index = month_name_map[month_word.lower()]
            year = int(year)
            start_date = datetime(year, month_index, 1)
            if month_index == 12:
                next_month = datetime(year + 1, 1, 1)
            else:
                next_month = datetime(year, month_index + 1, 1)
            end_date = next_month - timedelta(microseconds=1)
            return (start_date, end_date)

        # Numeric month/year formats (e.g., 2025-11 or 11/2025)
        numeric_month_patterns = [
            (re.compile(r'\b(\d{4})[-/](0?[1-9]|1[0-2])\b'), 'ym'),
            (re.compile(r'\b(0?[1-9]|1[0-2])[-/](\d{4})\b'), 'my')
        ]
        for pattern, fmt in numeric_month_patterns:
            match = pattern.search(query)
            if match:
                year, month = match.groups()
                if fmt == 'my':
                    month, year = year, month
                year, month = int(year), int(month)
                start_date = datetime(year, month, 1)
                if month == 12:
                    next_month = datetime(year + 1, 1, 1)
                else:
                    next_month = datetime(year, month + 1, 1)
                end_date = next_month - timedelta(microseconds=1)
                return (start_date, end_date)

        # Try to parse remaining date expressions with dateparser
        try:
            settings = {'TIMEZONE': 'UTC', 'PREFER_DAY_OF_MONTH': 'first'}
            parsed_date = dateparser.parse(query, settings=settings)
            if parsed_date:
                month_keywords = ['january', 'february', 'march', 'april', 'may', 'june',
                                   'july', 'august', 'september', 'october', 'november', 'december']
                contains_month = any(keyword in query for keyword in month_keywords)
                day_pattern = re.compile(r'\b(0?[1-9]|[12][0-9]|3[01])\b')
                has_explicit_day = bool(day_pattern.search(query))

                if contains_month and not has_explicit_day:
                    start_date = parsed_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                    if start_date.month == 12:
                        next_month = start_date.replace(year=start_date.year + 1, month=1)
                    else:
                        next_month = start_date.replace(month=start_date.month + 1)
                    end_date = next_month - timedelta(microseconds=1)
                else:
                    start_date = parsed_date.replace(hour=0, minute=0, second=0, microsecond=0)
                    end_date = parsed_date.replace(hour=23, minute=59, second=59, microsecond=999999)
                return (start_date, end_date)
        except Exception as e:
            logger.debug(f"dateparser failed for query '{query}': {e}")

        return None
    
    def _extract_filters(self, query: str) -> Dict[str, Any]:
        """
        Extract filter conditions from the query.
        
        Args:
            query: Query string
        
        Returns:
            Dictionary of filter conditions
        """
        filters = {}
        
        # Duration filters
        match = self.patterns['greater_than'].search(query)
        if match and 'duration' in query:
            filters['min_duration'] = int(match.group(2))
        
        match = self.patterns['less_than'].search(query)
        if match and 'duration' in query:
            filters['max_duration'] = int(match.group(2))
        
        match = self.patterns['between'].search(query)
        if match and 'duration' in query:
            filters['min_duration'] = int(match.group(1))
            filters['max_duration'] = int(match.group(2))
        
        # Amount filters
        if 'amount' in query or '$' in query:
            match = self.patterns['greater_than'].search(query)
            if match:
                filters['min_amount'] = float(match.group(2))
            
            match = self.patterns['less_than'].search(query)
            if match:
                filters['max_amount'] = float(match.group(2))
        
        # Connection status
        if 'connected' in query:
            filters['connection_status'] = 'Connected'
        elif 'disconnected' in query:
            filters['connection_status'] = 'Disconnected'
        
        return filters
    
    def _extract_aggregations(self, query: str) -> List[str]:
        """
        Extract aggregation operations from the query.
        
        Args:
            query: Query string
        
        Returns:
            List of aggregation operations
        """
        aggregations = []
        
        if self.patterns['count'].search(query):
            aggregations.append('count')
        
        if self.patterns['average'].search(query):
            aggregations.append('average')
        
        if self.patterns['sum'].search(query):
            aggregations.append('sum')
        
        if self.patterns['max'].search(query):
            aggregations.append('max')
        
        if self.patterns['min'].search(query):
            aggregations.append('min')
        
        return aggregations
    
    def _calculate_confidence(self, 
                             action: str,
                             entities: Dict[str, List[str]],
                             time_range: Optional[Tuple[datetime, datetime]],
                             filters: Dict[str, Any],
                             aggregations: List[str]) -> float:
        """
        Calculate confidence score for the interpretation.
        
        Args:
            action: Extracted action
            entities: Extracted entities
            time_range: Extracted time range
            filters: Extracted filters
            aggregations: Extracted aggregations
        
        Returns:
            Confidence score between 0 and 1
        """
        score = 0.0
        components = 0
        
        # Action contributes to confidence
        if action != 'filter':  # Default action
            score += 0.2
            components += 1
        
        # Entities contribute
        entity_count = sum(len(v) for v in entities.values())
        if entity_count > 0:
            score += min(0.3, entity_count * 0.1)
            components += 1
        
        # Time range contributes
        if time_range:
            score += 0.2
            components += 1
        
        # Filters contribute
        if filters:
            score += min(0.2, len(filters) * 0.1)
            components += 1
        
        # Aggregations contribute
        if aggregations:
            score += min(0.1, len(aggregations) * 0.05)
            components += 1
        
        # Calculate final confidence
        if components > 0:
            confidence = min(1.0, score + (components * 0.1))
        else:
            confidence = 0.1  # Minimum confidence
        
        return confidence
    
    def to_filter_spec(self, intent: QueryIntent) -> Dict[str, Any]:
        """
        Convert QueryIntent to a filter specification.
        
        Args:
            intent: QueryIntent object
        
        Returns:
            Filter specification dictionary
        """
        spec = {}
        
        # Add time range
        if intent.time_range:
            spec['start_date'] = intent.time_range[0].isoformat()
            spec['end_date'] = intent.time_range[1].isoformat()
        
        # Add entities
        if intent.entities.get('agents'):
            spec['agents'] = intent.entities['agents']
        
        if intent.entities.get('campaigns'):
            spec['campaigns'] = intent.entities['campaigns']
        
        if intent.entities.get('call_types'):
            spec['call_types'] = intent.entities['call_types']
        
        if intent.entities.get('outcomes'):
            spec['outcomes'] = intent.entities['outcomes']
        
        # Add filters
        spec.update(intent.filters)
        
        # Add action metadata
        spec['_action'] = intent.action
        spec['_aggregations'] = intent.aggregations
        spec['_confidence'] = intent.confidence
        
        return spec
    
    def generate_explanation(self, intent: QueryIntent) -> str:
        """
        Generate a human-readable explanation of the interpretation.
        
        Args:
            intent: QueryIntent object
        
        Returns:
            Explanation string
        """
        parts = []
        
        # Explain action
        action_explanations = {
            'search': 'Searching for',
            'filter': 'Filtering',
            'aggregate': 'Calculating statistics for',
            'compare': 'Comparing',
            'analyze': 'Analyzing'
        }
        parts.append(action_explanations.get(intent.action, 'Processing'))
        
        # Explain entities
        if intent.entities.get('call_types'):
            parts.append(f"call types: {', '.join(intent.entities['call_types'])}")
        
        if intent.entities.get('outcomes'):
            parts.append(f"outcomes: {', '.join(intent.entities['outcomes'])}")
        
        if intent.entities.get('agents'):
            parts.append(f"agents: {', '.join(intent.entities['agents'])}")
        
        # Explain time range
        if intent.time_range:
            start = intent.time_range[0].strftime('%Y-%m-%d')
            end = intent.time_range[1].strftime('%Y-%m-%d')
            if start == end:
                parts.append(f"on {start}")
            else:
                parts.append(f"from {start} to {end}")
        
        # Explain filters
        if intent.filters.get('min_duration'):
            parts.append(f"duration > {intent.filters['min_duration']}s")
        
        if intent.filters.get('max_duration'):
            parts.append(f"duration < {intent.filters['max_duration']}s")
        
        # Explain aggregations
        if intent.aggregations:
            parts.append(f"calculating: {', '.join(intent.aggregations)}")
        
        explanation = ' '.join(parts)
        
        # Add confidence note
        if intent.confidence < 0.5:
            explanation += " (low confidence interpretation)"
        
        return explanation
