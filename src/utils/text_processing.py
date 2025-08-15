"""
Text Processing Utilities

Common text processing functions used throughout the application.
"""

import re
import string
from typing import List, Dict, Any, Optional, Tuple
import unicodedata
from collections import Counter


def clean_text(text: str, 
               remove_punctuation: bool = False,
               lowercase: bool = False,
               remove_numbers: bool = False) -> str:
    """
    Clean text with various options.
    
    Args:
        text: Input text
        remove_punctuation: Whether to remove punctuation
        lowercase: Whether to convert to lowercase
        remove_numbers: Whether to remove numbers
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Normalize unicode
    text = unicodedata.normalize('NFKD', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    if lowercase:
        text = text.lower()
    
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    return text.strip()


def extract_keywords(text: str, 
                    top_k: int = 10,
                    min_length: int = 3) -> List[str]:
    """
    Extract keywords from text using simple frequency analysis.
    
    Args:
        text: Input text
        top_k: Number of keywords to extract
        min_length: Minimum keyword length
    
    Returns:
        List of keywords
    """
    # Common English stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'under', 'along',
        'following', 'behind', 'beyond', 'within', 'without', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
        'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'must', 'can', 'shall', 'it', 'its', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'we', 'they', 'them', 'their', 'what',
        'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
        'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just'
    }
    
    # Clean and tokenize
    text = clean_text(text, remove_punctuation=True, lowercase=True)
    words = text.split()
    
    # Filter words
    filtered_words = [
        word for word in words
        if len(word) >= min_length and word not in stop_words
    ]
    
    # Count frequencies
    word_counts = Counter(filtered_words)
    
    # Get top keywords
    keywords = [word for word, _ in word_counts.most_common(top_k)]
    
    return keywords


def tokenize(text: str, method: str = 'word') -> List[str]:
    """
    Tokenize text using specified method.
    
    Args:
        text: Input text
        method: Tokenization method ('word', 'sentence', 'paragraph')
    
    Returns:
        List of tokens
    """
    if not text:
        return []
    
    if method == 'word':
        # Simple word tokenization
        return text.split()
    
    elif method == 'sentence':
        # Sentence tokenization
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    elif method == 'paragraph':
        # Paragraph tokenization
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]
    
    else:
        raise ValueError(f"Unknown tokenization method: {method}")


def calculate_text_statistics(text: str) -> Dict[str, Any]:
    """
    Calculate various statistics about text.
    
    Args:
        text: Input text
    
    Returns:
        Dictionary of statistics
    """
    if not text:
        return {
            'character_count': 0,
            'word_count': 0,
            'sentence_count': 0,
            'average_word_length': 0,
            'average_sentence_length': 0,
            'unique_words': 0,
            'lexical_diversity': 0
        }
    
    # Basic counts
    char_count = len(text)
    words = tokenize(text, 'word')
    word_count = len(words)
    sentences = tokenize(text, 'sentence')
    sentence_count = len(sentences)
    
    # Word statistics
    avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0
    unique_words = len(set(w.lower() for w in words))
    lexical_diversity = unique_words / word_count if word_count > 0 else 0
    
    # Sentence statistics
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    
    return {
        'character_count': char_count,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'average_word_length': round(avg_word_length, 2),
        'average_sentence_length': round(avg_sentence_length, 2),
        'unique_words': unique_words,
        'lexical_diversity': round(lexical_diversity, 3)
    }


def find_patterns(text: str, patterns: List[str]) -> Dict[str, List[str]]:
    """
    Find regex patterns in text.
    
    Args:
        text: Input text
        patterns: List of regex patterns
    
    Returns:
        Dictionary mapping patterns to matches
    """
    results = {}
    
    for pattern in patterns:
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
            matches = compiled.findall(text)
            if matches:
                results[pattern] = matches
        except re.error:
            # Invalid regex pattern
            continue
    
    return results


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract simple entities from text.
    
    Args:
        text: Input text
    
    Returns:
        Dictionary of entity types to values
    """
    entities = {
        'emails': [],
        'phone_numbers': [],
        'urls': [],
        'dates': [],
        'times': [],
        'money': []
    }
    
    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    entities['emails'] = re.findall(email_pattern, text)
    
    # Phone number pattern (simple US format)
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    entities['phone_numbers'] = re.findall(phone_pattern, text)
    
    # URL pattern
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    entities['urls'] = re.findall(url_pattern, text)
    
    # Date pattern (MM/DD/YYYY or MM-DD-YYYY)
    date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
    entities['dates'] = re.findall(date_pattern, text)
    
    # Time pattern (HH:MM AM/PM)
    time_pattern = r'\b\d{1,2}:\d{2}\s?(?:AM|PM|am|pm)?\b'
    entities['times'] = re.findall(time_pattern, text)
    
    # Money pattern
    money_pattern = r'\$\d+(?:,\d{3})*(?:\.\d{2})?'
    entities['money'] = re.findall(money_pattern, text)
    
    return entities


def truncate_text(text: str, 
                 max_length: int,
                 suffix: str = '...') -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add when truncated
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Account for suffix length
    truncate_at = max_length - len(suffix)
    
    # Try to break at word boundary
    if ' ' in text[:truncate_at]:
        truncate_at = text[:truncate_at].rfind(' ')
    
    return text[:truncate_at] + suffix


def highlight_text(text: str, 
                  terms: List[str],
                  tag: str = 'mark') -> str:
    """
    Highlight terms in text with HTML tags.
    
    Args:
        text: Input text
        terms: Terms to highlight
        tag: HTML tag to use for highlighting
    
    Returns:
        Text with HTML highlighting
    """
    if not text or not terms:
        return text
    
    # Sort terms by length (longest first) to avoid nested highlights
    sorted_terms = sorted(terms, key=len, reverse=True)
    
    for term in sorted_terms:
        # Case-insensitive replacement
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        text = pattern.sub(f'<{tag}>\\g<0></{tag}>', text)
    
    return text


def calculate_similarity(text1: str, text2: str, method: str = 'jaccard') -> float:
    """
    Calculate similarity between two texts.
    
    Args:
        text1: First text
        text2: Second text
        method: Similarity method ('jaccard', 'cosine', 'levenshtein')
    
    Returns:
        Similarity score (0-1)
    """
    if not text1 or not text2:
        return 0.0
    
    if method == 'jaccard':
        # Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    elif method == 'cosine':
        # Simple cosine similarity based on word frequency
        from collections import Counter
        import math
        
        words1 = Counter(text1.lower().split())
        words2 = Counter(text2.lower().split())
        
        # Get all unique words
        all_words = set(words1.keys()) | set(words2.keys())
        
        # Create vectors
        vec1 = [words1.get(word, 0) for word in all_words]
        vec2 = [words2.get(word, 0) for word in all_words]
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    elif method == 'levenshtein':
        # Normalized Levenshtein distance
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = levenshtein_distance(text1, text2)
        max_length = max(len(text1), len(text2))
        
        return 1 - (distance / max_length) if max_length > 0 else 1.0
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def mask_pii(text: str) -> str:
    """
    Mask personally identifiable information in text.
    
    Args:
        text: Input text
    
    Returns:
        Text with PII masked
    """
    if not text:
        return text
    
    # Mask email addresses
    text = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        '[EMAIL]',
        text
    )
    
    # Mask phone numbers
    text = re.sub(
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        '[PHONE]',
        text
    )
    
    # Mask SSN-like patterns
    text = re.sub(
        r'\b\d{3}-\d{2}-\d{4}\b',
        '[SSN]',
        text
    )
    
    # Mask credit card-like patterns
    text = re.sub(
        r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
        '[CREDIT_CARD]',
        text
    )
    
    return text