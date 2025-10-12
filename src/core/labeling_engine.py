"""
Call Labeling Engine

Applies rule-based labeling to calls for categorization
and outcome determination based on configurable rules.
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from fuzzywuzzy import fuzz

logger = logging.getLogger(__name__)


@dataclass
class LabelingResult:
    """Container for labeling results"""

    connection_status: str
    call_type: str
    outcome: str
    confidence_scores: dict[str, float]
    matched_keywords: list[str]


class LabelingEngine:
    """
    Engine for applying labeling rules to call transcripts.
    Uses keyword matching, pattern recognition, and scoring
    to categorize calls and determine outcomes.
    """

    def __init__(self, rules_config: dict):
        """
        Initialize the labeling engine with rules configuration.

        Args:
            rules_config: Dictionary containing labeling rules
        """
        self.rules = rules_config
        self.connection_rules = rules_config.get("connection", {})
        self.call_type_rules = rules_config.get("call_types", {})
        self.outcome_rules = rules_config.get("outcomes", {})
        self.scoring_config = rules_config.get("scoring", {})
        self.priorities = rules_config.get("priorities", {})
        self.custom_patterns = self._compile_custom_patterns()

        # Configure fuzzy matching
        self.fuzzy_enabled = self.scoring_config.get("fuzzy_matching", True)
        self.fuzzy_threshold = self.scoring_config.get("fuzzy_threshold", 0.85) * 100

        logger.info("LabelingEngine initialized with rules")

    def _compile_custom_patterns(self) -> dict[str, re.Pattern]:
        """
        Compile custom regex patterns from configuration.

        Returns:
            Dictionary of compiled regex patterns
        """
        patterns = {}
        custom_rules = self.rules.get("custom_rules", {})

        for name, pattern_str in custom_rules.items():
            try:
                patterns[name] = re.compile(pattern_str, re.IGNORECASE)
                logger.debug(f"Compiled pattern: {name}")
            except re.error as e:
                logger.error(f"Invalid regex pattern for {name}: {e}")

        return patterns

    def label_call(
        self, transcript: str, duration_seconds: float, metadata: dict[str, Any] | None = None
    ) -> LabelingResult:
        """
        Apply all labeling rules to a call.

        Args:
            transcript: Call transcript text
            duration_seconds: Call duration in seconds
            metadata: Optional call metadata

        Returns:
            LabelingResult with all labels and scores
        """
        # Initialize tracking variables
        matched_keywords = []
        confidence_scores = {}

        # Determine connection status
        connection_status, conn_confidence, conn_keywords = self._determine_connection_status(
            transcript, duration_seconds
        )
        matched_keywords.extend(conn_keywords)
        confidence_scores["connection"] = conn_confidence

        # Determine call type
        call_type, type_confidence, type_keywords = self._determine_call_type(transcript)
        matched_keywords.extend(type_keywords)
        confidence_scores["call_type"] = type_confidence

        # Determine outcome
        outcome, outcome_confidence, outcome_keywords = self._determine_outcome(transcript)
        matched_keywords.extend(outcome_keywords)
        confidence_scores["outcome"] = outcome_confidence

        # Apply custom pattern overrides if matched
        call_type, outcome = self._apply_custom_patterns(transcript, call_type, outcome)

        return LabelingResult(
            connection_status=connection_status,
            call_type=call_type,
            outcome=outcome,
            confidence_scores=confidence_scores,
            matched_keywords=list(set(matched_keywords)),  # Remove duplicates
        )

    def _determine_connection_status(
        self, transcript: str, duration: float
    ) -> tuple[str, float, list[str]]:
        """
        Determine if the call was connected or disconnected.

        Args:
            transcript: Call transcript
            duration: Call duration in seconds

        Returns:
            Tuple of (status, confidence, matched_keywords)
        """
        confidence = 0.0
        matched_keywords = []

        # Check duration threshold
        min_duration = self.connection_rules.get("min_duration_seconds", 30)
        min_words = self.connection_rules.get("min_transcript_words", 40)

        # Check for disconnection keywords
        disconnection_keywords = self.connection_rules.get("disconnection_keywords", [])
        for keyword in disconnection_keywords:
            if self._keyword_match(transcript, keyword):
                matched_keywords.append(keyword)
                confidence = 0.9
                return "Disconnected", confidence, matched_keywords

        # Check duration and transcript length
        word_count = len(transcript.split())

        if duration < min_duration or word_count < min_words:
            confidence = 0.8
            return "Disconnected", confidence, []

        # Check for connection keywords
        connection_keywords = self.connection_rules.get("connection_keywords", [])
        for keyword in connection_keywords:
            if self._keyword_match(transcript, keyword):
                matched_keywords.append(keyword)
                confidence += 0.2

        if matched_keywords:
            confidence = min(1.0, confidence + 0.5)
            return "Connected", confidence, matched_keywords

        # Default to connected if duration and word count are sufficient
        if duration >= min_duration and word_count >= min_words:
            confidence = 0.7
            return "Connected", confidence, []

        return "Unknown", 0.3, []

    def _score_entities(
        self,
        transcript: str,
        rules: dict[str, list[str]],
        scorer: Callable[[str], float],
    ) -> tuple[dict[str, float], dict[str, list[str]]]:
        """Return score and keyword matches for each rule group."""
        scores: dict[str, float] = {}
        keyword_matches: dict[str, list[str]] = {}

        for label, keywords in rules.items():
            matches = [kw for kw in keywords if self._keyword_match(transcript, kw)]
            keyword_matches[label] = matches
            scores[label] = float(sum(scorer(keyword) for keyword in matches))

        return scores, keyword_matches

    def _resolve_best_label(self, scores: dict[str, float], priority: list[str]) -> str | None:
        """Resolve the highest-scoring label, honoring optional priority ordering."""
        if not scores:
            return None

        max_score = max(scores.values(), default=0.0)
        if max_score == 0:
            return None

        tied = [label for label, score in scores.items() if score == max_score]
        if len(tied) == 1:
            return tied[0]

        for candidate in priority:
            if candidate in tied:
                return candidate

        return tied[0] if tied else None

    def _calculate_confidence(
        self,
        scores: dict[str, float],
        label: str,
        rules: dict[str, list[str]],
    ) -> float:
        """Convert the winning label's score into a bounded confidence value."""
        total_keywords = sum(len(keywords) for keywords in rules.values())
        denominator = max(total_keywords * 0.1, 1)
        return min(1.0, scores.get(label, 0.0) / denominator)

    def _outcome_analysis_segment(self, transcript: str) -> str:
        """Return the transcript segment used for outcome bonus scoring."""
        if len(transcript) <= 500:
            return transcript
        start_index = int(len(transcript) * 0.7)
        return transcript[start_index:]

    def _determine_call_type(self, transcript: str) -> tuple[str, float, list[str]]:
        """
        Determine the call type based on transcript content.

        Args:
            transcript: Call transcript

        Returns:
            Tuple of (call_type, confidence, matched_keywords)
        """
        keyword_weight = float(self.scoring_config.get("keyword_weight", 1.0))
        phrase_weight = float(self.scoring_config.get("phrase_weight", 2.0))

        def call_type_score(keyword: str) -> float:
            return phrase_weight if len(keyword.split()) > 1 else keyword_weight

        scores, keyword_matches = self._score_entities(
            transcript=transcript,
            rules=self.call_type_rules,
            scorer=call_type_score,
        )
        best_type = self._resolve_best_label(scores, self.priorities.get("call_type_priority", []))

        if not best_type:
            return "Unknown", 0.0, []

        confidence = self._calculate_confidence(scores, best_type, self.call_type_rules)

        # Apply minimum confidence threshold
        min_confidence = self.scoring_config.get("min_confidence_score", 0.6)
        if confidence < min_confidence:
            return "Unknown", confidence, []

        return best_type.replace("_", "/").title(), confidence, keyword_matches[best_type]

    def _determine_outcome(self, transcript: str) -> tuple[str, float, list[str]]:
        """
        Determine the call outcome based on transcript content.

        Args:
            transcript: Call transcript

        Returns:
            Tuple of (outcome, confidence, matched_keywords)
        """
        last_portion = self._outcome_analysis_segment(transcript)
        keyword_weight = float(self.scoring_config.get("keyword_weight", 1.0))

        def outcome_score(keyword: str) -> float:
            bonus = 0.5 if self._keyword_match(last_portion, keyword) else 0.0
            return keyword_weight + bonus

        scores, keyword_matches = self._score_entities(
            transcript=transcript,
            rules=self.outcome_rules,
            scorer=outcome_score,
        )
        best_outcome = self._resolve_best_label(scores, self.priorities.get("outcome_priority", []))

        if not best_outcome:
            return "Unknown", 0.0, []

        confidence = self._calculate_confidence(scores, best_outcome, self.outcome_rules)

        # Apply minimum confidence threshold
        min_confidence = self.scoring_config.get("min_confidence_score", 0.6)
        if confidence < min_confidence:
            return "Unknown", confidence, []

        # Format outcome name
        outcome_formatted = best_outcome.replace("_", "-").title()

        return outcome_formatted, confidence, keyword_matches[best_outcome]

    def _keyword_match(self, text: str, keyword: str) -> bool:
        """
        Check if a keyword matches in the text.
        Supports exact and fuzzy matching.

        Args:
            text: Text to search in
            keyword: Keyword to search for

        Returns:
            True if keyword is found
        """
        text_lower = text.lower()
        keyword_lower = keyword.lower()

        # Exact match
        if keyword_lower in text_lower:
            return True

        # Fuzzy match if enabled
        if self.fuzzy_enabled:
            # Split text into chunks for fuzzy matching
            words = text_lower.split()
            keyword_words = keyword_lower.split()

            # For single word keywords
            if len(keyword_words) == 1:
                for word in words:
                    if fuzz.ratio(word, keyword_lower) >= self.fuzzy_threshold:
                        return True
            else:
                # For phrase keywords, check substring fuzzy match
                if fuzz.partial_ratio(keyword_lower, text_lower) >= self.fuzzy_threshold:
                    return True

        return False

    def _apply_custom_patterns(
        self,
        transcript: str,
        call_type: str,
        outcome: str,
    ) -> tuple[str, str]:
        """
        Apply custom regex patterns to override labels if matched.

        Args:
            transcript: Call transcript
            call_type: Current call type
            outcome: Current outcome

        Returns:
            Tuple of (call_type, outcome) with overrides applied
        """
        # Check for urgent complaint pattern
        if "urgent_complaint" in self.custom_patterns and self.custom_patterns[
            "urgent_complaint"
        ].search(transcript):
            call_type = "Complaint"
            logger.debug("Applied urgent_complaint pattern override")

        # Check for technical escalation pattern
        if "technical_escalation" in self.custom_patterns and self.custom_patterns[
            "technical_escalation"
        ].search(transcript):
            outcome = "Callback"  # Technical escalations often need callbacks
            logger.debug("Applied technical_escalation pattern override")

        return call_type, outcome

    def label_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply labeling to an entire DataFrame of calls.

        Args:
            df: DataFrame with transcript and duration columns

        Returns:
            DataFrame with added label columns
        """
        # Ensure required columns exist
        if "transcript" not in df.columns or "duration_seconds" not in df.columns:
            raise ValueError("DataFrame must have 'transcript' and 'duration_seconds' columns")

        # Initialize new columns
        df["connection_status"] = ""
        df["call_type"] = ""
        df["outcome"] = ""
        df["labeling_confidence"] = 0.0
        df["matched_keywords"] = ""

        # Process each row
        for idx, row in df.iterrows():
            try:
                result = self.label_call(
                    transcript=str(row["transcript"]),
                    duration_seconds=float(row["duration_seconds"]),
                    metadata=row.to_dict(),
                )

                df.at[idx, "connection_status"] = result.connection_status
                df.at[idx, "call_type"] = result.call_type
                df.at[idx, "outcome"] = result.outcome

                # Average confidence across all labels
                avg_confidence = sum(result.confidence_scores.values()) / len(
                    result.confidence_scores
                )
                df.at[idx, "labeling_confidence"] = avg_confidence

                # Store matched keywords as comma-separated string
                df.at[idx, "matched_keywords"] = ", ".join(
                    result.matched_keywords[:5]
                )  # Limit to 5

            except Exception as e:
                logger.error(f"Error labeling row {idx}: {e}")
                df.at[idx, "connection_status"] = "Unknown"
                df.at[idx, "call_type"] = "Unknown"
                df.at[idx, "outcome"] = "Unknown"
                df.at[idx, "labeling_confidence"] = 0.0

        logger.info(f"Labeled {len(df)} calls")
        return df

    def get_labeling_statistics(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Get statistics about the labeling results.

        Args:
            df: DataFrame with label columns

        Returns:
            Dictionary with labeling statistics
        """
        stats = {
            "total_calls": len(df),
            "connection_distribution": {},
            "type_distribution": {},
            "outcome_distribution": {},
            "average_confidence": 0.0,
            "low_confidence_count": 0,
        }

        # Connection distribution
        if "connection_status" in df.columns:
            stats["connection_distribution"] = df["connection_status"].value_counts().to_dict()

        # Type distribution
        if "call_type" in df.columns:
            stats["type_distribution"] = df["call_type"].value_counts().to_dict()

        # Outcome distribution
        if "outcome" in df.columns:
            stats["outcome_distribution"] = df["outcome"].value_counts().to_dict()

        # Confidence statistics
        if "labeling_confidence" in df.columns:
            stats["average_confidence"] = float(df["labeling_confidence"].mean())
            stats["low_confidence_count"] = int((df["labeling_confidence"] < 0.6).sum())

        return stats
