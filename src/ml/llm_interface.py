"""
Local LLM Interface Module

Provides interface to local language models for Q&A,
summarization, and other NLP tasks.
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import requests

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OLLAMA = "ollama"
    NONE = "none"


@dataclass
class LLMResponse:
    """Container for LLM response"""
    text: str
    model: str
    provider: str
    tokens_used: int
    success: bool
    error: str | None = None


class LocalLLMInterface:
    """
    Interface for interacting with local language models.
    Currently supports Ollama, with easy extension for other providers.
    """

    def __init__(self, config: dict):
        """
        Initialize the LLM interface.
        
        Args:
            config: Configuration dictionary with LLM settings
        """
        self.provider = LLMProvider(config.get('provider', 'none'))
        self.model_name = config.get('model_name', 'llama3:instruct')
        self.endpoint = config.get('endpoint', 'http://localhost:11434')
        self.temperature = config.get('temperature', 0.2)
        self.max_tokens = config.get('max_tokens', 256)
        self.timeout = config.get('timeout_seconds', 30)
        self.prompts = config.get('prompts', {})

        # Test connection
        self.is_available = self._test_connection()

        logger.info(f"LocalLLMInterface initialized with provider: {self.provider}")

    def _test_connection(self) -> bool:
        """
        Test if the LLM service is available.
        
        Returns:
            True if service is available
        """
        if self.provider == LLMProvider.NONE:
            return False

        if self.provider == LLMProvider.OLLAMA:
            try:
                response = requests.get(
                    f"{self.endpoint}/api/tags",
                    timeout=2
                )
                return response.status_code == 200
            except:
                logger.warning("Ollama service not available")
                return False

        return False

    def generate(self,
                prompt: str,
                system_prompt: str | None = None,
                temperature: float | None = None,
                max_tokens: int | None = None) -> LLMResponse:
        """
        Generate text using the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
        
        Returns:
            LLMResponse with generated text
        """
        if not self.is_available:
            return LLMResponse(
                text="",
                model=self.model_name,
                provider=self.provider.value,
                tokens_used=0,
                success=False,
                error="LLM service not available"
            )

        if self.provider == LLMProvider.OLLAMA:
            return self._generate_ollama(
                prompt,
                system_prompt,
                temperature or self.temperature,
                max_tokens or self.max_tokens
            )

        return LLMResponse(
            text="",
            model=self.model_name,
            provider=self.provider.value,
            tokens_used=0,
            success=False,
            error="Unsupported provider"
        )

    def _generate_ollama(self,
                        prompt: str,
                        system_prompt: str | None,
                        temperature: float,
                        max_tokens: int) -> LLMResponse:
        """
        Generate text using Ollama.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Temperature setting
            max_tokens: Maximum tokens
        
        Returns:
            LLMResponse
        """
        try:
            # Combine system and user prompts
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            # Prepare request
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }

            # Make request
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                return LLMResponse(
                    text=data.get('response', '').strip(),
                    model=self.model_name,
                    provider=self.provider.value,
                    tokens_used=data.get('eval_count', 0),
                    success=True
                )
            else:
                return LLMResponse(
                    text="",
                    model=self.model_name,
                    provider=self.provider.value,
                    tokens_used=0,
                    success=False,
                    error=f"API error: {response.status_code}"
                )

        except requests.exceptions.Timeout:
            return LLMResponse(
                text="",
                model=self.model_name,
                provider=self.provider.value,
                tokens_used=0,
                success=False,
                error="Request timeout"
            )
        except Exception as e:
            logger.error(f"Error generating with Ollama: {e}")
            return LLMResponse(
                text="",
                model=self.model_name,
                provider=self.provider.value,
                tokens_used=0,
                success=False,
                error=str(e)
            )

    def answer_question(self,
                       question: str,
                       context: str,
                       max_tokens: int | None = None) -> str:
        """
        Answer a question based on provided context.
        
        Args:
            question: User question
            context: Relevant context
            max_tokens: Maximum response length
        
        Returns:
            Answer text
        """
        # Use template if available
        template = self.prompts.get('answer_question',
            "Context: {context}\n\nQuestion: {question}\n\nAnswer:")

        prompt = template.format(context=context, question=question)

        response = self.generate(
            prompt=prompt,
            system_prompt="You are a helpful assistant analyzing call transcripts. Answer based only on the provided context.",
            max_tokens=max_tokens or self.max_tokens
        )

        return response.text if response.success else "Unable to generate answer."

    def summarize(self,
                 text: str,
                 max_length: int = 100) -> str:
        """
        Summarize text.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length in tokens
        
        Returns:
            Summary text
        """
        template = self.prompts.get('summarize',
            "Summarize the following text in 2-3 sentences:\n\n{text}\n\nSummary:")

        prompt = template.format(text=text[:2000])  # Limit input length

        response = self.generate(
            prompt=prompt,
            system_prompt="You are a concise summarizer. Provide clear, brief summaries.",
            max_tokens=max_length
        )

        return response.text if response.success else "Unable to generate summary."

    def extract_insights(self,
                        transcript: str,
                        focus_areas: list[str] = None) -> dict[str, Any]:
        """
        Extract insights from a call transcript.
        
        Args:
            transcript: Call transcript
            focus_areas: Specific areas to focus on
        
        Returns:
            Dictionary of insights
        """
        if not focus_areas:
            focus_areas = ["sentiment", "key_topics", "action_items", "issues"]

        insights = {}

        for area in focus_areas:
            if area == "sentiment":
                insights['sentiment'] = self._analyze_sentiment(transcript)
            elif area == "key_topics":
                insights['key_topics'] = self._extract_topics(transcript)
            elif area == "action_items":
                insights['action_items'] = self._extract_action_items(transcript)
            elif area == "issues":
                insights['issues'] = self._identify_issues(transcript)

        return insights

    def _analyze_sentiment(self, text: str) -> str:
        """
        Analyze sentiment of text.
        
        Args:
            text: Text to analyze
        
        Returns:
            Sentiment analysis result
        """
        prompt = f"""
        Analyze the sentiment of this call transcript.
        Classify as: Positive, Negative, or Neutral.
        Provide a brief explanation.
        
        Transcript: {text[:1000]}
        
        Sentiment:"""

        response = self.generate(prompt, max_tokens=50)
        return response.text if response.success else "Unknown"

    def _extract_topics(self, text: str) -> list[str]:
        """
        Extract key topics from text.
        
        Args:
            text: Text to analyze
        
        Returns:
            List of topics
        """
        prompt = f"""
        List the top 3-5 key topics discussed in this call.
        Provide only the topic names, one per line.
        
        Transcript: {text[:1000]}
        
        Topics:"""

        response = self.generate(prompt, max_tokens=100)

        if response.success:
            # Parse topics from response
            topics = [
                line.strip().lstrip('- ').lstrip('• ')
                for line in response.text.split('\n')
                if line.strip()
            ]
            return topics[:5]

        return []

    def _extract_action_items(self, text: str) -> list[str]:
        """
        Extract action items from text.
        
        Args:
            text: Text to analyze
        
        Returns:
            List of action items
        """
        prompt = f"""
        Extract any action items or follow-up tasks from this call.
        List them clearly, one per line.
        
        Transcript: {text[:1000]}
        
        Action Items:"""

        response = self.generate(prompt, max_tokens=150)

        if response.success:
            items = [
                line.strip().lstrip('- ').lstrip('• ')
                for line in response.text.split('\n')
                if line.strip()
            ]
            return items

        return []

    def _identify_issues(self, text: str) -> list[str]:
        """
        Identify issues mentioned in text.
        
        Args:
            text: Text to analyze
        
        Returns:
            List of issues
        """
        prompt = f"""
        Identify any problems or issues mentioned in this call.
        List them briefly, one per line.
        
        Transcript: {text[:1000]}
        
        Issues:"""

        response = self.generate(prompt, max_tokens=150)

        if response.success:
            issues = [
                line.strip().lstrip('- ').lstrip('• ')
                for line in response.text.split('\n')
                if line.strip()
            ]
            return issues

        return []

    def compare_calls(self,
                     call1: dict[str, Any],
                     call2: dict[str, Any]) -> str:
        """
        Compare two calls and highlight differences.
        
        Args:
            call1: First call data
            call2: Second call data
        
        Returns:
            Comparison summary
        """
        prompt = f"""
        Compare these two calls and highlight key differences:
        
        Call 1:
        - Type: {call1.get('call_type', 'Unknown')}
        - Outcome: {call1.get('outcome', 'Unknown')}
        - Duration: {call1.get('duration_seconds', 0)}s
        - Summary: {call1.get('transcript', '')[:200]}
        
        Call 2:
        - Type: {call2.get('call_type', 'Unknown')}
        - Outcome: {call2.get('outcome', 'Unknown')}
        - Duration: {call2.get('duration_seconds', 0)}s
        - Summary: {call2.get('transcript', '')[:200]}
        
        Comparison:"""

        response = self.generate(prompt, max_tokens=200)

        return response.text if response.success else "Unable to compare calls."

    def generate_report_section(self,
                               data: dict[str, Any],
                               section_type: str) -> str:
        """
        Generate a report section based on data.
        
        Args:
            data: Data for the report section
            section_type: Type of section to generate
        
        Returns:
            Generated report text
        """
        if section_type == "executive_summary":
            prompt = f"""
            Write an executive summary based on this data:
            - Total calls: {data.get('total_calls', 0)}
            - Connection rate: {data.get('connection_rate', 0)}%
            - Top issues: {', '.join(data.get('top_issues', [])[:3])}
            - Average duration: {data.get('avg_duration', 0)}s
            
            Executive Summary:"""

        elif section_type == "recommendations":
            prompt = f"""
            Based on this call analysis data, provide 3-5 recommendations:
            - Low connection rate areas: {data.get('low_connection_areas', [])}
            - Common complaints: {data.get('common_complaints', [])}
            - Peak problem times: {data.get('peak_problem_times', [])}
            
            Recommendations:"""

        else:
            prompt = f"Generate a {section_type} section based on: {json.dumps(data, indent=2)[:500]}"

        response = self.generate(prompt, max_tokens=300)

        return response.text if response.success else f"Unable to generate {section_type}."

    def check_availability(self) -> bool:
        """
        Check if the LLM service is currently available.
        
        Returns:
            True if available
        """
        self.is_available = self._test_connection()
        return self.is_available

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Model information dictionary
        """
        info = {
            'provider': self.provider.value,
            'model': self.model_name,
            'available': self.is_available,
            'endpoint': self.endpoint,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }

        if self.is_available and self.provider == LLMProvider.OLLAMA:
            try:
                response = requests.get(f"{self.endpoint}/api/tags", timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    models = [m['name'] for m in data.get('models', [])]
                    info['available_models'] = models
                    info['model_loaded'] = self.model_name in models
            except:
                pass

        return info
