"""
Q&A Interface Page Module for Call Analytics System

This module implements the question-answering interface that allows users
to interact with the system using natural language queries, leveraging
local LLMs for intelligent responses about call data and analytics.
"""

import logging
import sys
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.analysis.query_interpreter import QueryInterpreter
from src.analysis.semantic_search import SemanticSearchEngine
from src.core.storage_manager import StorageManager
from src.ui.components import MetricsGrid, MetricValue

# Configure module logger
logger = logging.getLogger(__name__)


class QAInterface:
    """
    Natural language Q&A interface for interacting with call analytics
    using local LLMs and semantic search capabilities.
    """

    def __init__(self, storage_manager: StorageManager, vector_store=None, llm_client=None):
        """
        Initialize Q&A interface with required components.

        Args:
            storage_manager: Storage manager instance
            vector_store: Optional vector store for semantic search
            llm_client: Optional LLM client for natural language processing
        """
        self.storage_manager = storage_manager
        self.vector_store = vector_store
        self.llm_client = llm_client

        # Initialize components if available
        if vector_store:
            self.search_engine = SemanticSearchEngine(vector_store)
        else:
            self.search_engine = None

        self.query_interpreter = QueryInterpreter()

        # Initialize conversation history
        if 'qa_history' not in st.session_state:
            st.session_state.qa_history = []

    def render(self) -> None:
        """
        Render the Q&A interface page with chat-like interaction.
        """
        try:
            # Page header
            st.title("💬 Q&A Assistant")
            st.markdown("Ask questions about your call data in natural language")

            # Check prerequisites
            if not self._check_prerequisites():
                return

            # Create main layout
            col1, col2 = st.columns([2, 1])

            with col1:
                self._render_chat_interface()

            with col2:
                self._render_context_panel()

            # Render suggested questions
            self._render_suggested_questions()

        except Exception as e:
            logger.error(f"Error rendering Q&A interface: {e}")
            st.error(f"Failed to load Q&A interface: {str(e)}")

    def _check_prerequisites(self) -> bool:
        """
        Check if required components are available.

        Returns:
            True if prerequisites met, False otherwise
        """
        warnings = []

        if not self.search_engine:
            warnings.append("• Vector database not configured - semantic search unavailable")

        if not self.llm_client:
            warnings.append("• LLM client not configured - using rule-based responses")

        data_count = self.storage_manager.get_record_count()
        if data_count == 0:
            warnings.append("• No call data available - please upload data first")

        if warnings:
            st.warning("**Setup Required:**\n" + "\n".join(warnings))

            with st.expander("Setup Instructions"):
                st.markdown("""
                ### To enable full Q&A functionality:

                1. **Vector Database**: Build the vector index using the Analysis page
                2. **LLM Client**: Configure Ollama or another local LLM in settings
                3. **Call Data**: Upload call records or audio files

                Once configured, you can:
                - Ask natural language questions about your data
                - Get AI-powered insights and summaries
                - Explore trends and patterns conversationally
                """)

            # Continue with limited functionality
            return data_count > 0

        return True

    def _render_chat_interface(self) -> None:
        """
        Render the main chat interface for Q&A interaction.
        """
        st.subheader("Chat with Your Data")

        # Display conversation history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.qa_history:
                self._render_message(message)

        # Input area
        with st.form("qa_input_form", clear_on_submit=True):
            col1, col2 = st.columns([5, 1])

            with col1:
                user_query = st.text_input(
                    "Ask a question...",
                    placeholder="e.g., What was the average call duration last week?",
                    label_visibility="collapsed"
                )

            with col2:
                submit_button = st.form_submit_button(
                    "Send",
                    type="primary",
                    use_container_width=True
                )

        # Process query
        if submit_button and user_query:
            self._process_query(user_query)

    def _render_message(self, message: dict[str, Any]) -> None:
        """
        Render a single message in the chat interface.

        Args:
            message: Message dictionary with role, content, and metadata
        """
        if message['role'] == 'user':
            with st.chat_message("user"):
                st.write(message['content'])
        else:
            with st.chat_message("assistant"):
                st.write(message['content'])

                # Display any associated data
                if 'data' in message and message['data']:
                    self._render_response_data(message['data'])

                # Display sources if available
                if 'sources' in message and message['sources']:
                    with st.expander("📚 Sources"):
                        for source in message['sources']:
                            st.caption(f"• {source}")

    def _render_response_data(self, data: Any) -> None:
        """
        Render data associated with a response (charts, tables, metrics).

        Args:
            data: Response data to render
        """
        if isinstance(data, pd.DataFrame):
            st.dataframe(data, use_container_width=True)
        elif isinstance(data, dict):
            if 'metrics' in data:
                # Render metrics
                metrics = [
                    MetricValue(
                        value=v,
                        label=k.replace('_', ' ').title()
                    )
                    for k, v in data['metrics'].items()
                ]
                MetricsGrid.render(metrics, st.container())
            elif 'chart' in data:
                # Render chart
                st.plotly_chart(data['chart'], use_container_width=True)
        elif isinstance(data, list):
            # Render list items
            for item in data:
                st.write(f"• {item}")

    def _render_context_panel(self) -> None:
        """
        Render the context panel with system status and capabilities.
        """
        st.subheader("System Context")

        # Data summary
        with st.expander("📊 Data Overview", expanded=True):
            try:
                total_calls = self.storage_manager.get_record_count()
                date_range = self.storage_manager.get_date_range()

                st.metric("Total Calls", f"{total_calls:,}")

                if date_range:
                    st.caption("**Date Range:**")
                    st.caption(f"{date_range[0]} to {date_range[1]}")

                # Show available fields
                st.caption("**Available Fields:**")
                fields = self.storage_manager.get_available_fields()
                if fields:
                    st.caption(", ".join(fields[:10]))
                    if len(fields) > 10:
                        st.caption(f"... and {len(fields) - 10} more")

            except Exception as e:
                logger.error(f"Error loading data overview: {e}")
                st.error("Failed to load data overview")

        # Capabilities
        with st.expander("🤖 Capabilities"):
            capabilities = self._get_system_capabilities()
            for cap in capabilities:
                st.caption(f"✓ {cap}")

        # Settings
        with st.expander("⚙️ Settings"):
            st.checkbox(
                "Show data sources",
                value=True,
                key="qa_show_sources"
            )

            st.checkbox(
                "Include visualizations",
                value=True,
                key="qa_include_viz"
            )

            st.slider(
                "Response detail level",
                min_value=1,
                max_value=5,
                value=3,
                key="qa_detail_level"
            )

    def _render_suggested_questions(self) -> None:
        """
        Render suggested questions based on available data.
        """
        st.divider()
        st.subheader("💡 Suggested Questions")

        suggestions = self._generate_suggestions()

        cols = st.columns(2)
        for idx, suggestion in enumerate(suggestions):
            col_idx = idx % 2
            with cols[col_idx]:
                if st.button(
                    suggestion,
                    key=f"suggestion_{idx}",
                    use_container_width=True
                ):
                    self._process_query(suggestion)

    def _process_query(self, query: str) -> None:
        """
        Process a user query and generate response.

        Args:
            query: User's natural language query
        """
        # Add user message to history
        st.session_state.qa_history.append({
            'role': 'user',
            'content': query,
            'timestamp': datetime.now()
        })

        # Show processing indicator
        with st.spinner("Thinking..."):
            try:
                # Interpret query intent
                intent_obj = self.query_interpreter.interpret(query)

                # Convert intent dataclass to dictionary for downstream handlers
                if hasattr(intent_obj, '__dataclass_fields__'):
                    intent = asdict(intent_obj)
                else:
                    intent = dict(intent_obj)

                action = intent.get('action', 'general') or 'general'

                # Map action to response type
                type_map = {
                    'aggregate': 'metric',
                    'filter': 'metric',
                    'search': 'search',
                    'compare': 'comparison',
                    'analyze': 'analysis'
                }
                intent['type'] = type_map.get(action, 'general')

                # Infer metric type from aggregations if available
                if intent['type'] == 'metric':
                    aggregations = intent.get('aggregations', []) or []
                    metric = None
                    if any(agg in aggregations for agg in ['count', 'sum']):
                        metric = 'total_calls'
                    if any(agg in aggregations for agg in ['average', 'avg']):
                        metric = 'average_duration'
                    intent['metric'] = metric or intent.get('metric', 'general')

                # Generate response based on intent
                response = self._generate_response(query, intent)

                # Add assistant response to history
                st.session_state.qa_history.append(response)

                # Rerun to update chat display
                st.rerun()

            except Exception as e:
                logger.error(f"Error processing query: {e}")
                error_response = {
                    'role': 'assistant',
                    'content': (
                        "I encountered an error processing your question: "
                        f"{e}. Please try rephrasing or ask a different question."
                    ),
                    'timestamp': datetime.now(),
                }
                st.session_state.qa_history.append(error_response)
                st.rerun()

    def _generate_response(self, query: str, intent: dict[str, Any]) -> dict[str, Any]:
        """
        Generate response based on query and intent.

        Args:
            query: Original user query
            intent: Interpreted intent from query

        Returns:
            Response dictionary with content and data
        """
        response = {
            'role': 'assistant',
            'timestamp': datetime.now(),
            'sources': []
        }

        try:
            # Determine response type based on intent
            intent_type = intent.get('type', 'general')

            if intent_type == 'metric':
                # Handle metric queries
                response = self._handle_metric_query(query, intent, response)

            elif intent_type == 'search':
                # Handle search queries
                response = self._handle_search_query(query, intent, response)

            elif intent_type == 'analysis':
                # Handle analysis queries
                response = self._handle_analysis_query(query, intent, response)

            elif intent_type == 'comparison':
                # Handle comparison queries
                response = self._handle_comparison_query(query, intent, response)

            else:
                # Handle general queries
                response = self._handle_general_query(query, intent, response)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            response['content'] = (
                "I had trouble understanding your question. Could you please rephrase it?"
            )

        return response

    def _handle_metric_query(
        self,
        query: str,
        intent: dict[str, Any],
        response: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Handle queries about metrics and statistics.

        Args:
            query: User query
            intent: Query intent
            response: Response dictionary to populate

        Returns:
            Updated response dictionary
        """
        # Extract time range from intent
        time_range = intent.get('time_range')
        metric_type = intent.get('metric', 'general')

        # Load appropriate data
        if isinstance(time_range, (list, tuple)) and len(time_range) == 2:
            start, end = time_range
            if isinstance(start, str):
                start = pd.to_datetime(start)
            if isinstance(end, str):
                end = pd.to_datetime(end)

            if isinstance(start, pd.Timestamp):
                start = start.to_pydatetime()
            if isinstance(end, pd.Timestamp):
                end = end.to_pydatetime()

            if isinstance(start, datetime):
                start = start.date()
            if isinstance(end, datetime):
                end = end.date()

            data = self.storage_manager.load_call_records(start, end)
        else:
            data = self._load_data_for_timerange(time_range or 'all')

        if data.empty:
            response['content'] = "No data available for the specified time range."
            return response

        # Calculate requested metrics
        metrics = {}

        if metric_type in ['duration', 'average_duration', 'avg_duration']:
            avg_duration = data['duration'].mean() / 60 if 'duration' in data.columns else 0
            metrics['Average Duration'] = f"{avg_duration:.1f} minutes"
            response['content'] = f"The average call duration is **{avg_duration:.1f} minutes**."

        elif metric_type in ['total_calls', 'count', 'volume']:
            total = len(data)
            metrics['Total Calls'] = total
            response['content'] = f"There were **{total:,} calls** in the specified period."

        elif metric_type in ['connection_rate', 'success_rate']:
            if 'outcome' in data.columns:
                rate = (data['outcome'] == 'connected').mean() * 100
                metrics['Connection Rate'] = f"{rate:.1f}%"
                response['content'] = f"The connection rate is **{rate:.1f}%**."

        elif metric_type in ['revenue', 'total_revenue']:
            if 'revenue' in data.columns:
                total_revenue = data['revenue'].sum()
                metrics['Total Revenue'] = f"${total_revenue:,.2f}"
                response['content'] = f"The total revenue is **${total_revenue:,.2f}**."

        else:
            # General metrics summary
            metrics = {'Total Calls': len(data)}

            if 'duration' in data.columns:
                avg_minutes = data['duration'].mean() / 60
                metrics['Avg Duration'] = f"{avg_minutes:.1f} min"
            else:
                metrics['Avg Duration'] = 'N/A'

            if 'outcome' in data.columns:
                connection_rate = (data['outcome'] == 'connected').mean() * 100
                metrics['Connection Rate'] = f"{connection_rate:.1f}%"
            else:
                metrics['Connection Rate'] = 'N/A'

            if 'revenue' in data.columns:
                metrics['Total Revenue'] = f"${data['revenue'].sum():,.2f}"
            else:
                metrics['Total Revenue'] = 'N/A'

            response['content'] = "Here's a summary of the metrics:"

        # Add metrics to response data
        if metrics:
            response['data'] = {'metrics': metrics}

        # Add sources
        response['sources'] = [f"Analyzed {len(data)} call records"]

        return response

    def _handle_search_query(
        self,
        query: str,
        intent: dict[str, Any],
        response: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Handle search queries using semantic search.

        Args:
            query: User query
            intent: Query intent
            response: Response dictionary to populate

        Returns:
            Updated response dictionary
        """
        if not self.search_engine:
            response['content'] = (
                'Semantic search is not available. Please configure the vector database.'
            )
            return response

        # Perform semantic search
        search_terms = intent.get('search_terms', query)
        results = self.search_engine.search(
            query=search_terms,
            top_k=5,
            threshold=0.35
        )

        if results:
            response['content'] = f"I found {len(results)} calls matching your search:"

            # Create DataFrame from results
            results_df = pd.DataFrame(results)
            response['data'] = results_df[['call_id', 'timestamp', 'agent_id', 'score']]

            # Add sources
            response['sources'] = [f"Semantic search: {len(results)} matches"]
        else:
            response['content'] = "No calls found matching your search criteria."

        return response

    def _handle_analysis_query(
        self,
        query: str,
        intent: dict[str, Any],
        response: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Handle analysis queries requiring data processing.

        Args:
            query: User query
            intent: Query intent
            response: Response dictionary to populate

        Returns:
            Updated response dictionary
        """
        analysis_type = intent.get('analysis_type', 'general')

        # Load data
        data = self._load_data_for_timerange(intent.get('time_range', 'all'))

        if data.empty:
            response['content'] = "No data available for analysis."
            return response

        if analysis_type == 'trend':
            response['content'] = "Here's the trend analysis:"
            # Add trend visualization
            # response['data'] = {'chart': create_trend_chart(data)}

        elif analysis_type == 'distribution':
            response['content'] = "Here's the distribution analysis:"
            # Add distribution visualization

        elif analysis_type == 'performance':
            if 'agent_id' in data.columns:
                # Agent performance analysis
                agent_stats = data.groupby('agent_id').agg({
                    'call_id': 'count',
                    'duration': 'mean',
                    'revenue': 'sum' if 'revenue' in data.columns else lambda x: 0
                }).round(2)

                response['content'] = "Here's the agent performance analysis:"
                response['data'] = agent_stats
        else:
            response['content'] = "I'll help you analyze the data. Here's what I found:"
            # General analysis

        response['sources'] = [f"Analyzed {len(data)} records"]

        return response

    def _handle_comparison_query(
        self,
        query: str,
        intent: dict[str, Any],
        response: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Handle comparison queries between periods or segments.

        Args:
            query: User query
            intent: Query intent
            response: Response dictionary to populate

        Returns:
            Updated response dictionary
        """
        # Extract comparison parameters
        compare_what = intent.get('compare', 'periods')

        if compare_what == 'periods':
            # Period comparison
            response['content'] = "Here's the period comparison:"
            # Implement period comparison logic

        elif compare_what == 'agents':
            # Agent comparison
            data = self._load_data_for_timerange('all')
            if 'agent_id' in data.columns:
                agent_comparison = data.groupby('agent_id').agg({
                    'call_id': 'count',
                    'outcome': lambda x: (x == 'connected').mean() * 100
                }).round(2)

                response['content'] = "Here's the agent comparison:"
                response['data'] = agent_comparison

        elif compare_what == 'campaigns':
            # Campaign comparison
            data = self._load_data_for_timerange('all')
            if 'campaign' in data.columns:
                campaign_comparison = data.groupby('campaign').agg({
                    'call_id': 'count',
                    'revenue': 'sum' if 'revenue' in data.columns else lambda x: 0
                }).round(2)

                response['content'] = "Here's the campaign comparison:"
                response['data'] = campaign_comparison

        return response

    def _handle_general_query(
        self,
        query: str,
        intent: dict[str, Any],
        response: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Handle general queries that don't fit specific categories.

        Args:
            query: User query
            intent: Query intent
            response: Response dictionary to populate

        Returns:
            Updated response dictionary
        """
        # Use LLM if available
        if self.llm_client:
            # Generate response using LLM
            llm_response = self._generate_llm_response(query)
            response['content'] = llm_response
        else:
            # Fallback response
            response['content'] = (
                "I understand you're asking about the call data. While I can't provide a "
                "specific answer without LLM support, I can help you with:\n"
                "- Metrics and statistics\n"
                "- Searching for specific calls\n"
                "- Analyzing trends and patterns\n"
                "- Comparing different time periods\n\n"
                "Try asking questions like:\n"
                '- "What was the average call duration last week?"\n'
                '- "Show me the top performing agents"\n'
                '- "Find calls about billing issues"\n'
            )

        return response

    def _load_data_for_timerange(self, time_range: str) -> pd.DataFrame:
        """
        Load data for specified time range.

        Args:
            time_range: Time range specification

        Returns:
            Filtered DataFrame
        """
        try:
            if time_range == 'today':
                start_date = datetime.now().date()
                end_date = start_date
            elif time_range == 'yesterday':
                end_date = datetime.now().date() - timedelta(days=1)
                start_date = end_date
            elif time_range == 'last_week':
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=7)
            elif time_range == 'last_month':
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=30)
            else:
                # Load all data
                return self.storage_manager.load_all_records()

            return self.storage_manager.load_call_records(start_date, end_date)

        except Exception as e:
            logger.error(f"Error loading data for timerange {time_range}: {e}")
            return pd.DataFrame()

    def _generate_llm_response(self, query: str) -> str:
        """
        Generate response using LLM client.

        Args:
            query: User query

        Returns:
            LLM-generated response
        """
        if not self.llm_client:
            return "LLM client unavailable."

        try:
            system_prompt = (
                "You are a helpful call analytics assistant. Respond with clear, concise"
                " insights grounded in the provided call center data."
            )

            result = self.llm_client.generate(
                prompt=query,
                system_prompt=system_prompt
            )

            if result.success and result.text:
                return result.text

            error_msg = result.error or "LLM returned an empty response."
            logger.warning(f"LLM response issue: {error_msg}")
            return (
                "I couldn't generate an AI-powered answer right now."
                " Please verify the LLM service is running and try again."
            )

        except Exception as exc:
            logger.error(f"Error generating LLM response: {exc}")
            return (
                "Something went wrong while contacting the LLM service."
                " Please check the logs for details."
            )

    def _get_system_capabilities(self) -> list[str]:
        """
        Get list of system capabilities.

        Returns:
            List of capability descriptions
        """
        capabilities = [
            "Answer questions about call metrics",
            "Search call transcripts and notes",
            "Analyze trends and patterns",
            "Compare time periods",
            "Generate summaries and insights"
        ]

        if self.search_engine:
            capabilities.append("Semantic search in transcripts")

        if self.llm_client:
            capabilities.append("AI-powered natural language understanding")

        return capabilities

    def _generate_suggestions(self) -> list[str]:
        """
        Generate suggested questions based on available data.

        Returns:
            List of suggested questions
        """
        suggestions = []

        # Basic suggestions always available
        suggestions.extend([
            "What is the total number of calls?",
            "Show me today's call summary"
        ])

        # Add suggestions based on available fields
        fields = self.storage_manager.get_available_fields()

        if 'duration' in fields:
            suggestions.append("What was the average call duration?")

        if 'outcome' in fields:
            suggestions.append("What is the connection rate?")

        if 'agent_id' in fields:
            suggestions.append("Who are the top performing agents?")

        if 'revenue' in fields:
            suggestions.append("What is the total revenue generated?")

        if 'campaign' in fields:
            suggestions.append("Compare campaign performance")

        return suggestions[:6]  # Limit to 6 suggestions


def render_qa_interface(storage_manager: StorageManager,
                        vector_store=None,
                        llm_client=None) -> None:
    """
    Main entry point for rendering the Q&A interface.

    Args:
        storage_manager: Storage manager instance
        vector_store: Optional vector store for semantic search
        llm_client: Optional LLM client for natural language processing
    """
    qa_interface = QAInterface(storage_manager, vector_store, llm_client)
    qa_interface.render()
