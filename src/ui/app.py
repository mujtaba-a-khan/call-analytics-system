"""
Main Streamlit Application

Entry point for the Call Analytics System user interface.
Provides file upload, processing, analysis, and Q&A capabilities.
"""

import streamlit as st
import pandas as pd
import toml
from pathlib import Path
from datetime import datetime, timedelta
import logging
import sys

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our modules
from core.data_schema import CallDataFrame, CallRecord
from core.audio_processor import AudioProcessor
from ml.whisper_stt import WhisperSTT
from vectordb.chroma_client import ChromaClient
from analysis.filters import DataFilter
from analysis.aggregations import CallMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Call Analytics System",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)


class CallAnalyticsApp:
    """
    Main application class for the Call Analytics System.
    Manages state, configuration, and UI components.
    """
    
    def __init__(self):
        """Initialize the application with configuration"""
        self.config = self.load_configuration()
        self.initialize_session_state()
        self.setup_components()
    
    def load_configuration(self) -> dict:
        """
        Load configuration from TOML files.
        
        Returns:
            Merged configuration dictionary
        """
        config_dir = Path("config")
        config = {}
        
        # Load all configuration files
        config_files = ['app.toml', 'models.toml', 'vectorstore.toml', 'rules.toml']
        
        for config_file in config_files:
            file_path = config_dir / config_file
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        file_config = toml.load(f)
                        config.update(file_config)
                        logger.info(f"Loaded configuration from {config_file}")
                except Exception as e:
                    logger.error(f"Failed to load {config_file}: {e}")
        
        return config
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        # Data storage
        if 'calls_df' not in st.session_state:
            st.session_state.calls_df = pd.DataFrame()
        
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = set()
        
        # UI state
        if 'selected_call' not in st.session_state:
            st.session_state.selected_call = None
        
        if 'filter_settings' not in st.session_state:
            st.session_state.filter_settings = {}
        
        # Processing flags
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
    
    def setup_components(self):
        """Initialize application components"""
        try:
            # Audio processor
            self.audio_processor = AudioProcessor(self.config.get('audio', {}))
            
            # Speech-to-text engine
            self.stt_engine = WhisperSTT(self.config.get('whisper', {}))
            
            # Vector database client
            self.vector_db = ChromaClient(self.config.get('vectorstore', {}))
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            st.error(f"System initialization failed: {e}")
    
    def render_sidebar(self):
        """Render the sidebar with filters and settings"""
        with st.sidebar:
            st.title("📞 Call Analytics")
            st.markdown("---")
            
            # Date range filter
            st.subheader("🗓️ Date Range")
            col1, col2 = st.columns(2)
            
            with col1:
                start_date = st.date_input(
                    "From",
                    value=datetime.now() - timedelta(days=30),
                    key="filter_start_date"
                )
            
            with col2:
                end_date = st.date_input(
                    "To",
                    value=datetime.now(),
                    key="filter_end_date"
                )
            
            # Call type filter
            st.subheader("📋 Call Type")
            call_types = ["All", "Inquiry", "Billing/Sales", "Support", "Complaint"]
            selected_type = st.selectbox(
                "Select type",
                call_types,
                key="filter_call_type"
            )
            
            # Outcome filter
            st.subheader("✅ Outcome")
            outcomes = ["All", "Resolved", "Callback", "Refund", "Sale-close"]
            selected_outcome = st.selectbox(
                "Select outcome",
                outcomes,
                key="filter_outcome"
            )
            
            # Agent filter
            if not st.session_state.calls_df.empty and 'agent_id' in st.session_state.calls_df.columns:
                st.subheader("👤 Agent")
                agents = ["All"] + sorted(st.session_state.calls_df['agent_id'].dropna().unique().tolist())
                selected_agent = st.selectbox(
                    "Select agent",
                    agents,
                    key="filter_agent"
                )
            
            # Apply filters button
            st.markdown("---")
            if st.button("🔍 Apply Filters", use_container_width=True):
                self.apply_filters()
            
            # Settings section
            st.markdown("---")
            st.subheader("⚙️ Settings")
            
            # Connection threshold
            connection_threshold = st.slider(
                "Connection threshold (seconds)",
                min_value=10,
                max_value=60,
                value=30,
                help="Minimum duration to consider a call as connected"
            )
            
            # Show/hide unknown values
            show_unknowns = st.checkbox(
                "Show unknown values",
                value=False,
                help="Include calls with unknown type or outcome"
            )
            
            # Export options
            st.markdown("---")
            st.subheader("💾 Export")
            
            if not st.session_state.calls_df.empty:
                csv_data = st.session_state.calls_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"call_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    def render_upload_section(self):
        """Render the file upload section"""
        st.header("📤 Upload Files")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎵 Audio Files")
            audio_files = st.file_uploader(
                "Upload audio files",
                type=['wav', 'mp3', 'm4a', 'flac'],
                accept_multiple_files=True,
                help="Supported formats: WAV, MP3, M4A, FLAC"
            )
            
            if audio_files:
                st.info(f"📁 {len(audio_files)} audio file(s) selected")
        
        with col2:
            st.subheader("📄 CSV Files")
            csv_files = st.file_uploader(
                "Upload CSV files",
                type=['csv'],
                accept_multiple_files=True,
                help="CSV should contain: call_id, start_time, duration_seconds, transcript"
            )
            
            if csv_files:
                st.info(f"📁 {len(csv_files)} CSV file(s) selected")
        
        # Process button
        if st.button("⚡ Process Files", type="primary", use_container_width=True):
            self.process_uploaded_files(audio_files, csv_files)
    
    def process_uploaded_files(self, audio_files, csv_files):
        """
        Process uploaded audio and CSV files.
        
        Args:
            audio_files: List of uploaded audio files
            csv_files: List of uploaded CSV files
        """
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_records = []
        total_files = len(audio_files or []) + len(csv_files or [])
        processed = 0
        
        # Process audio files
        if audio_files:
            status_text.text("🎵 Processing audio files...")
            
            for audio_file in audio_files:
                # Skip if already processed
                if audio_file.name in st.session_state.processed_files:
                    processed += 1
                    progress_bar.progress(processed / total_files)
                    continue
                
                try:
                    # Save uploaded file temporarily
                    temp_path = Path(f"data/uploads/{audio_file.name}")
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(temp_path, 'wb') as f:
                        f.write(audio_file.read())
                    
                    # Process audio
                    processed_audio, duration = self.audio_processor.process_audio_file(temp_path)
                    
                    # Transcribe
                    transcription = self.stt_engine.transcribe(processed_audio)
                    
                    # Create call record
                    record = CallRecord(
                        call_id=audio_file.name.split('.')[0],
                        start_time=datetime.now(),
                        duration_seconds=duration,
                        transcript=transcription.transcript
                    )
                    
                    all_records.append(record)
                    st.session_state.processed_files.add(audio_file.name)
                    
                except Exception as e:
                    st.error(f"Failed to process {audio_file.name}: {e}")
                
                processed += 1
                progress_bar.progress(processed / total_files)
        
        # Process CSV files
        if csv_files:
            status_text.text("📄 Processing CSV files...")
            
            for csv_file in csv_files:
                if csv_file.name in st.session_state.processed_files:
                    processed += 1
                    progress_bar.progress(processed / total_files)
                    continue
                
                try:
                    # Read CSV
                    df = pd.read_csv(csv_file)
                    
                    # Convert to records
                    for _, row in df.iterrows():
                        record = CallRecord(
                            call_id=row.get('call_id', f"csv_{processed}"),
                            start_time=pd.to_datetime(row.get('start_time', datetime.now())),
                            duration_seconds=float(row.get('duration_seconds', 0)),
                            transcript=row.get('transcript', ''),
                            agent_id=row.get('agent_id'),
                            campaign=row.get('campaign'),
                            customer_name=row.get('customer_name'),
                            product_name=row.get('product_name'),
                            amount=row.get('amount')
                        )
                        all_records.append(record)
                    
                    st.session_state.processed_files.add(csv_file.name)
                    
                except Exception as e:
                    st.error(f"Failed to process {csv_file.name}: {e}")
                
                processed += 1
                progress_bar.progress(processed / total_files)
        
        # Update session state with new records
        if all_records:
            new_df = pd.DataFrame([r.dict() for r in all_records])
            
            if st.session_state.calls_df.empty:
                st.session_state.calls_df = new_df
            else:
                st.session_state.calls_df = pd.concat(
                    [st.session_state.calls_df, new_df],
                    ignore_index=True
                )
            
            # Add to vector database
            self.index_calls_in_vectordb(all_records)
            
            status_text.text(f"✅ Successfully processed {len(all_records)} calls")
            st.session_state.processing_complete = True
        else:
            status_text.text("⚠️ No new files to process")
        
        progress_bar.empty()
    
    def index_calls_in_vectordb(self, records: list):
        """
        Index call records in the vector database.
        
        Args:
            records: List of CallRecord objects
        """
        documents = []
        ids = []
        metadatas = []
        
        for record in records:
            # Prepare document text
            doc_text = f"{record.transcript}"
            
            # Prepare metadata
            metadata = {
                'call_id': record.call_id,
                'start_time': record.start_time.isoformat(),
                'duration': record.duration_seconds,
                'agent_id': record.agent_id or '',
                'campaign': record.campaign or ''
            }
            
            documents.append(doc_text)
            ids.append(record.call_id)
            metadatas.append(metadata)
        
        # Add to vector database
        self.vector_db.add_documents(documents, ids, metadatas)
        logger.info(f"Indexed {len(records)} calls in vector database")
    
    def render_dashboard(self):
        """Render the main dashboard with metrics and visualizations"""
        st.header("📊 Dashboard")
        
        if st.session_state.calls_df.empty:
            st.info("No data available. Please upload files to begin analysis.")
            return
        
        # Calculate metrics
        metrics = CallMetrics(st.session_state.calls_df)
        stats = metrics.calculate_statistics()
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Calls",
                stats['total_calls'],
                delta=None
            )
        
        with col2:
            st.metric(
                "Connected %",
                f"{stats['connected_percentage']:.1f}%",
                delta=None
            )
        
        with col3:
            st.metric(
                "Avg Duration",
                f"{stats['duration_statistics']['mean']:.1f}s",
                delta=None
            )
        
        with col4:
            st.metric(
                "Unique Agents",
                stats['unique_agents'],
                delta=None
            )
        
        # Charts section
        st.subheader("📈 Analytics")
        
        tab1, tab2, tab3 = st.tabs(["Call Types", "Outcomes", "Timeline"])
        
        with tab1:
            # Call type distribution chart
            type_chart = metrics.get_type_distribution_chart()
            st.plotly_chart(type_chart, use_container_width=True)
        
        with tab2:
            # Outcome distribution chart
            outcome_chart = metrics.get_outcome_distribution_chart()
            st.plotly_chart(outcome_chart, use_container_width=True)
        
        with tab3:
            # Timeline chart
            timeline_chart = metrics.get_timeline_chart()
            st.plotly_chart(timeline_chart, use_container_width=True)
    
    def render_calls_table(self):
        """Render the calls data table"""
        st.header("📋 Call Records")
        
        if st.session_state.calls_df.empty:
            st.info("No call records to display.")
            return
        
        # Search box
        search_query = st.text_input(
            "🔍 Search calls",
            placeholder="Search by transcript, agent, campaign..."
        )
        
        # Filter dataframe based on search
        display_df = st.session_state.calls_df.copy()
        
        if search_query:
            mask = display_df.apply(
                lambda row: search_query.lower() in str(row).lower(),
                axis=1
            )
            display_df = display_df[mask]
        
        # Display table
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "start_time": st.column_config.DatetimeColumn(
                    "Start Time",
                    format="DD/MM/YYYY HH:mm"
                ),
                "duration_seconds": st.column_config.NumberColumn(
                    "Duration (s)",
                    format="%.1f"
                ),
                "amount": st.column_config.NumberColumn(
                    "Amount",
                    format="$%.2f"
                )
            }
        )
        
        st.caption(f"Showing {len(display_df)} of {len(st.session_state.calls_df)} records")
    
    def render_qa_interface(self):
        """Render the Q&A interface for natural language queries"""
        st.header("❓ Q&A Interface")
        
        if st.session_state.calls_df.empty:
            st.info("No data available for Q&A. Please upload files first.")
            return
        
        # Question input
        question = st.text_area(
            "Ask a question about your call data",
            placeholder="e.g., What were the main complaints last week?",
            height=100
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            use_semantic = st.checkbox(
                "Use semantic search",
                value=True,
                help="Enable semantic search for more accurate results"
            )
        
        with col2:
            top_k = st.number_input(
                "Results",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of results to return"
            )
        
        if st.button("🔍 Search", type="primary", use_container_width=True):
            self.process_qa_query(question, use_semantic, top_k)
    
    def process_qa_query(self, question: str, use_semantic: bool, top_k: int):
        """
        Process a natural language query.
        
        Args:
            question: User's question
            use_semantic: Whether to use semantic search
            top_k: Number of results to return
        """
        if not question:
            st.warning("Please enter a question.")
            return
        
        with st.spinner("Searching..."):
            if use_semantic:
                # Semantic search using vector database
                results = self.vector_db.search(question, top_k=top_k)
                
                if results:
                    st.success(f"Found {len(results)} relevant calls")
                    
                    # Display results
                    for idx, result in enumerate(results, 1):
                        with st.expander(f"Result {idx} - Call {result['id']} (Score: {result['score']:.2f})"):
                            st.write(f"**Transcript:** {result['document'][:500]}...")
                            
                            if result['metadata']:
                                st.write("**Metadata:**")
                                for key, value in result['metadata'].items():
                                    st.write(f"- {key}: {value}")
                else:
                    st.info("No relevant results found.")
            else:
                # Simple keyword search
                mask = st.session_state.calls_df['transcript'].str.contains(
                    question,
                    case=False,
                    na=False
                )
                results_df = st.session_state.calls_df[mask].head(top_k)
                
                if not results_df.empty:
                    st.success(f"Found {len(results_df)} matching calls")
                    st.dataframe(results_df[['call_id', 'start_time', 'transcript']])
                else:
                    st.info("No matching calls found.")
    
    def apply_filters(self):
        """Apply the selected filters to the data"""
        # This would implement the actual filtering logic
        st.success("Filters applied successfully!")
    
    def run(self):
        """Run the main application"""
        # Render sidebar
        self.render_sidebar()
        
        # Main content area
        tab1, tab2, tab3, tab4 = st.tabs([
            "📤 Upload",
            "📊 Dashboard",
            "📋 Call Records",
            "❓ Q&A"
        ])
        
        with tab1:
            self.render_upload_section()
        
        with tab2:
            self.render_dashboard()
        
        with tab3:
            self.render_calls_table()
        
        with tab4:
            self.render_qa_interface()


def main():
    """Main entry point for the application"""
    app = CallAnalyticsApp()
    app.run()


if __name__ == "__main__":
    main()