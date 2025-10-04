"""
Main Streamlit Application

Entry point for the Call Analytics System user interface.
Provides file upload, processing, analysis, and Q&A capabilities.
Compatible with Python 3.13+
"""

import sys
import os
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import traceback
from datetime import datetime, timedelta

# Ensure we're using Python 3.13+
if sys.version_info < (3, 13):
    print(f"Error: Python 3.13 or higher is required. Current version: {sys.version}")
    sys.exit(1)

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging before any other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/app.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Import with error handling
try:
    import streamlit as st
    import pandas as pd
    import toml
except ImportError as e:
    logger.error(f"Failed to import required package: {e}")
    print(f"Error: Missing required package. Please install dependencies: pip install -e .")
    sys.exit(1)

def _has_streamlit_context() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:
        try:
            from streamlit.script_run_context import get_script_run_ctx  # type: ignore
        except ImportError:
            return False

    try:
        return get_script_run_ctx() is not None
    except RuntimeError:
        return False


def _configure_streamlit_page() -> None:
    """Apply page config and sidebar styling when Streamlit context exists."""

    if not _has_streamlit_context():
        return

    st.set_page_config(
        page_title="Call Analytics System",
        page_icon="📞",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/mujtaba-a-khan/call-analytics-system',
            'Report a bug': 'https://github.com/mujtaba-a-khan/call-analytics-system/issues',
            'About': 'Call Analytics System v1.0.0 - Professional call center analytics'
        }
    )

    st.markdown(
        """
        <style>
        div[data-testid="stSidebarNav"] { display: none; }
        [data-testid="stSidebar"] {
            padding: 0 !important;
            min-width: 300px;
            width: 300px;
        }
        [data-testid="stSidebar"] > div:first-child {
            padding: 1.5rem 1.25rem 2rem;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] > div:first-child {
            flex: 1 1 auto;
            overflow-y: auto;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] > div:last-child {
            margin-top: auto;
        }
        [data-testid="stSidebar"] .sidebar-footer {
            font-size: 0.85rem;
            opacity: 0.75;
            padding-top: 0.75rem;
        }
        [data-testid="stSidebar"] .sidebar-footer span {
            display: block;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


class CallAnalyticsApp:
    """
    Main application class for the Call Analytics System.
    Manages state, configuration, and UI components.
    """
    
    def __init__(self):
        """Initialize the application with configuration"""
        try:
            self.config = self.load_configuration()
            self.initialize_session_state()
            self.setup_components()
            logger.info("Application initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize application: {e}")
            st.error(f"Failed to initialize application: {str(e)}")
            raise
    
    def load_configuration(self) -> Dict[str, Any]:
        """
        Load configuration from TOML files.
        
        Returns:
            Dict[str, Any]: Merged configuration dictionary
        """
        config = self.get_default_config()
        
        # Try to load configuration files
        config_dir = Path('config')
        if config_dir.exists():
            config_files = sorted(config_dir.glob('*.toml'))

            for config_path in config_files:
                if config_path.exists():
                    try:
                        with open(config_path, 'r') as f:
                            file_config = toml.load(f)
                            config = self.merge_configs(config, file_config)
                            logger.info(f"Loaded configuration from {config_path.name}")
                    except Exception as e:
                        logger.warning(f"Failed to load {config_path.name}: {e}")
        
        return config
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration.
        
        Returns:
            Dict[str, Any]: Default configuration
        """
        return {
            'app': {
                'name': 'Call Analytics System',
                'version': '1.0.0',
                'debug': False,
                'cache_enabled': True,
                'max_upload_size_mb': 500
            },
            'paths': {
                'data': 'data',
                'models': 'models',
                'logs': 'logs',
                'exports': 'data/exports',
                'vector_db': 'data/vector_db'
            },
            'whisper': {
                'enabled': True,
                'model_size': 'small',
                'device': 'cpu',
                'compute_type': 'int8'
            },
            'vectordb': {
                'enabled': True,
                'persist_directory': 'data/vector_db',
                'collection_name': 'call_transcripts'
            },
            'ollama': {
                'enabled': True,
                'model': 'llama3:8b',
                'api_base': 'http://localhost:11434'
            }
        }

    @staticmethod
    def merge_configs(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = CallAnalyticsApp.merge_configs(base[key], value)
            else:
                base[key] = value
        return base
    
    def initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.data = None
            st.session_state.filtered_data = None
            st.session_state.vector_store = None
            st.session_state.storage_manager = None
            st.session_state.llm_client = None
            st.session_state.search_results = []
            st.session_state.current_page = 'Dashboard'
            st.session_state.processing = False
            st.session_state.upload_history = []
            st.session_state.filter_state = {}
            st.session_state.qa_history = []
            logger.info("Session state initialized")
    
    def setup_components(self) -> None:
        """
        Setup application components with lazy loading.
        Components are only initialized when needed.
        """
        try:
            # Create required directories
            for path_key, path_value in self.config.get('paths', {}).items():
                Path(path_value).mkdir(parents=True, exist_ok=True)
            
            # Initialize storage manager if not already done
            if st.session_state.storage_manager is None:
                from core.storage_manager import StorageManager
                st.session_state.storage_manager = StorageManager(
                    base_path=Path(self.config['paths']['data'])
                )
                logger.info("Storage manager initialized")
            from vectordb.chroma_client import ChromaClient
            if st.session_state.vector_store is None:
                vector_cfg = self.config.get('vectorstore', {})
                st.session_state.vector_store = ChromaClient(vector_cfg)

                # Automatically populate the vector store if it is empty
                try:
                    stats = st.session_state.vector_store.get_statistics()
                    if stats.get('total_documents', 0) == 0:
                        data_df = st.session_state.storage_manager.load_all_records()
                        if data_df is not None and not data_df.empty:
                            from vectordb.indexer import DocumentIndexer

                            indexing_config = dict(vector_cfg.get('indexing', {}))
                            indexing_config.setdefault('text_fields', ['transcript', 'notes'])
                            indexing_config.setdefault('min_text_length', 10)
                            indexing_config.setdefault(
                                'metadata_fields',
                                [
                                    'call_id',
                                    'agent_id',
                                    'campaign',
                                    'call_type',
                                    'outcome',
                                    'timestamp',
                                    'duration',
                                    'revenue'
                                ]
                            )

                            indexer = DocumentIndexer(
                                st.session_state.vector_store,
                                config=indexing_config
                            )

                            required_fields = set(indexing_config['metadata_fields'])
                            existing_fields: set[str] = set()
                            try:
                                sample = st.session_state.vector_store.collection.peek(1)
                                if sample and sample.get('metadatas'):
                                    metadata_sample = sample['metadatas'][0]
                                    if metadata_sample:
                                        existing_fields = set(metadata_sample.keys())
                            except Exception as peek_error:
                                logger.debug(
                                    "Unable to inspect vector store metadata: %s",
                                    peek_error
                                )

                            needs_reindex = stats.get('total_documents', 0) == 0
                            if not needs_reindex and required_fields - existing_fields:
                                needs_reindex = True
                                logger.info(
                                    "Vector store metadata missing fields %s; triggering reindex",
                                    required_fields - existing_fields
                                )

                            if needs_reindex:
                                if stats.get('total_documents', 0) > 0:
                                    indexed_count = indexer.reindex_all(data_df)
                                else:
                                    indexed_count = indexer.index_dataframe(data_df)

                                logger.info(
                                    "Populated vector store with %d document(s)",
                                    indexed_count
                                )
                        else:
                            logger.info("No call records available to index into vector store")
                except Exception as index_error:
                    logger.warning(
                        "Vector store initialization skipped: %s",
                        index_error
                    )

            # Initialize LLM client if enabled
            if st.session_state.llm_client is None and self.config.get('ollama', {}).get('enabled', False):
                try:
                    from ml.llm_interface import LocalLLMInterface

                    llm_config = dict(self.config.get('llm', {}))
                    ollama_cfg = self.config.get('ollama', {})

                    provider = llm_config.get('provider', 'ollama')
                    llm_config['provider'] = provider

                    model_name = llm_config.pop('model', None) or llm_config.get('model_name') or ollama_cfg.get('model', 'llama3')
                    llm_config['model_name'] = model_name

                    llm_config.setdefault('endpoint', ollama_cfg.get('api_base', 'http://localhost:11434'))
                    llm_config.setdefault('temperature', ollama_cfg.get('temperature', 0.7))
                    llm_config.setdefault('max_tokens', ollama_cfg.get('max_tokens', 1024))

                    st.session_state.llm_client = LocalLLMInterface(llm_config)

                    if not st.session_state.llm_client.is_available:
                        logger.warning("LLM client initialized but service is unavailable")
                except Exception as llm_error:
                    logger.warning(f"Failed to initialize LLM client: {llm_error}")

            
            # Components will be initialized on-demand when pages are accessed
            self.components_ready = True
            logger.info("Components setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup components: {e}")
            self.components_ready = False
            raise
    
    def render_sidebar(self) -> str:
        """
        Render the sidebar navigation.
        
        Returns:
            str: Selected page name
        """
        with st.sidebar:
            sidebar_main = st.container()
            sidebar_footer = st.container()

            with sidebar_main:
                st.title("📞 Call Analytics")
                st.divider()
                # Navigation
                pages = {
                    "Dashboard": "📊",
                    "Upload Data": "📤",
                    "Analysis": "🔍",
                    "Q&A Interface": "💬",
                    "Settings": "⚙️"
                }

                selected_page = st.radio(
                    "Navigation",
                    options=list(pages.keys()),
                    format_func=lambda x: f"{pages[x]} {x}",
                    index=list(pages.keys()).index(st.session_state.current_page),
                    label_visibility="collapsed"
                )

                st.session_state.current_page = selected_page

                # System status
                st.divider()
                st.caption("System Status")

                status_items: list[str] = []

                # Check data status
                row_count = 0
                try:
                    row_count = st.session_state.storage_manager.get_record_count()
                except Exception as count_error:
                    logger.warning(f"Unable to determine record count: {count_error}")

                if row_count > 0:
                    status_items.append(f"✅ {row_count:,} records loaded")
                else:
                    status_items.append("⚠️ No data loaded")

                # Check vector store status
                if st.session_state.vector_store is not None:
                    status_items.append("✅ Vector store ready")
                else:
                    status_items.append("⚠️ Vector store not initialized")

                # Check Whisper status
                if self.config.get('whisper', {}).get('enabled', False):
                    status_items.append("✅ Whisper STT available")
                else:
                    status_items.append("ℹ️ Whisper STT disabled")

                # Check Ollama status
                if self.config.get('ollama', {}).get('enabled', False):
                    status_items.append("✅ Ollama LLM available")
                else:
                    status_items.append("ℹ️ Ollama LLM disabled")

                for item in status_items:
                    st.caption(item)

            version_text = self.config.get('app', {}).get('version', '1.0.0')
            python_version = f"Python {sys.version_info.major}.{sys.version_info.minor}"

            with sidebar_footer:
                st.markdown(
                    f"""
                    <div class="sidebar-footer">
                        <hr />
                        <span>v{version_text}</span>
                        <span>{python_version}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        return selected_page
    
    def render_page(self, page_name: str) -> None:
        """
        Render the selected page with lazy loading.
        
        Args:
            page_name: Name of the page to render
        """
        try:
            if page_name == "Dashboard":
                self.render_dashboard()
            elif page_name == "Upload Data":
                self.render_upload()
            elif page_name == "Analysis":
                self.render_analysis()
            elif page_name == "Q&A Interface":
                self.render_qa_interface()
            elif page_name == "Settings":
                self.render_settings()
            else:
                st.error(f"Unknown page: {page_name}")
                
        except ImportError as e:
            st.error(f"Failed to load page components: {str(e)}")
            st.info("Please ensure all dependencies are installed: pip install -e .")
            logger.error(f"Import error rendering {page_name}: {e}")
            
        except Exception as e:
            st.error(f"Error rendering page: {str(e)}")
            logger.error(f"Error rendering {page_name}: {e}\n{traceback.format_exc()}")
    
    def render_dashboard(self) -> None:
        """Render the dashboard page with lazy loading"""
        try:
            from ui.pages.dashboard import render_dashboard_page
            
            # Pass storage manager instead of data
            render_dashboard_page(
                storage_manager=st.session_state.storage_manager
            )
        except ImportError:
            # Fallback to basic dashboard
            st.header("📊 Dashboard")
            st.info("Advanced dashboard components are being loaded...")
            
            if st.session_state.data is not None:
                st.subheader("Data Overview")
                st.write(f"Total Records: {len(st.session_state.data):,}")
                st.dataframe(st.session_state.data.head(10))
            else:
                st.warning("No data loaded. Please upload data first.")
    
    def render_upload(self) -> None:
        """Render the upload page with lazy loading"""
        try:
            from ui.pages.upload import render_upload_page
            
            render_upload_page(
                storage_manager=st.session_state.storage_manager,
                config=self.config
            )
        except ImportError as e:
            # Fallback to basic upload
            st.header("📤 Upload Data")
            st.info("Upload components are being loaded...")
            
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload a CSV file containing call transcripts"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.data = df
                    st.success(f"Loaded {len(df):,} records")
                    st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
    
    def render_analysis(self) -> None:
        """Render the analysis page with lazy loading"""
        try:
            from ui.pages.analysis import render_analysis_page
            
            render_analysis_page(
                storage_manager=st.session_state.storage_manager,
                vector_store=st.session_state.vector_store
            )
        except ImportError:
            # Fallback to basic analysis
            st.header("🔍 Analysis")
            st.info("Analysis components are being loaded...")
            
            if st.session_state.data is not None:
                st.subheader("Basic Statistics")
                st.write(st.session_state.data.describe())
            else:
                st.warning("No data loaded for analysis.")
    
    def render_qa_interface(self) -> None:
        """Render the Q&A interface page with lazy loading"""
        try:
            from ui.pages.qa_interface import render_qa_interface

            render_qa_interface(
                storage_manager=st.session_state.storage_manager,
                vector_store=st.session_state.vector_store,
                llm_client=st.session_state.llm_client
            )
        except ImportError:
            # Fallback to basic Q&A
            st.header("💬 Q&A Interface")
            st.info("Q&A components are being loaded...")
            
            question = st.text_input("Ask a question about your data:")
            if question:
                st.info("Q&A functionality requires vector store and LLM setup.")
    
    def render_settings(self) -> None:
        """Render the settings page"""
        st.header("⚙️ Settings")
        
        # Display configuration sections
        st.subheader("Application Configuration")
        
        # Show current configuration
        with st.expander("Current Configuration", expanded=False):
            st.json(self.config)
        
        # System information
        st.subheader("System Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Python Version", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        
        with col2:
            st.metric("Streamlit Version", st.__version__)
        
        with col3:
            import platform
            st.metric("Platform", platform.system())
        
        # Clear cache button
        if st.button("Clear Cache", type="secondary"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared successfully")
            st.rerun()
    
    def run(self) -> None:
        """Main application loop"""
        try:
            # Render sidebar and get selected page
            selected_page = self.render_sidebar()
            
            # Render selected page
            self.render_page(selected_page)
            
        except Exception as e:
            logger.error(f"Application error: {e}\n{traceback.format_exc()}")
            st.error("An unexpected error occurred. Please check the logs.")
            
            if st.checkbox("Show error details"):
                st.exception(e)


def main() -> None:
    """Main entry point for the application"""

    try:
        _configure_streamlit_page()
        Path('logs').mkdir(exist_ok=True)
        app = CallAnalyticsApp()
        app.run()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Critical application error: {e}\n{traceback.format_exc()}")
        st.error(f"Critical error: {str(e)}")
        st.stop()


if __name__ == "__main__":
    main()
