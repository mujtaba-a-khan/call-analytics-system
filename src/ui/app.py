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

# Page configuration - must be first Streamlit command
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
        config = {}
        config_dir = Path(__file__).parent.parent.parent / 'config'
        
        # Load all TOML files in config directory
        config_files = ['app.toml', 'models.toml', 'vectorstore.toml', 'rules.toml']
        
        for config_file in config_files:
            config_path = config_dir / config_file
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        file_config = toml.load(f)
                        config.update(file_config)
                        logger.info(f"Loaded configuration from {config_file}")
                except Exception as e:
                    logger.warning(f"Failed to load {config_file}: {e}")
            else:
                logger.warning(f"Configuration file not found: {config_file}")
        
        # Set defaults if config is empty
        if not config:
            logger.warning("No configuration files found, using defaults")
            config = self.get_default_config()
        
        return config
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration if config files are missing.
        
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
                'enabled': False,
                'model': 'llama3',
                'api_base': 'http://localhost:11434'
            }
        }
    
    def initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.data = None
            st.session_state.filtered_data = None
            st.session_state.vector_store = None
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
            
            # Show component status
            status_items = []
            
            # Check data status
            if st.session_state.data is not None:
                row_count = len(st.session_state.data)
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
            
            # Footer
            st.divider()
            st.caption(f"v{self.config.get('app', {}).get('version', '1.0.0')}")
            st.caption(f"Python {sys.version_info.major}.{sys.version_info.minor}")
            
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
            render_dashboard_page(
                data=st.session_state.data,
                config=self.config
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
            from core.storage_manager import StorageManager
            
            storage_manager = StorageManager(
                base_path=Path(self.config['paths']['data'])
            )
            render_upload_page(
                storage_manager=storage_manager,
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
                data=st.session_state.data,
                vector_store=st.session_state.vector_store,
                config=self.config
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
        """Render the Q&A interface with lazy loading"""
        try:
            from ui.pages.qa_interface import render_qa_interface
            render_qa_interface(
                data=st.session_state.data,
                vector_store=st.session_state.vector_store,
                config=self.config
            )
        except ImportError:
            # Fallback to basic Q&A
            st.header("💬 Q&A Interface")
            st.info("Q&A components are being loaded...")
            
            query = st.text_input("Ask a question about your data:")
            if query and st.button("Submit"):
                st.info("Processing your question...")
                # Basic keyword search fallback
                if st.session_state.data is not None and 'transcript' in st.session_state.data.columns:
                    results = st.session_state.data[
                        st.session_state.data['transcript'].str.contains(
                            query, case=False, na=False
                        )
                    ]
                    if not results.empty:
                        st.write(f"Found {len(results)} matching records")
                        st.dataframe(results.head())
                    else:
                        st.write("No matching records found")
                else:
                    st.warning("No data available for search")
    
    def render_settings(self) -> None:
        """Render the settings page"""
        st.header("⚙️ Settings")
        
        # Display current configuration
        st.subheader("Current Configuration")
        
        with st.expander("Application Settings"):
            st.json(self.config.get('app', {}))
        
        with st.expander("Model Settings"):
            st.json(self.config.get('whisper', {}))
            st.json(self.config.get('ollama', {}))
        
        with st.expander("Storage Settings"):
            st.json(self.config.get('paths', {}))
            st.json(self.config.get('vectordb', {}))
        
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


def main():
    """Main entry point for the application"""
    try:
        # Create logs directory if it doesn't exist
        Path('logs').mkdir(exist_ok=True)
        
        # Initialize and run application
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