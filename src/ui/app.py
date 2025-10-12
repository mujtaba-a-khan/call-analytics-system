"""
Main Streamlit Application

Entry point for the Call Analytics System user interface.
Provides file upload, processing, analysis, and Q&A capabilities.
Compatible with Python 3.11+
"""

import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging before any other imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/app.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Import with error handling
try:
    import pandas as pd
    import streamlit as st
    import toml
except ImportError as e:
    logger.error(f"Failed to import required package: {e}")
    print("Error: Missing required package. Please install dependencies: pip install -e .")
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
            "Get Help": "https://github.com/mujtaba-a-khan/call-analytics-system",
            "Report a bug": "https://github.com/mujtaba-a-khan/call-analytics-system/issues",
            "About": "Call Analytics System v1.0.0 - Professional call center analytics",
        },
    )

    st.markdown(
        """
        <style>
        :root {
            --sidebar-width: 300px;
        }

        html, body {
            margin: 0;
            padding: 0;
            width: 100%;
        }

        .stApp {
            margin: 0;
            padding: 0;
        }

        div[data-testid="stSidebarNav"] { display: none; }

        [data-testid="stSidebar"] {
            padding: 0 !important;
            min-width: var(--sidebar-width);
            width: var(--sidebar-width);
            position: fixed;
            top: 0;
            left: 0;
            bottom: 0;
            background: rgba(13, 16, 24, 0.98);
            border-right: 1px solid rgba(255, 255, 255, 0.06);
            z-index: 60;
            transform: none !important;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding: 3.25rem 1.25rem 1.25rem;
            height: 100%;
        }

        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            display: flex;
            flex-direction: column;
            height: 100%;
            overflow-y: auto;
        }

        [data-testid="stSidebar"] [data-testid="stSidebarContent"] > div:first-child {
            flex: 0 0 auto;
            padding-right: 0.35rem;
        }

        [data-testid="stSidebar"] [data-testid="stSidebarContent"] > div + div {
            margin-top: 1.5rem;
        }

        [data-testid="stSidebar"] button[title="Collapse sidebar"],
        [data-testid="stSidebar"] button[aria-label="Collapse sidebar"],
        [data-testid="stSidebar"] button[aria-label="Close sidebar"],
        button[title*="Sidebar"],
        button[aria-label*="sidebar"],
        button[aria-label*="Sidebar"],
        button[data-testid="collapsedSidebarButton"],
        button[data-testid="baseButton-sidebarCollapseButton"],
        button[data-testid="baseButton-collapsedSidebarButton"],
        button[data-testid="stSidebarNavToggle"],
        [data-testid*="sidebarCollapse"],
        [data-testid*="SidebarCollapse"],
        [data-testid*="collapsedSidebar"],
        [data-testid*="CollapsedSidebar"],
        div[data-testid="collapsedControl"],
        div[data-testid="collapsedSidebarButton"],
        div[data-testid="collapsedSidebarButton"] button,
        div[data-testid="collapsedControl"] button {
            display: none !important;
        }

        [data-testid="stSidebar"] h1 {
            margin-top: 0;
            margin-bottom: 0.75rem;
        }

        [data-testid="stSidebar"] .sidebar-footer {
            font-size: 0.85rem;
            opacity: 0.75;
            padding: 0.75rem 0 0.9rem;
            position: sticky;
            bottom: 0;
            background: linear-gradient(180deg, rgba(13, 16, 24, 0) 0%, rgba(13, 16, 24, 0.95) 45%);
            border-top: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] .sidebar-footer span {
            display: block;
        }

        /* Style the "View" cells in the recent calls table */
        div[data-testid="stDataFrame"][data-st-key="recent_calls_table"]
        [data-testid="stDataFrameSelectionHeader"],
        div[data-testid="stDataFrame"][data-st-key="recent_calls_table"]
        [data-testid="stDataFrameRowCheckbox"],
        div[data-testid="stDataFrame"][data-st-key="recent_calls_table"]
        [aria-label="Select rows"] {
            display: none !important;
        }
        div[data-testid="stDataFrame"][data-st-key="recent_calls_table"] tbody tr td:last-child {
            white-space: nowrap;
        }
        div[data-testid="stDataFrame"][data-st-key="recent_calls_table"] tbody tr td:last-child a {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.35rem;
            padding: 0.25rem 0.9rem;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(255, 255, 255, 0.08);
            color: rgba(255, 255, 255, 0.85);
            font-size: 0.85rem;
            font-weight: 500;
            text-decoration: none;
            cursor: pointer;
            transition: background 0.15s ease, border-color 0.15s ease;
        }
        div[data-testid="stDataFrame"][data-st-key="recent_calls_table"]
        tbody tr td:last-child a::before {
            content: '🔍';
        }
        div[data-testid="stDataFrame"][data-st-key="recent_calls_table"]
        tbody tr td:last-child a:hover {
            background: rgba(255, 255, 255, 0.18);
            border-color: rgba(255, 255, 255, 0.2);
            color: rgba(255, 255, 255, 0.95);
        }

        [data-testid="stAppViewContainer"] {
            padding-top: 3.25rem;
            padding-bottom: 4.5rem;
            margin-left: 0;
            padding-left: 0;
            padding-right: 0;
        }

        [data-testid="stAppViewContainer"] > .main {
            padding-top: 0 !important;
            padding-right: 0 !important;
            padding-left: 0 !important;
            margin-left: var(--sidebar-width);
            width: calc(100% - var(--sidebar-width));
        }

        [data-testid="stAppViewContainer"] > .main .block-container {
            padding: 3rem 1.5rem 3.75rem 0.75rem !important;
            margin: 0 !important;
            max-width: none !important;
            width: 100% !important;
        }

        .main .block-container {
            padding: 3rem 1.5rem 3.75rem 0.75rem !important;
            margin: 0 !important;
            max-width: none !important;
            width: 100% !important;
        }

        .main .block-container > h1:first-child {
            margin-top: 0;
            margin-bottom: 0.75rem;
        }

        .app-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 0.75rem 1.5rem;
            padding-left: 1.25rem;
            background: rgba(15, 17, 24, 0.95);
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            display: grid;
            grid-template-columns: minmax(0, auto) minmax(0, 1fr) minmax(0, auto);
            align-items: center;
            gap: 1rem;
            color: rgba(255, 255, 255, 0.7);
            z-index: 100;
        }

        .app-footer__meta {
            display: flex;
            gap: 0.75rem;
            align-items: center;
            flex-wrap: wrap;
            justify-content: flex-start;
        }

        .app-footer__status {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            align-items: center;
            justify-content: center;
        }

        .status-chip {
            padding: 0.25rem 0.6rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.08);
            font-size: 0.75rem;
            color: rgba(255, 255, 255, 0.75);
            white-space: nowrap;
        }

        .status-chip.status-ok {
            background: rgba(34, 197, 94, 0.18);
            color: #4ade80;
        }

        .status-chip.status-warn {
            background: rgba(239, 68, 68, 0.18);
            color: #f87171;
        }

        .status-chip.status-info {
            background: rgba(59, 130, 246, 0.18);
            color: #60a5fa;
        }

        .app-footer__links {
            display: flex;
            gap: 0.75rem;
            align-items: center;
            flex-wrap: wrap;
            justify-content: flex-end;
        }

        .app-footer__links a {
            color: rgba(255, 255, 255, 0.85);
            text-decoration: none;
            font-weight: 500;
        }

        .app-footer__links a:hover {
            text-decoration: underline;
        }

        @media (max-width: 1200px) {
            [data-testid="stAppViewContainer"] {
                margin-left: 0;
                padding-top: 2.75rem;
                padding-left: 0;
            }
            .app-footer {
                padding-left: 1rem;
                grid-template-columns: 1fr;
                text-align: left;
                gap: 0.75rem;
            }
            .app-footer__meta,
            .app-footer__status,
            .app-footer__links {
                justify-content: flex-start;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
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

    def load_configuration(self) -> dict[str, Any]:
        """
        Load configuration from TOML files.

        Returns:
            Dict[str, Any]: Merged configuration dictionary
        """
        config = self.get_default_config()

        # Try to load configuration files
        config_dir = Path("config")
        if config_dir.exists():
            config_files = sorted(config_dir.glob("*.toml"))

            for config_path in config_files:
                if config_path.exists():
                    try:
                        with open(config_path) as f:
                            file_config = toml.load(f)
                            config = self.merge_configs(config, file_config)
                            logger.info(f"Loaded configuration from {config_path.name}")
                    except Exception as e:
                        logger.warning(f"Failed to load {config_path.name}: {e}")

        return config

    def get_default_config(self) -> dict[str, Any]:
        """
        Get default configuration.

        Returns:
            Dict[str, Any]: Default configuration
        """
        return {
            "app": {
                "name": "Call Analytics System",
                "version": "1.0.0",
                "debug": False,
                "cache_enabled": True,
                "max_upload_size_mb": 500,
            },
            "paths": {
                "data": "data",
                "models": "models",
                "logs": "logs",
                "exports": "data/exports",
                "vector_db": "data/vector_db",
            },
            "whisper": {
                "enabled": True,
                "model_size": "small",
                "device": "cpu",
                "compute_type": "int8",
            },
            "vectordb": {
                "enabled": True,
                "persist_directory": "data/vector_db",
                "collection_name": "call_transcripts",
            },
            "ollama": {"enabled": True, "model": "llama3:8b", "api_base": "http://localhost:11434"},
        }

    @staticmethod
    def merge_configs(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = CallAnalyticsApp.merge_configs(base[key], value)
            else:
                base[key] = value
        return base

    def initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables"""
        if "initialized" not in st.session_state:
            st.session_state.initialized = True
            st.session_state.data = None
            st.session_state.filtered_data = None
            st.session_state.vector_store = None
            st.session_state.storage_manager = None
            st.session_state.llm_client = None
            st.session_state.search_results = []
            st.session_state.current_page = "Dashboard"
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
            self._ensure_directories()
            self._ensure_storage_manager()
            self._ensure_vector_store()
            self._ensure_llm_client()

            self.components_ready = True
            logger.info("Components setup completed")

        except Exception as e:
            logger.error("Failed to setup components: %s", e)
            self.components_ready = False
            raise

    def _ensure_directories(self) -> None:
        """Create required directories defined in configuration."""
        for path_value in self.config.get("paths", {}).values():
            Path(path_value).mkdir(parents=True, exist_ok=True)

    def _ensure_storage_manager(self) -> None:
        """Initialize the storage manager if it has not been set up."""
        if st.session_state.storage_manager is not None:
            return

        from core.storage_manager import StorageManager

        st.session_state.storage_manager = StorageManager(
            base_path=Path(self.config["paths"]["data"])
        )
        logger.info("Storage manager initialized")

    def _ensure_vector_store(self) -> None:
        """Initialize the vector store and populate it when necessary."""
        if st.session_state.vector_store is not None:
            return

        from vectordb.chroma_client import ChromaClient

        vector_cfg = self.config.get("vectorstore", {})
        st.session_state.vector_store = ChromaClient(vector_cfg)

        try:
            self._populate_vector_store_if_needed(vector_cfg)
        except Exception as index_error:
            logger.warning("Vector store initialization skipped: %s", index_error)

    def _populate_vector_store_if_needed(self, vector_cfg: dict[str, Any]) -> None:
        """Populate or reindex the vector store depending on current state."""
        vector_store = st.session_state.vector_store
        stats = vector_store.get_statistics()
        total_documents = stats.get("total_documents", 0)

        indexing_config = self._build_indexing_config(vector_cfg)

        if total_documents > 0 and not self._find_missing_metadata_fields(indexing_config):
            return

        data_df = st.session_state.storage_manager.load_all_records()
        if data_df is None or data_df.empty:
            logger.info("No call records available to index into vector store")
            return

        from vectordb.indexer import DocumentIndexer

        indexer = DocumentIndexer(vector_store, config=indexing_config)

        if total_documents > 0:
            indexed_count = indexer.reindex_all(data_df)
        else:
            indexed_count = indexer.index_dataframe(data_df)

        logger.info("Populated vector store with %d document(s)", indexed_count)

    def _build_indexing_config(self, vector_cfg: dict[str, Any]) -> dict[str, Any]:
        """Construct indexing configuration with sensible defaults."""
        indexing_config = dict(vector_cfg.get("indexing", {}))
        indexing_config.setdefault("text_fields", ["transcript", "notes"])
        indexing_config.setdefault("min_text_length", 10)
        indexing_config.setdefault(
            "metadata_fields",
            [
                "call_id",
                "agent_id",
                "campaign",
                "call_type",
                "outcome",
                "timestamp",
                "duration",
                "revenue",
            ],
        )
        return indexing_config

    def _find_missing_metadata_fields(self, indexing_config: dict[str, Any]) -> set[str]:
        """Inspect the vector store and return metadata fields that are missing."""
        required_fields = set(indexing_config["metadata_fields"])
        existing_fields: set[str] = set()

        try:
            sample = st.session_state.vector_store.collection.peek(1)
            if sample and sample.get("metadatas"):
                metadata_sample = sample["metadatas"][0] or {}
                existing_fields = set(metadata_sample.keys())
        except Exception as peek_error:
            logger.debug("Unable to inspect vector store metadata: %s", peek_error)

        missing_fields = required_fields - existing_fields
        if missing_fields:
            logger.info(
                "Vector store metadata missing fields %s; triggering reindex",
                missing_fields,
            )
        return missing_fields

    def _ensure_llm_client(self) -> None:
        """Initialize the local LLM client when Ollama support is enabled."""
        if st.session_state.llm_client is not None:
            return

        ollama_cfg = self.config.get("ollama", {})
        if not ollama_cfg.get("enabled", False):
            return

        try:
            from ml.llm_interface import LocalLLMInterface

            llm_config = dict(self.config.get("llm", {}))
            llm_config["provider"] = llm_config.get("provider", "ollama")

            model_name = (
                llm_config.pop("model", None)
                or llm_config.get("model_name")
                or ollama_cfg.get("model", "llama3")
            )
            llm_config["model_name"] = model_name

            llm_config.setdefault("endpoint", ollama_cfg.get("api_base", "http://localhost:11434"))
            llm_config.setdefault("temperature", ollama_cfg.get("temperature", 0.7))
            llm_config.setdefault("max_tokens", ollama_cfg.get("max_tokens", 1024))

            st.session_state.llm_client = LocalLLMInterface(llm_config)

            if not st.session_state.llm_client.is_available:
                logger.warning("LLM client initialized but service is unavailable")
        except Exception as llm_error:
            logger.warning("Failed to initialize LLM client: %s", llm_error)

    def render_sidebar(self) -> str:
        """
        Render the sidebar navigation.

        Returns:
            str: Selected page name
        """
        with st.sidebar:
            sidebar_main = st.container()

            with sidebar_main:
                st.title("📞 Call Analytics")
                st.divider()

                # Navigation
                pages = {
                    "Dashboard": "📊",
                    "Upload Data": "📤",
                    "Analysis": "🔍",
                    "Q&A Interface": "💬",
                    "Settings": "⚙️",
                }

                selected_page = st.radio(
                    "Navigation",
                    options=list(pages.keys()),
                    format_func=lambda x: f"{pages[x]} {x}",
                    index=list(pages.keys()).index(st.session_state.current_page),
                    label_visibility="collapsed",
                )
                st.divider()
                st.session_state.current_page = selected_page

        return selected_page

    def _collect_system_status(self) -> list[tuple[str, str]]:
        """Gather formatted system status indicators for display."""

        status_items: list[tuple[str, str]] = []

        storage_manager = st.session_state.get("storage_manager")
        if storage_manager is not None:
            try:
                row_count = storage_manager.get_record_count()
                if row_count > 0:
                    status_items.append((f"✅ {row_count:,} records", "status-ok"))
                else:
                    status_items.append(("⚠️ No records", "status-warn"))
            except Exception as count_error:
                logger.warning(f"Unable to determine record count: {count_error}")
                status_items.append(("ℹ️ Records unavailable", "status-info"))
        else:
            status_items.append(("ℹ️ Storage not ready", "status-info"))

        vector_store = st.session_state.get("vector_store")
        if vector_store is not None:
            status_items.append(("✅ Vector store", "status-ok"))
        else:
            status_items.append(("⚠️ Vector store", "status-warn"))

        whisper_enabled = self.config.get("whisper", {}).get("enabled", False)
        if whisper_enabled:
            status_items.append(("✅ Whisper STT", "status-ok"))
        else:
            status_items.append(("ℹ️ Whisper off", "status-info"))

        ollama_enabled = self.config.get("ollama", {}).get("enabled", False)
        if ollama_enabled:
            status_items.append(("✅ Ollama LLM", "status-ok"))
        else:
            status_items.append(("ℹ️ Ollama off", "status-info"))

        return status_items

    def render_app_footer(self) -> None:
        """Render a persistent footer for the main application area."""

        current_year = datetime.now().year
        version_text = self.config.get("app", {}).get("version", "1.0.0")
        python_version = f"Python {sys.version_info.major}.{sys.version_info.minor}"

        status_badges = self._collect_system_status()
        status_html = (
            "".join(f'<span class="status-chip {cls}">{text}</span>' for text, cls in status_badges)
            or '<span class="status-chip status-info">ℹ️ Status pending</span>'
        )

        footer_links = [
            ("Repository", "https://github.com/mujtaba-a-khan/call-analytics-system"),
            ("Support", "https://github.com/mujtaba-a-khan/call-analytics-system/issues"),
            ("Streamlit Docs", "https://docs.streamlit.io"),
        ]
        links_html = "".join(
            f'<a href="{url}" target="_blank" rel="noopener noreferrer">{label}</a>'
            for label, url in footer_links
        )

        st.markdown(
            f"""
            <div class="app-footer">
                <div class="app-footer__meta">
                    <span>© {current_year} Call Analytics System</span>
                    <span>{python_version} · v{version_text}</span>
                </div>
                <div class="app-footer__status">{status_html}</div>
                <div class="app-footer__links">{links_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
            render_dashboard_page(storage_manager=st.session_state.storage_manager)
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

            render_upload_page(storage_manager=st.session_state.storage_manager, config=self.config)
        except ImportError:
            # Fallback to basic upload
            st.header("📤 Upload Data")
            st.info("Upload components are being loaded...")

            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=["csv"],
                help="Upload a CSV file containing call transcripts",
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
                vector_store=st.session_state.vector_store,
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
                llm_client=st.session_state.llm_client,
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
            python_version = (
                f"{sys.version_info.major}."
                f"{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            )
            st.metric("Python Version", python_version)

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
            self.render_app_footer()

        except Exception as e:
            logger.error(f"Application error: {e}\n{traceback.format_exc()}")
            st.error("An unexpected error occurred. Please check the logs.")

            if st.checkbox("Show error details"):
                st.exception(e)


def main() -> None:
    """Main entry point for the application"""

    try:
        _configure_streamlit_page()
        Path("logs").mkdir(exist_ok=True)
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
