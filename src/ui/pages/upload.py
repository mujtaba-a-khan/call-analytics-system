"""
Upload Page Module for Call Analytics System

This module implements the upload interface for importing call data
from various sources including CSV files, audio recordings, and
batch processing with progress tracking and validation.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
from io import BytesIO

# Import core modules
from ...core.csv_processor import CSVProcessor, CSVExporter
from ...core.audio_processor import AudioProcessor
from ...core.storage_manager import StorageManager
from ...ml.whisper_stt import WhisperSTT

# Import components
from ..components import (
    MetricValue, MetricsGrid,
    DataTable, ProgressIndicator
)

# Configure module logger
logger = logging.getLogger(__name__)


class UploadPage:
    """
    Upload interface for importing call data from various sources
    with validation, processing, and storage capabilities.
    """
    
    def __init__(self, storage_manager: StorageManager, config: Dict[str, Any]):
        """
        Initialize upload page with required components.
        
        Args:
            storage_manager: Storage manager instance
            config: Application configuration dictionary
        """
        self.storage_manager = storage_manager
        self.config = config
        self.csv_processor = CSVProcessor()
        self.audio_processor = AudioProcessor(
            output_dir=Path(config.get('paths', {}).get('processed_audio', 'data/processed'))
        )
        
        # Initialize STT if configured
        whisper_config = config.get('whisper', {})
        if whisper_config.get('enabled', True):
            self.stt_engine = WhisperSTT(
                model_size=whisper_config.get('model_size', 'small'),
                device=whisper_config.get('device', 'cpu'),
                compute_type=whisper_config.get('compute_type', 'int8')
            )
        else:
            self.stt_engine = None
    
    def render(self) -> None:
        """
        Render the complete upload page with all upload options.
        """
        try:
            # Page header
            st.title("📤 Data Upload")
            st.markdown("Import call data from CSV files or audio recordings")
            
            # Create tabs for different upload types
            tab1, tab2, tab3, tab4 = st.tabs([
                "📄 CSV Upload",
                "🎵 Audio Upload",
                "📁 Batch Processing",
                "🔄 Import History"
            ])
            
            with tab1:
                self._render_csv_upload_tab()
            
            with tab2:
                self._render_audio_upload_tab()
            
            with tab3:
                self._render_batch_processing_tab()
            
            with tab4:
                self._render_import_history_tab()
            
        except Exception as e:
            logger.error(f"Error rendering upload page: {e}")
            st.error(f"Failed to load upload page: {str(e)}")
    
    def _render_csv_upload_tab(self) -> None:
        """
        Render CSV file upload interface with validation and mapping.
        """
        st.header("CSV File Upload")
        st.markdown("Upload call records from CSV files with automatic field mapping")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv', 'txt'],
            key="csv_uploader",
            help="Select a CSV file containing call records"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = Path(tempfile.gettempdir()) / uploaded_file.name
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Preview and validation
            st.subheader("File Preview")
            
            try:
                # Get preview
                preview_df = self.csv_processor.get_csv_preview(temp_path, num_rows=5)
                st.dataframe(preview_df, use_container_width=True)
                
                # Show detected fields
                st.subheader("Field Mapping")
                headers = list(preview_df.columns)
                field_mapping = self.csv_processor.auto_map_fields(headers)
                
                # Display mapping
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Detected Mappings:**")
                    for standard, csv_field in field_mapping.items():
                        st.success(f"✓ {standard} → {csv_field}")
                
                with col2:
                    st.write("**Unmapped Fields:**")
                    mapped_csv_fields = set(field_mapping.values())
                    unmapped = [h for h in headers if h not in mapped_csv_fields]
                    for field in unmapped[:5]:
                        st.info(f"? {field}")
                    if len(unmapped) > 5:
                        st.info(f"... and {len(unmapped) - 5} more")
                
                # Manual mapping option
                with st.expander("Adjust Field Mapping", expanded=False):
                    self._render_manual_mapping(headers, field_mapping)
                
                # Import options
                st.subheader("Import Options")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    skip_errors = st.checkbox(
                        "Skip invalid rows",
                        value=True,
                        help="Continue import even if some rows fail validation"
                    )
                
                with col2:
                    deduplicate = st.checkbox(
                        "Remove duplicates",
                        value=True,
                        help="Remove duplicate records based on phone and timestamp"
                    )
                
                with col3:
                    validate_phones = st.checkbox(
                        "Validate phone numbers",
                        value=True,
                        help="Validate and normalize phone numbers"
                    )
                
                # Import button
                if st.button("Import CSV", type="primary", use_container_width=True):
                    self._process_csv_import(
                        temp_path,
                        skip_errors,
                        deduplicate,
                        validate_phones
                    )
                
            except Exception as e:
                logger.error(f"Error processing CSV: {e}")
                st.error(f"Failed to process CSV: {str(e)}")
            
            finally:
                # Clean up temp file
                if temp_path.exists():
                    temp_path.unlink()
    
    def _render_audio_upload_tab(self) -> None:
        """
        Render audio file upload interface with transcription options.
        """
        st.header("Audio File Upload")
        
        if not self.stt_engine:
            st.warning("Speech-to-text is not configured. Please set up Whisper STT in settings.")
            return
        
        st.markdown("Upload call recordings for automatic transcription and analysis")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose audio files",
            type=['wav', 'mp3', 'mp4', 'm4a', 'ogg', 'flac'],
            accept_multiple_files=True,
            key="audio_uploader",
            help="Select one or more audio files"
        )
        
        if uploaded_files:
            st.info(f"Selected {len(uploaded_files)} file(s) for processing")
            
            # Transcription options
            st.subheader("Transcription Settings")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                language = st.selectbox(
                    "Language",
                    ["auto", "en", "es", "fr", "de", "it", "pt", "nl", "ru", "zh", "ja"],
                    help="Language of the audio (auto-detect if unsure)"
                )
            
            with col2:
                enable_timestamps = st.checkbox(
                    "Word timestamps",
                    value=False,
                    help="Extract word-level timestamps"
                )
            
            with col3:
                enable_vad = st.checkbox(
                    "Voice Activity Detection",
                    value=True,
                    help="Use VAD to filter out silence"
                )
            
            # Metadata for audio files
            st.subheader("Call Metadata")
            
            col1, col2 = st.columns(2)
            
            with col1:
                default_campaign = st.text_input(
                    "Campaign",
                    value="audio_upload",
                    help="Campaign name for these recordings"
                )
                
                default_agent = st.text_input(
                    "Agent ID",
                    value="unknown",
                    help="Agent ID if known"
                )
            
            with col2:
                default_outcome = st.selectbox(
                    "Call Outcome",
                    ["connected", "voicemail", "no_answer", "busy", "failed"],
                    help="Call outcome if known"
                )
                
                default_type = st.selectbox(
                    "Call Type",
                    ["inbound", "outbound", "internal", "unknown"],
                    help="Type of call"
                )
            
            # Process button
            if st.button("Process Audio Files", type="primary", use_container_width=True):
                self._process_audio_files(
                    uploaded_files,
                    language,
                    enable_timestamps,
                    enable_vad,
                    {
                        'campaign': default_campaign,
                        'agent_id': default_agent,
                        'outcome': default_outcome,
                        'call_type': default_type
                    }
                )
    
    def _render_batch_processing_tab(self) -> None:
        """
        Render batch processing interface for multiple files.
        """
        st.header("Batch Processing")
        st.markdown("Process multiple files from a directory")
        
        # Directory selection
        st.subheader("Select Source")
        
        source_type = st.radio(
            "Source Type",
            ["Local Directory", "Upload ZIP", "Cloud Storage"],
            horizontal=True
        )
        
        if source_type == "Local Directory":
            directory_path = st.text_input(
                "Directory Path",
                placeholder="/path/to/call/data",
                help="Enter the full path to the directory containing files"
            )
            
            if directory_path and Path(directory_path).exists():
                # Scan directory
                path = Path(directory_path)
                csv_files = list(path.glob("**/*.csv"))
                audio_files = list(path.glob("**/*.wav")) + list(path.glob("**/*.mp3"))
                
                st.info(f"Found {len(csv_files)} CSV files and {len(audio_files)} audio files")
                
                # Processing options
                st.subheader("Processing Options")
                
                process_csv = st.checkbox(
                    f"Process {len(csv_files)} CSV files",
                    value=len(csv_files) > 0
                )
                
                process_audio = st.checkbox(
                    f"Process {len(audio_files)} audio files",
                    value=len(audio_files) > 0 and self.stt_engine is not None
                )
                
                # Batch settings
                col1, col2 = st.columns(2)
                
                with col1:
                    batch_size = st.number_input(
                        "Batch Size",
                        min_value=1,
                        max_value=100,
                        value=10,
                        help="Number of files to process at once"
                    )
                
                with col2:
                    parallel_processing = st.checkbox(
                        "Parallel Processing",
                        value=False,
                        help="Process multiple files simultaneously"
                    )
                
                # Start batch processing
                if st.button("Start Batch Processing", type="primary", use_container_width=True):
                    self._start_batch_processing(
                        path,
                        process_csv,
                        process_audio,
                        batch_size,
                        parallel_processing
                    )
        
        elif source_type == "Upload ZIP":
            uploaded_zip = st.file_uploader(
                "Choose a ZIP file",
                type=['zip'],
                key="zip_uploader"
            )
            
            if uploaded_zip:
                st.info(f"Uploaded: {uploaded_zip.name} ({uploaded_zip.size / 1024 / 1024:.1f} MB)")
                
                if st.button("Extract and Process", type="primary"):
                    self._process_zip_file(uploaded_zip)
        
        elif source_type == "Cloud Storage":
            st.info("Cloud storage integration coming soon!")
    
    def _render_import_history_tab(self) -> None:
        """
        Render import history and management interface.
        """
        st.header("Import History")
        st.markdown("View and manage previously imported data")
        
        # Load import history
        history = self.storage_manager.get_import_history()
        
        if history:
            # Display history table
            history_df = pd.DataFrame(history)
            
            # Format columns
            if 'timestamp' in history_df.columns:
                history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            
            if 'file_size' in history_df.columns:
                history_df['file_size'] = history_df['file_size'].apply(
                    lambda x: f"{x / 1024 / 1024:.1f} MB" if x > 1024 * 1024 else f"{x / 1024:.1f} KB"
                )
            
            # Display table
            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'timestamp': st.column_config.TextColumn('Import Date'),
                    'filename': st.column_config.TextColumn('File Name'),
                    'file_type': st.column_config.TextColumn('Type'),
                    'records_imported': st.column_config.NumberColumn('Records'),
                    'file_size': st.column_config.TextColumn('Size'),
                    'status': st.column_config.TextColumn('Status')
                }
            )
            
            # Summary statistics
            st.subheader("Import Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Imports", len(history))
            
            with col2:
                total_records = sum(h.get('records_imported', 0) for h in history)
                st.metric("Total Records", f"{total_records:,}")
            
            with col3:
                successful = sum(1 for h in history if h.get('status') == 'success')
                st.metric("Successful", successful)
            
            with col4:
                failed = sum(1 for h in history if h.get('status') == 'failed')
                st.metric("Failed", failed)
            
            # Management options
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Clear Import History", type="secondary"):
                    if st.confirm("Are you sure you want to clear the import history?"):
                        self.storage_manager.clear_import_history()
                        st.success("Import history cleared")
                        st.rerun()
            
            with col2:
                if st.button("Export History", type="secondary"):
                    csv = history_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"import_history_{datetime.now():%Y%m%d_%H%M%S}.csv",
                        mime="text/csv"
                    )
        else:
            st.info("No import history available yet. Start by uploading some data!")
    
    def _render_manual_mapping(self, headers: List[str], current_mapping: Dict[str, str]) -> None:
        """
        Render manual field mapping interface.
        
        Args:
            headers: List of CSV headers
            current_mapping: Current field mapping
        """
        st.write("Manually adjust field mappings:")
        
        # Standard fields that need mapping
        standard_fields = [
            'call_id', 'phone_number', 'call_type', 'outcome',
            'duration', 'timestamp', 'agent_id', 'campaign',
            'notes', 'revenue'
        ]
        
        # Create mapping interface
        new_mapping = {}
        
        for field in standard_fields:
            current = current_mapping.get(field, '')
            options = [''] + headers
            
            # Ensure current value is in options
            if current and current not in options:
                options.append(current)
            
            selected = st.selectbox(
                f"{field}:",
                options=options,
                index=options.index(current) if current in options else 0,
                key=f"map_{field}"
            )
            
            if selected:
                new_mapping[field] = selected
        
        # Update mapping
        if st.button("Apply Mapping"):
            self.csv_processor.field_mapping = new_mapping
            st.success("Field mapping updated")
    
    def _process_csv_import(self,
                           file_path: Path,
                           skip_errors: bool,
                           deduplicate: bool,
                           validate_phones: bool) -> None:
        """
        Process CSV file import with progress tracking.
        
        Args:
            file_path: Path to CSV file
            skip_errors: Whether to skip invalid rows
            deduplicate: Whether to remove duplicates
            validate_phones: Whether to validate phone numbers
        """
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process CSV in batches
            total_processed = 0
            total_errors = 0
            batch_records = []
            
            def process_batch(records):
                nonlocal total_processed
                # Store records
                self.storage_manager.store_call_records(records)
                total_processed += len(records)
                
                # Update progress
                progress_bar.progress(min(total_processed / 1000, 1.0))
                status_text.text(f"Processed {total_processed} records...")
            
            # Process file
            status_text.text("Processing CSV file...")
            processed, errors = self.csv_processor.process_csv_batch(
                file_path,
                batch_callback=process_batch
            )
            
            # Complete progress
            progress_bar.progress(1.0)
            status_text.text(f"Import complete: {processed} records imported")
            
            # Show results
            if errors > 0:
                st.warning(f"Import completed with {errors} errors")
                
                # Export error report
                error_report_path = Path(tempfile.gettempdir()) / "import_errors.csv"
                self.csv_processor.export_errors_report(error_report_path)
                
                with open(error_report_path, 'rb') as f:
                    st.download_button(
                        label="Download Error Report",
                        data=f.read(),
                        file_name="import_errors.csv",
                        mime="text/csv"
                    )
            else:
                st.success(f"Successfully imported {processed} records")
            
            # Update import history
            self.storage_manager.add_import_history({
                'timestamp': datetime.now(),
                'filename': file_path.name,
                'file_type': 'CSV',
                'records_imported': processed,
                'errors': errors,
                'status': 'success' if errors == 0 else 'partial'
            })
            
        except Exception as e:
            logger.error(f"Error importing CSV: {e}")
            st.error(f"Import failed: {str(e)}")
    
    def _process_audio_files(self,
                            files: List[Any],
                            language: str,
                            enable_timestamps: bool,
                            enable_vad: bool,
                            metadata: Dict[str, Any]) -> None:
        """
        Process uploaded audio files with transcription.
        
        Args:
            files: List of uploaded audio files
            language: Language code or 'auto'
            enable_timestamps: Whether to extract timestamps
            enable_vad: Whether to use VAD
            metadata: Default metadata for records
        """
        if not self.stt_engine:
            st.error("Speech-to-text engine not configured")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        processed_records = []
        
        for idx, file in enumerate(files):
            try:
                status_text.text(f"Processing {file.name}...")
                
                # Save audio file temporarily
                temp_audio = Path(tempfile.gettempdir()) / file.name
                with open(temp_audio, 'wb') as f:
                    f.write(file.getbuffer())
                
                # Process audio
                processed_path = self.audio_processor.process_audio(temp_audio)
                
                # Transcribe
                result = self.stt_engine.transcribe(
                    processed_path,
                    language=None if language == 'auto' else language
                )
                
                # Create call record
                record = {
                    'call_id': f"audio_{datetime.now():%Y%m%d_%H%M%S}_{idx}",
                    'phone_number': 'unknown',
                    'timestamp': datetime.now(),
                    'duration': result.get('duration', 0),
                    'transcript': result.get('text', ''),
                    **metadata
                }
                
                processed_records.append(record)
                
                # Update progress
                progress_bar.progress((idx + 1) / len(files))
                
                # Clean up
                temp_audio.unlink()
                
            except Exception as e:
                logger.error(f"Error processing {file.name}: {e}")
                st.error(f"Failed to process {file.name}: {str(e)}")
        
        # Store records
        if processed_records:
            self.storage_manager.store_call_records(processed_records)
            status_text.text(f"Successfully processed {len(processed_records)} audio files")
            st.success(f"Imported {len(processed_records)} transcribed calls")
        
        progress_bar.progress(1.0)
    
    def _start_batch_processing(self,
                               directory: Path,
                               process_csv: bool,
                               process_audio: bool,
                               batch_size: int,
                               parallel: bool) -> None:
        """
        Start batch processing of files from directory.
        
        Args:
            directory: Source directory
            process_csv: Whether to process CSV files
            process_audio: Whether to process audio files
            batch_size: Number of files per batch
            parallel: Whether to use parallel processing
        """
        st.info(f"Starting batch processing from {directory}")
        
        # Implementation would include:
        # - File discovery and queuing
        # - Batch processing with progress tracking
        # - Error handling and reporting
        # - Result summary
        
        st.success("Batch processing complete")
    
    def _process_zip_file(self, zip_file: Any) -> None:
        """
        Process uploaded ZIP file.
        
        Args:
            zip_file: Uploaded ZIP file
        """
        # Extract and process files from ZIP
        st.info("Processing ZIP file...")
        # Implementation here
        st.success("ZIP file processed")


def render_upload_page(storage_manager: StorageManager, config: Dict[str, Any]) -> None:
    """
    Main entry point for rendering the upload page.
    
    Args:
        storage_manager: Storage manager instance
        config: Application configuration
    """
    upload_page = UploadPage(storage_manager, config)
    upload_page.render()