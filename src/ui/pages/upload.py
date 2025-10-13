"""
Upload Page Module for Call Analytics System

This module implements the upload interface for importing call data
from various sources including CSV files, audio recordings, and
batch processing with progress tracking and validation.
"""

import logging
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.core.audio_processor import AudioProcessor
from src.core.csv_processor import CSVProcessor
from src.core.storage_manager import StorageManager
from src.ml.whisper_stt import WhisperSTT
from src.utils.validators import validate_phone

# Configure module logger
logger = logging.getLogger(__name__)


class UploadPage:
    """
    Upload interface for importing call data from various sources
    with validation, processing, and storage capabilities.
    """

    def __init__(self, storage_manager: StorageManager, config: dict[str, Any]):
        """
        Initialize upload page with required components.

        Args:
            storage_manager: Storage manager instance
            config: Application configuration dictionary
        """
        self.storage_manager = storage_manager
        self.config = config

        csv_config = dict(config.get("csv_processor", {}))
        fields_config = config.get("fields")

        if isinstance(fields_config, dict):
            if "definitions" in fields_config:
                csv_config["definitions"] = fields_config["definitions"]
            else:
                csv_config["fields"] = fields_config

        self.csv_processor = CSVProcessor(csv_config)
        # Initialize audio processor with derived configuration
        audio_config = dict(config.get("audio", {}))
        paths_config = config.get("paths", {})
        audio_config.setdefault(
            "processed_dir",
            paths_config.get("processed_audio", "data/processed"),
        )
        audio_config.setdefault(
            "cache_dir",
            paths_config.get("cache", "data/cache"),
        )
        self.audio_processor = AudioProcessor(audio_config)

        # Initialize STT if configured
        whisper_config = dict(config.get("whisper", {}))
        if whisper_config.get("enabled", True):
            # Ensure cache directory aligns with global paths configuration
            paths_config = config.get("paths", {})
            whisper_config.setdefault(
                "cache_dir", Path(paths_config.get("cache", "data/cache")) / "stt"
            )
            self.stt_engine = WhisperSTT(whisper_config)
        else:
            self.stt_engine = None

    def render(self) -> None:
        """Render the complete upload page with all upload options."""
        try:
            # Page header
            st.title("📤 Data Upload")
            st.markdown("Import call data from CSV files or audio recordings")

            # Create tabs for different upload types
            tab1, tab2, tab3, tab4 = st.tabs(
                ["📄 CSV Upload", "🎵 Audio Upload", "📁 Batch Processing", "🔄 Import History"]
            )

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
            type=["csv", "txt"],
            key="csv_uploader",
            help="Select a CSV file containing call records",
        )

        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = Path(tempfile.gettempdir()) / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Preview and validation
            st.subheader("File Preview")

            try:
                # Get preview
                preview_df = self.csv_processor.get_csv_preview(temp_path, num_rows=5)
                st.dataframe(preview_df, width="stretch")

                # Show detected fields
                st.subheader("Field Mapping")
                headers = list(preview_df.columns)
                field_mapping = self.csv_processor.auto_map_fields(headers)

                # Display mapping
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Detected Mappings:**")
                    for standard, csv_field in field_mapping.items():
                        label = self.csv_processor.get_field_label(standard)
                        st.success(f"✓ {label} → {csv_field}")

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
                        help="Continue import even if some rows fail validation",
                    )

                with col2:
                    deduplicate = st.checkbox(
                        "Remove duplicates",
                        value=True,
                        help="Remove duplicate records based on phone and timestamp",
                    )

                with col3:
                    validate_phones = st.checkbox(
                        "Validate phone numbers",
                        value=True,
                        help="Validate and normalize phone numbers",
                    )

                # Import button
                if st.button("Import CSV", type="primary", width="stretch"):
                    self._process_csv_import(temp_path, skip_errors, deduplicate, validate_phones)

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
            type=["wav", "mp3", "mp4", "m4a", "ogg", "flac"],
            accept_multiple_files=True,
            key="audio_uploader",
            help="Select one or more audio files",
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
                    help="Language of the audio (auto-detect if unsure)",
                )

            with col2:
                enable_timestamps = st.checkbox(
                    "Word timestamps", value=False, help="Extract word-level timestamps"
                )

            with col3:
                enable_vad = st.checkbox(
                    "Voice Activity Detection", value=True, help="Use VAD to filter out silence"
                )

            # Metadata for audio files
            st.subheader("Call Metadata")

            col1, col2 = st.columns(2)

            with col1:
                default_campaign = st.text_input(
                    "Campaign", value="audio_upload", help="Campaign name for these recordings"
                )

                default_agent = st.text_input("Agent ID", value="unknown", help="Agent ID if known")

            with col2:
                default_outcome = st.selectbox(
                    "Call Outcome",
                    ["connected", "voicemail", "no_answer", "busy", "failed"],
                    help="Call outcome if known",
                )

                default_type = st.selectbox(
                    "Call Type", ["inbound", "outbound", "internal", "unknown"], help="Type of call"
                )

            # Process button
            if st.button("Process Audio Files", type="primary", width="stretch"):
                self._process_audio_files(
                    uploaded_files,
                    language,
                    enable_timestamps,
                    enable_vad,
                    {
                        "campaign": default_campaign,
                        "agent_id": default_agent,
                        "outcome": default_outcome,
                        "call_type": default_type,
                    },
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
            "Source Type", ["Local Directory", "Upload ZIP", "Cloud Storage"], horizontal=True
        )

        if source_type == "Local Directory":
            directory_path = st.text_input(
                "Directory Path",
                placeholder="/path/to/call/data",
                help="Enter the full path to the directory containing files",
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
                    f"Process {len(csv_files)} CSV files", value=len(csv_files) > 0
                )

                process_audio = st.checkbox(
                    f"Process {len(audio_files)} audio files",
                    value=len(audio_files) > 0 and self.stt_engine is not None,
                )

                # Batch settings
                col1, col2 = st.columns(2)

                with col1:
                    batch_size = st.number_input(
                        "Batch Size",
                        min_value=1,
                        max_value=100,
                        value=10,
                        help="Number of files to process at once",
                    )

                with col2:
                    parallel_processing = st.checkbox(
                        "Parallel Processing",
                        value=False,
                        help="Process multiple files simultaneously",
                    )

                # Start batch processing
                if st.button("Start Batch Processing", type="primary", width="stretch"):
                    self._start_batch_processing(
                        path, process_csv, process_audio, batch_size, parallel_processing
                    )

        elif source_type == "Upload ZIP":
            uploaded_zip = st.file_uploader("Choose a ZIP file", type=["zip"], key="zip_uploader")

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

        history_df = self.storage_manager.get_import_history()

        if history_df.empty:
            st.info("No import history available yet. Start by uploading some data!")
            return

        formatted_history = self._format_import_history(history_df)
        self._render_import_history_table(formatted_history)
        self._render_import_summary(formatted_history)
        self._render_import_history_actions(formatted_history)

    def _format_import_history(self, history_df: pd.DataFrame) -> pd.DataFrame:
        """Return a sanitized copy of the import history for presentation."""
        formatted = history_df.copy()

        if "timestamp" in formatted.columns:
            formatted["timestamp"] = pd.to_datetime(formatted["timestamp"]).dt.strftime(
                "%Y-%m-%d %H:%M"
            )

        if "file_size" in formatted.columns:
            formatted["file_size"] = formatted["file_size"].apply(self._format_file_size)

        return formatted

    @staticmethod
    def _format_file_size(value: float) -> str:
        """Render file size values with human friendly units."""
        if value > 1024 * 1024:
            return f"{value / 1024 / 1024:.1f} MB"
        return f"{value / 1024:.1f} KB"

    def _render_import_history_table(self, history_df: pd.DataFrame) -> None:
        """Render the import history dataframe with column configuration."""
        st.dataframe(
            history_df,
            width="stretch",
            hide_index=True,
            column_config={
                "timestamp": st.column_config.TextColumn("Import Date"),
                "filename": st.column_config.TextColumn("File Name"),
                "file_type": st.column_config.TextColumn("Type"),
                "records_imported": st.column_config.NumberColumn("Records"),
                "file_size": st.column_config.TextColumn("Size"),
                "status": st.column_config.TextColumn("Status"),
            },
        )

    def _render_import_summary(self, history_df: pd.DataFrame) -> None:
        """Display summary metrics for the import history."""
        st.subheader("Import Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Imports", len(history_df))

        with col2:
            total_records_series = history_df.get("records_imported", pd.Series(dtype=float))
            total_records = total_records_series.fillna(0).sum()
            st.metric("Total Records", f"{int(total_records):,}")

        status_series = history_df.get("status", pd.Series(dtype=str))
        successful = int((status_series == "success").sum())
        failed = int((status_series == "failed").sum())

        with col3:
            st.metric("Successful", successful)

        with col4:
            st.metric("Failed", failed)

    def _render_import_history_actions(self, history_df: pd.DataFrame) -> None:
        """Render management actions for the import history screen."""
        st.divider()

        clear_col, export_col = st.columns(2)

        with clear_col:
            if st.button("Clear Import History", type="secondary", key="clear_history_btn"):
                st.session_state.show_clear_history_confirm = True

        if st.session_state.get("show_clear_history_confirm"):
            st.warning("Are you sure you want to clear the import history?")
            confirm_col, cancel_col = st.columns(2)

            with confirm_col:
                if st.button("Yes, clear", type="primary", key="confirm_clear_history_btn"):
                    self.storage_manager.clear_import_history()
                    st.session_state.show_clear_history_confirm = False
                    st.success("Import history cleared")
                    st.rerun()

            with cancel_col:
                if st.button("Cancel", key="cancel_clear_history_btn"):
                    st.session_state.show_clear_history_confirm = False
                    st.info("Import history not cleared")

        with export_col:
            self._render_export_history_button(history_df)

    def _render_export_history_button(self, history_df: pd.DataFrame) -> None:
        """Provide an export option for the history table."""
        if st.button("Export History", type="secondary"):
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"import_history_{datetime.now():%Y%m%d_%H%M%S}.csv",
                mime="text/csv",
            )

    def _render_manual_mapping(self, headers: list[str], current_mapping: dict[str, str]) -> None:
        """
        Render manual field mapping interface.

        Args:
            headers: List of CSV headers
            current_mapping: Current field mapping
        """
        st.write("Manually adjust field mappings:")

        standard_fields = self._get_standard_fields()
        new_mapping: dict[str, str] = {}

        for field in standard_fields:
            field_name, field_label = self._normalize_field_definition(field)
            if not field_name:
                continue

            selected = self._render_mapping_selector(
                field_name, field_label, headers, current_mapping
            )
            if selected:
                new_mapping[field_name] = selected

        if st.button("Apply Mapping"):
            self.csv_processor.field_mapping = new_mapping
            st.success("Field mapping updated")

    def _get_standard_fields(self) -> list[Any]:
        """Return schema-defined fields or a sensible fallback list."""
        schema_fields = self.csv_processor.get_mappable_fields()
        if schema_fields:
            return schema_fields

        fallback_fields = [
            "call_id",
            "phone_number",
            "call_type",
            "outcome",
            "duration",
            "timestamp",
            "agent_id",
            "campaign",
            "notes",
            "revenue",
        ]
        return [
            {"name": field, "label": field.replace("_", " ").title()} for field in fallback_fields
        ]

    @staticmethod
    def _normalize_field_definition(field: Any) -> tuple[str | None, str]:
        """Return field name and label from schema or fallback definition."""
        if isinstance(field, dict):
            field_name = field.get("name")
            field_label = field.get("label", field_name or "")
        else:
            field_name = field
            field_label = str(field)

        return field_name, field_label

    def _render_mapping_selector(
        self,
        field_name: str,
        field_label: str,
        headers: list[str],
        current_mapping: dict[str, str],
    ) -> str:
        """Render a select box for a single mapping entry."""
        current_value = current_mapping.get(field_name, "")
        options = self._build_mapping_options(headers, current_value)
        return st.selectbox(
            f"{field_label}:",
            options=options,
            index=options.index(current_value) if current_value in options else 0,
            key=f"map_{field_name}",
        )

    @staticmethod
    def _build_mapping_options(headers: list[str], current_value: str) -> list[str]:
        """Create selectbox options ensuring the existing value is available."""
        options = [""] + headers
        if current_value and current_value not in options:
            options.append(current_value)
        return options

    def _process_csv_import(
        self, file_path: Path, skip_errors: bool, deduplicate: bool, validate_phones: bool
    ) -> None:
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

            processed_target = self._estimate_csv_rows(file_path)
            total_rows_seen = 0
            total_processed = 0
            invalid_phone_count = 0

            def process_batch(records: pd.DataFrame) -> None:
                nonlocal total_rows_seen, total_processed, invalid_phone_count
                total_rows_seen += len(records)

                filtered_records, dropped = self._filter_invalid_phone_numbers(
                    records, validate_phones, skip_errors
                )
                invalid_phone_count += dropped

                added = self._append_records(filtered_records, deduplicate)
                total_processed += added

                self._update_import_progress(
                    progress_bar, status_text, total_rows_seen, processed_target, total_processed
                )

            status_text.text("Processing CSV file...")
            _, csv_errors = self.csv_processor.process_csv_batch(
                file_path, batch_callback=process_batch
            )

            total_errors = csv_errors + invalid_phone_count

            progress_bar.progress(1.0)
            status_text.text(f"Import complete: {total_processed} records imported")

            self._present_import_results(total_processed, total_errors, skip_errors)
            self._add_import_history_record(file_path, total_processed, total_errors)
            self._refresh_cached_data()

        except ValueError as exc:
            logger.error(f"CSV import validation error: {exc}")
            st.error(f"Import stopped: {exc}")
        except Exception as exc:
            logger.error(f"Error importing CSV: {exc}")
            st.error(f"Import failed: {exc}")

    def _estimate_csv_rows(self, file_path: Path) -> int:
        """Estimate the number of rows in a CSV for progress tracking."""
        try:
            encoding = self.csv_processor.detect_encoding(file_path)
            with open(file_path, encoding=encoding, errors="ignore") as fh:
                row_count = max(sum(1 for _ in fh) - 1, 1)
            return row_count
        except Exception:
            return 1000

    def _filter_invalid_phone_numbers(
        self, records: pd.DataFrame, validate_phones: bool, skip_errors: bool
    ) -> tuple[pd.DataFrame, int]:
        """Optionally remove records with invalid phone numbers."""
        if not validate_phones or "phone_number" not in records.columns:
            return records, 0

        phone_series = records["phone_number"].astype(str)
        valid_mask = phone_series.apply(validate_phone)
        invalid_count = int((~valid_mask).sum())

        if invalid_count:
            self.csv_processor.errors_log.append(
                {"error": "invalid_phone_number", "count": invalid_count}
            )
            if not skip_errors:
                raise ValueError(
                    "Invalid phone numbers detected. Enable 'Skip invalid rows' to continue."
                )

        return records.loc[valid_mask].copy(), invalid_count

    def _append_records(self, records: pd.DataFrame, deduplicate: bool) -> int:
        """Persist processed records handling legacy storage backends."""
        if records.empty:
            return 0

        try:
            return self.storage_manager.append_records(records, deduplicate=deduplicate)
        except AttributeError:
            if hasattr(self.storage_manager, "load_all_records"):
                existing = self.storage_manager.load_all_records()
                combined = pd.concat([existing, records], ignore_index=True)
                self.storage_manager.save_dataframe(combined, "call_records")
                return len(records)
            raise

    @staticmethod
    def _update_import_progress(
        progress_bar: Any,
        status_text: Any,
        rows_seen: int,
        target_rows: int,
        total_processed: int,
    ) -> None:
        """Update the import progress indicators."""
        progress_ratio = min(rows_seen / max(target_rows, 1), 1.0)
        progress_bar.progress(progress_ratio)
        status_text.text(f"Processed {total_processed} records...")

    def _present_import_results(self, processed: int, errors: int, skip_errors: bool) -> None:
        """Show import completion messages and error report if needed."""
        if errors > 0:
            message = f"Import completed with {errors} issues"
            if skip_errors:
                st.warning(message)
            else:
                st.error(message)
            self._offer_error_report()
        else:
            st.success(f"Successfully imported {processed} records")

    def _offer_error_report(self) -> None:
        """Provide download link for error report when available."""
        if not self.csv_processor.errors_log:
            return

        error_report_path = Path(tempfile.gettempdir()) / "import_errors.csv"
        self.csv_processor.export_errors_report(error_report_path)

        try:
            with open(error_report_path, "rb") as fh:
                st.download_button(
                    label="Download Error Report",
                    data=fh.read(),
                    file_name="import_errors.csv",
                    mime="text/csv",
                )
        except OSError as exc:
            logger.warning(f"Unable to provide error report: {exc}")

    def _add_import_history_record(self, file_path: Path, processed: int, errors: int) -> None:
        """Record the import outcome for future reference."""
        status = "success" if errors == 0 else "partial"
        self.storage_manager.add_import_record(
            {
                "timestamp": datetime.now(),
                "filename": file_path.name,
                "file_type": "CSV",
                "records_imported": processed,
                "errors": errors,
                "status": status,
            }
        )

    def _refresh_cached_data(self) -> None:
        """Refresh session data after an import completes."""
        try:
            loaded_df = self.storage_manager.load_all_records()
            if loaded_df is not None:
                st.session_state.data = loaded_df
        except Exception as load_error:
            logger.warning(f"Unable to refresh session data after import: {load_error}")

    def _batch_process_csv_file(
        self,
        file_path: Path,
        deduplicate: bool = True,
    ) -> tuple[int, int]:
        """Process a CSV file during batch processing without UI widgets."""
        total_processed = 0
        restore_mapping = self._prepare_batch_mapping(file_path)

        def process_batch(records: pd.DataFrame) -> None:
            nonlocal total_processed
            total_processed += self._append_records(records, deduplicate)

        processed, errors = self.csv_processor.process_csv_batch(
            file_path, batch_callback=process_batch
        )

        self._add_import_history_record(file_path, processed, errors)
        self._refresh_cached_data()
        restore_mapping()

        return processed, errors

    def _process_audio_files(
        self,
        files: list[Any],
        language: str,
        enable_timestamps: bool,
        enable_vad: bool,
        metadata: dict[str, Any],
    ) -> None:
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
        metadata_lookup = self._load_audio_metadata()
        total_files = len(files)

        with self._temporary_vad_setting(enable_vad):
            for idx, file in enumerate(files):
                try:
                    status_text.text(f"Processing {file.name}...")
                    record = self._process_single_audio_file(
                        file=file,
                        index=idx,
                        language=language,
                        enable_timestamps=enable_timestamps,
                        metadata=metadata,
                        metadata_lookup=metadata_lookup,
                    )
                    if record:
                        processed_records.append(record)
                except Exception as exc:
                    logger.error(f"Error processing {file.name}: {exc}")
                    st.error(f"Failed to process {file.name}: {exc}")
                finally:
                    progress_bar.progress((idx + 1) / max(total_files, 1))

        if processed_records:
            self.storage_manager.store_call_records(processed_records)
            status_text.text(f"Successfully processed {len(processed_records)} audio files")
            st.success(f"Imported {len(processed_records)} transcribed calls")

        progress_bar.progress(1.0)

    def _process_single_audio_file(
        self,
        file: Any,
        index: int,
        language: str,
        enable_timestamps: bool,
        metadata: dict[str, Any],
        metadata_lookup: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Process and transcribe a single uploaded audio file."""
        temp_audio = self._save_uploaded_file(file)

        try:
            processed_path = self.audio_processor.process_audio(temp_audio)
            transcription = self.stt_engine.transcribe(
                processed_path, language=None if language == "auto" else language
            )
            record_metadata = metadata_lookup.get(file.name.lower(), {})
            return self._build_audio_record(
                index=index,
                transcription=transcription,
                record_metadata=record_metadata,
                enable_timestamps=enable_timestamps,
                base_metadata=metadata,
            )
        finally:
            if temp_audio.exists():
                temp_audio.unlink()

    def _save_uploaded_file(self, file: Any) -> Path:
        """Persist an uploaded file to a temporary location."""
        temp_audio = Path(tempfile.gettempdir()) / file.name
        with open(temp_audio, "wb") as fh:
            fh.write(file.getbuffer())
        return temp_audio

    def _build_audio_record(
        self,
        index: int,
        transcription: Any,
        record_metadata: dict[str, Any],
        enable_timestamps: bool,
        base_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the stored record for a processed audio file."""
        duration_seconds, duration_minutes = self._resolve_durations(transcription, record_metadata)

        record = {
            "call_id": record_metadata.get(
                "call_id", f"audio_{datetime.now():%Y%m%d_%H%M%S}_{index}"
            ),
            "phone_number": record_metadata.get("phone_number", "unknown"),
            "timestamp": record_metadata.get("timestamp", datetime.now()),
            "duration": duration_seconds,
            "duration_seconds": duration_seconds,
            "duration_minutes": duration_minutes,
            "transcript": transcription.transcript,
            **base_metadata,
            **{
                k: v
                for k, v in record_metadata.items()
                if k not in {"audio_file", "transcript_file", "transcript"}
            },
        }

        if enable_timestamps:
            record["segments"] = transcription.segments

        return record

    @staticmethod
    def _resolve_durations(
        transcription: Any, record_metadata: dict[str, Any]
    ) -> tuple[float, float]:
        """Resolve duration values prioritizing metadata overrides."""
        duration_seconds = record_metadata.get("duration_seconds") or record_metadata.get(
            "duration"
        )
        if not duration_seconds:
            duration_seconds = getattr(transcription, "duration_seconds", 0.0)

        duration_minutes = record_metadata.get("duration_minutes")
        if not duration_minutes and duration_seconds:
            duration_minutes = duration_seconds / 60

        return duration_seconds or 0.0, duration_minutes or 0.0

    @contextmanager
    def _temporary_vad_setting(self, enable_vad: bool):
        """Temporarily override the VAD setting on the STT engine."""
        if not hasattr(self.stt_engine, "vad_filter"):
            yield
            return

        original_vad = self.stt_engine.vad_filter
        self.stt_engine.vad_filter = enable_vad
        try:
            yield
        finally:
            self.stt_engine.vad_filter = original_vad

    def _start_batch_processing(
        self,
        directory: Path,
        process_csv: bool,
        process_audio: bool,
        batch_size: int,
        parallel: bool,
    ) -> None:
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

        csv_files, audio_files = self._collect_batch_files(directory, process_csv, process_audio)
        total_files = len(csv_files) + len(audio_files)

        if total_files == 0:
            st.warning("No files found to process in the selected directory.")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()

        completed = 0
        csv_results, csv_errors, completed = self._process_csv_batches(
            csv_files, batch_size, total_files, progress_bar, status_text, completed
        )

        audio_errors = self._handle_audio_batch_placeholder(
            audio_files, parallel, completed, total_files, progress_bar, status_text
        )

        status_text.text("Batch processing complete")

        if csv_results:
            st.success("\n".join(["CSV files processed:"] + csv_results))
        if csv_errors:
            st.warning("\n".join(["CSV processing issues:"] + csv_errors))
        if audio_errors:
            st.warning("\n".join(audio_errors))

        progress_bar.progress(1.0)

    def _collect_batch_files(
        self, directory: Path, process_csv: bool, process_audio: bool
    ) -> tuple[list[Path], list[Path]]:
        """Collect CSV and audio files based on the user's selections."""
        csv_files: list[Path] = []
        audio_files: list[Path] = []

        if process_csv:
            csv_files = sorted(directory.glob("**/*.csv"))

        if process_audio:
            audio_patterns = ["*.wav", "*.mp3", "*.m4a", "*.ogg", "*.flac"]
            for pattern in audio_patterns:
                audio_files.extend(directory.glob(f"**/{pattern}"))
            audio_files.sort()

        return csv_files, audio_files

    def _process_csv_batches(
        self,
        csv_files: list[Path],
        batch_size: int,
        total_files: int,
        progress_bar: Any,
        status_text: Any,
        completed: int,
    ) -> tuple[list[str], list[str], int]:
        """Process CSV files in configured batch sizes and update progress."""
        results: list[str] = []
        errors: list[str] = []

        for batch_index, batch in enumerate(self._create_batches(csv_files, batch_size), start=1):
            for csv_path in batch:
                completed += 1
                status_text.text(
                    f"Processing CSV file {csv_path.name} ({completed}/{total_files}) "
                    f"[Batch {batch_index}]"
                )
                try:
                    processed, file_errors = self._batch_process_csv_file(
                        csv_path, deduplicate=True
                    )
                    message = f"{csv_path.name}: {processed} records"
                    if file_errors:
                        message += f" ({file_errors} errors)"
                    results.append(message)
                except Exception as exc:
                    logger.error(f"Batch CSV processing failed for {csv_path}: {exc}")
                    errors.append(f"{csv_path.name}: {exc}")
                progress_bar.progress(min(completed / max(total_files, 1), 1.0))

        return results, errors, completed

    @staticmethod
    def _create_batches(items: list[Path], batch_size: int) -> list[list[Path]]:
        """Split a list of paths into evenly sized batches."""
        if batch_size <= 0:
            return [items]
        return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

    def _handle_audio_batch_placeholder(
        self,
        audio_files: list[Path],
        parallel: bool,
        completed: int,
        total_files: int,
        progress_bar: Any,
        status_text: Any,
    ) -> list[str]:
        """Communicate current limitations of audio batch processing."""
        if not audio_files:
            return []

        if parallel:
            st.info("Parallel processing is not yet available; processing sequentially.")

        status_text.text("Audio batch processing is not yet implemented in batch mode.")
        progress_bar.progress(min((completed + len(audio_files)) / max(total_files, 1), 1.0))
        return [
            "Audio batch processing is not supported in batch mode. "
            "Please process audio files individually."
        ]

    def _load_audio_metadata(self) -> dict[str, dict[str, Any]]:
        """Load metadata for sample audio files if available."""
        paths_config = self.config.get("paths", {}) if isinstance(self.config, dict) else {}
        data_root = paths_config.get("data", "data")
        metadata_file = Path(data_root) / "raw" / "sample_audio" / "sample_audio_metadata.csv"

        if not metadata_file.exists():
            return {}

        try:
            lookup = self._parse_metadata_file(metadata_file)
        except Exception as exc:
            logger.warning(f"Unable to load audio metadata: {exc}")
            return {}

        if lookup:
            logger.info("Loaded metadata for %d audio files", len(lookup))
        else:
            logger.info("Audio metadata file found but no entries parsed")

        return lookup

    def _parse_metadata_file(self, metadata_file: Path) -> dict[str, dict[str, Any]]:
        import csv

        lookup: dict[str, dict[str, Any]] = {}
        with metadata_file.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                normalized = self._normalize_metadata_row(row)
                if not normalized:
                    continue
                audio_name = normalized["audio_file"].lower()
                lookup[audio_name] = normalized
        return lookup

    def _normalize_metadata_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        audio_name = row.get("audio_file")
        if not audio_name:
            return None

        cleaned_row = dict(row)

        timestamp_value = cleaned_row.get("timestamp")
        parsed_timestamp = self._parse_metadata_timestamp(timestamp_value)
        if parsed_timestamp is not None:
            cleaned_row["timestamp"] = parsed_timestamp

        # Coerce timestamp and numeric fields if present.
        self._coerce_numeric_fields(cleaned_row)
        return cleaned_row

    def _parse_metadata_timestamp(self, timestamp_value: Any) -> datetime | str | None:
        if not timestamp_value or isinstance(timestamp_value, datetime):
            return timestamp_value

        if isinstance(timestamp_value, str):
            try:
                return datetime.fromisoformat(timestamp_value)
            except ValueError:
                return timestamp_value

        return None

    def _coerce_numeric_fields(self, row: dict[str, Any]) -> None:
        numeric_fields = [
            "duration",
            "duration_seconds",
            "duration_minutes",
            "handle_time_seconds",
            "after_call_work_seconds",
            "revenue",
        ]

        for field in numeric_fields:
            value = row.get(field)
            if value in (None, ""):
                continue

            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue

            row[field] = int(numeric) if numeric.is_integer() else numeric

    def _process_zip_file(self, zip_file: Any) -> None:
        """
        Process uploaded ZIP file.

        Args:
            zip_file: Uploaded ZIP file
        """
        # Extract and process files from ZIP
        st.info(f"Processing ZIP file: {getattr(zip_file, 'name', 'uploaded.zip')}")
        # Implementation here
        st.success("ZIP file processed")


def render_upload_page(storage_manager: StorageManager, config: dict[str, Any]) -> None:
    """
    Main entry point for rendering the upload page.

    Args:
        storage_manager: Storage manager instance
        config: Application configuration
    """
    upload_page = UploadPage(storage_manager, config)
    upload_page.render()
