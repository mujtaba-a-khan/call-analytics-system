"""
CSV Processing Module

Handles importing, validation, processing, and exporting of CSV files
containing call transcripts and metadata.
"""

import json
import logging
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import chardet
import pandas as pd

logger = logging.getLogger(__name__)


class CSVProcessor:
    """
    Processes CSV files containing call data.
    Handles various formats and encodings, validates data,
    and standardizes output format.
    """

    def __init__(self, config: dict | None = None):
        """
        Initialize the CSV processor with configuration.

        Args:
            config: Configuration dictionary with CSV settings
        """
        config = config or {}

        self.encoding = config.get("encoding", "utf-8")

        default_required = ["call_id", "timestamp", "duration", "transcript"]
        default_optional = [
            "agent_id",
            "campaign",
            "customer_name",
            "phone_number",
            "product_name",
            "amount",
            "revenue",
            "outcome",
            "call_type",
            "notes",
        ]

        schema_definitions = self._extract_field_schema(config)

        explicit_required = config.get("required_columns")
        explicit_optional = config.get("optional_columns")

        derived_required = [
            field.get("name")
            for field in schema_definitions
            if isinstance(field, dict) and field.get("required", False)
        ]
        derived_required = [name for name in derived_required if name]

        derived_optional = [
            field.get("name")
            for field in schema_definitions
            if isinstance(field, dict) and not field.get("required", False)
        ]
        derived_optional = [name for name in derived_optional if name]

        base_optional = explicit_optional or default_optional
        if derived_optional:
            base_optional = [col for col in base_optional if col not in derived_required]
            base_optional = derived_optional + base_optional

        self.required_columns = explicit_required or derived_required or default_required
        self.optional_columns = list(dict.fromkeys(base_optional))
        self.field_definitions = schema_definitions
        self.field_labels = {
            field["name"]: field.get("label", field["name"])
            for field in self.field_definitions
            if isinstance(field, dict) and field.get("name")
        }
        self.field_alias_map = {
            field["name"]: field.get("aliases", [])
            for field in self.field_definitions
            if isinstance(field, dict) and field.get("name")
        }
        self.field_pattern_map = {
            field["name"]: field.get("patterns", [])
            for field in self.field_definitions
            if isinstance(field, dict) and field.get("name")
        }
        self.date_formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
        ]

        # Field mapping for auto-detection
        self.field_mapping = {}
        self.errors_log = []

        logger.info("CSVProcessor initialized")

    @staticmethod
    def _extract_field_schema(config: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract field schema definitions from configuration."""
        if not config:
            return []

        if isinstance(config, dict):
            if isinstance(config.get("definitions"), list):
                return config["definitions"]

            fields_section = config.get("fields")
            if isinstance(fields_section, dict) and isinstance(
                fields_section.get("definitions"), list
            ):
                return fields_section["definitions"]

        return []

    @staticmethod
    def _normalize_header(value: str) -> str:
        """Normalize header strings for comparison."""
        if not isinstance(value, str):
            return ""
        return re.sub(r"[^a-z0-9]", "", value.lower())

    def get_field_label(self, name: str) -> str:
        """Return a human-friendly label for a field name."""
        return self.field_labels.get(name, name)

    def get_mappable_fields(self) -> list[dict[str, Any]]:
        """Return schema fields that should appear in the mapping UI."""
        return [
            field
            for field in self.field_definitions
            if isinstance(field, dict) and field.get("show_in_mapping", True)
        ]

    def detect_encoding(self, file_path: Path) -> str:
        """
        Detect the encoding of a CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            Detected encoding string
        """
        try:
            with open(file_path, "rb") as f:
                # Read first 10000 bytes for detection
                raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                encoding = result["encoding"]
                confidence = result["confidence"]

                logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")

                # Fall back to utf-8 if confidence is low
                if confidence < 0.7:
                    logger.warning("Low confidence in encoding detection, using UTF-8")
                    return "utf-8"

                return encoding
        except Exception as e:
            logger.error(f"Error detecting encoding: {e}")
            return "utf-8"

    def read_csv(self, file_path: Path) -> pd.DataFrame:
        """
        Read a CSV file with automatic encoding detection.

        Args:
            file_path: Path to the CSV file

        Returns:
            DataFrame with the CSV data
        """
        # Detect encoding
        encoding = self.detect_encoding(file_path)

        try:
            # Try reading with detected encoding
            df = pd.read_csv(file_path, encoding=encoding)
            logger.info(f"Successfully read CSV with {len(df)} rows")
            return df
        except UnicodeDecodeError:
            # Fallback to different encodings
            encodings_to_try = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]

            for enc in encodings_to_try:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    logger.info(f"Successfully read CSV with {enc} encoding")
                    return df
                except UnicodeDecodeError:
                    continue

            # If all fail, read with error handling
            logger.warning("Reading CSV with error handling")
            return pd.read_csv(file_path, encoding="utf-8", errors="ignore")

    def validate_columns(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """
        Validate that required columns are present.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, missing_columns)
        """
        missing_columns = []

        for col in self.required_columns:
            if col not in df.columns and col not in self.field_mapping.values():
                missing_columns.append(col)

        is_valid = len(missing_columns) == 0

        if not is_valid:
            logger.warning(f"Missing required columns: {missing_columns}")
        else:
            logger.info("All required columns present")

        return is_valid, missing_columns

    def auto_map_fields(self, headers: list[str]) -> dict[str, str]:
        """
        Automatically map CSV headers to standard field names.

        Args:
            headers: List of CSV column headers

        Returns:
            Dictionary mapping standard fields to CSV columns
        """
        if not headers:
            return {}

        if self.field_definitions:
            mapping = self._auto_map_with_schema(headers)
        else:
            mapping = self._auto_map_with_defaults(headers)

        self.field_mapping = mapping
        logger.info(f"Auto-mapped {len(mapping)} fields")

        return mapping

    def _auto_map_with_schema(self, headers: list[str]) -> dict[str, str]:
        """Auto-map using configured field schema."""
        mapping: dict[str, str] = {}
        used_headers: set[str] = set()
        normalized_headers = {header: self._normalize_header(header) for header in headers}

        for field in self.field_definitions:
            if not isinstance(field, dict):
                continue

            name = field.get("name")
            if not name:
                continue

            matched_header = self._match_field_to_header(
                field, headers, normalized_headers, used_headers
            )

            if matched_header:
                mapping[name] = matched_header
                used_headers.add(matched_header)

        return mapping

    def _auto_map_with_defaults(self, headers: list[str]) -> dict[str, str]:
        """Fallback auto-mapping when no schema is provided."""
        mapping: dict[str, str] = {}
        patterns = {
            "call_id": r"(call[\s_-]?id|id|identifier|record[\s_-]?id)",
            "phone_number": r"(phone|telephone|number|contact|mobile|cell)",
            "timestamp": r"(time|date|timestamp|created|started|datetime)",
            "duration": r"(duration|length|call[\s_-]?time|call[\s_-]?length|seconds|minutes)",
            "agent_id": r"(agent|employee|staff|representative|rep[\s_-]?id)",
            "campaign": r"(campaign|project|program|initiative)",
            "outcome": r"(outcome|result|status|disposition)",
            "call_type": r"(type|category|classification|reason)",
            "notes": r"(notes|comments|remarks|description|transcript)",
            "revenue": r"(revenue|amount|value|price|cost|sale)",
        }

        for standard_field, pattern in patterns.items():
            for header in headers:
                if re.search(pattern, header.lower()):
                    mapping[standard_field] = header
                    break

        return mapping

    def _match_field_to_header(
        self,
        field: dict[str, Any],
        headers: list[str],
        normalized_headers: dict[str, str],
        used_headers: set[str],
    ) -> str | None:
        """Match a single field definition to the best header."""
        name = field.get("name")
        if not name:
            return None

        label = field.get("label")
        aliases = self.field_alias_map.get(name, [])
        candidates = {self._normalize_header(name)}

        if label:
            candidates.add(self._normalize_header(label))

        candidates.update(self._normalize_header(alias) for alias in aliases)

        for header in headers:
            if header in used_headers:
                continue
            if normalized_headers.get(header) in candidates:
                return header

        patterns = self.field_pattern_map.get(name, [])
        for pattern in patterns:
            try:
                compiled = re.compile(pattern)
            except re.error as exc:
                logger.warning(f"Invalid regex pattern '{pattern}' for field '{name}': {exc}")
                continue

            for header in headers:
                if header in used_headers:
                    continue
                normalized_value = normalized_headers.get(header, "")
                if compiled.search(header.lower()) or compiled.search(normalized_value):
                    return header

        return None

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names and add missing optional columns.

        Args:
            df: DataFrame to standardize

        Returns:
            DataFrame with standardized columns
        """
        # Apply field mapping if available
        if self.field_mapping:
            df = df.rename(columns={v: k for k, v in self.field_mapping.items()})

        # Normalize column names (lowercase, replace spaces with underscores)
        df.columns = [col.lower().replace(" ", "_").replace("-", "_") for col in df.columns]

        # Remove duplicate columns that can break downstream processing
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
            logger.info("Removed duplicate columns after standardization")

        if "timestamp" not in df.columns:
            for fallback in ["start_time", "created_at", "date"]:
                if fallback in df.columns:
                    df["timestamp"] = df[fallback]
                    logger.debug(f"Created timestamp column from {fallback}")
                    break

        # Add missing optional columns with default values
        for col in self.optional_columns:
            if col not in df.columns:
                df[col] = None

        logger.info(f"Standardized columns: {list(df.columns)}")
        return df

    def parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse date columns to datetime format.

        Args:
            df: DataFrame with date columns

        Returns:
            DataFrame with parsed dates
        """
        date_columns = ["timestamp", "start_time", "end_time", "created_at", "updated_at"]

        for col in date_columns:
            if col in df.columns:
                for date_format in self.date_formats:
                    try:
                        df[col] = pd.to_datetime(df[col], format=date_format)
                        logger.info(f"Parsed {col} with format {date_format}")
                        break
                    except (TypeError, ValueError):
                        continue

                # If no format worked, try pandas auto-detection
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    try:
                        df[col] = pd.to_datetime(df[col])
                        logger.info(f"Parsed {col} with auto-detection")
                    except (TypeError, ValueError):
                        logger.warning(f"Could not parse dates in column {col}")

        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the data.

        Args:
            df: DataFrame to clean

        Returns:
            Cleaned DataFrame
        """
        # Remove duplicate rows
        original_len = len(df)
        df = df.drop_duplicates()
        if len(df) < original_len:
            logger.info(f"Removed {original_len - len(df)} duplicate rows")

        # Clean phone numbers (remove non-numeric characters)
        if "phone_number" in df.columns:
            df["phone_number"] = df["phone_number"].astype(str).str.replace(r"\D", "", regex=True)

        # Convert duration to seconds if in different format
        if "duration" in df.columns and df["duration"].dtype == "object":
            # If duration is a string like "5:30", convert to seconds
            def parse_duration(val):
                if pd.isna(val):
                    return 0
                if ":" in str(val):
                    parts = str(val).split(":")
                    if len(parts) == 2:
                        return int(parts[0]) * 60 + int(parts[1])
                    if len(parts) == 3:
                        hours, minutes, seconds = (int(part) for part in parts)
                        return hours * 3600 + minutes * 60 + seconds
                return float(val)

            df["duration"] = df["duration"].apply(parse_duration)

        # Trim whitespace from string columns
        string_columns = df.select_dtypes(include=["object"]).columns
        for col in string_columns:
            df[col] = df[col].str.strip() if df[col].dtype == "object" else df[col]

        # Handle missing values
        df["notes"] = df.get("notes", "").fillna("")
        df["outcome"] = df.get("outcome", "unknown").fillna("unknown")

        logger.info("Data cleaning completed")
        return df

    def validate_data_quality(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Validate data quality and generate metrics.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with quality metrics
        """
        metrics = {
            "total_rows": len(df),
            "duplicate_rows": len(df[df.duplicated()]),
            "missing_values": {},
            "quality_score": 100.0,
        }

        # Check for missing values in required columns
        for col in self.required_columns:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    metrics["missing_values"][col] = missing_count
                    metrics["quality_score"] -= (missing_count / len(df)) * 20

        # Additional quality checks
        if "phone_number" in df.columns:
            invalid_phones = df["phone_number"].str.len() < 10
            metrics["invalid_phone_numbers"] = invalid_phones.sum()
            if metrics["invalid_phone_numbers"] > 0:
                metrics["quality_score"] -= 5

        if "duration" in df.columns:
            negative_durations = df["duration"] < 0
            metrics["negative_durations"] = negative_durations.sum()
            if metrics["negative_durations"] > 0:
                metrics["quality_score"] -= 10

        metrics["quality_score"] = max(0, metrics["quality_score"])

        logger.info(f"Data quality score: {metrics['quality_score']:.1f}%")
        return metrics

    def get_csv_preview(self, file_path: Path, num_rows: int = 5) -> pd.DataFrame:
        """
        Get a preview of CSV file contents.

        Args:
            file_path: Path to CSV file
            num_rows: Number of rows to preview

        Returns:
            DataFrame with preview data
        """
        try:
            encoding = self.detect_encoding(file_path)
            df = pd.read_csv(file_path, encoding=encoding, nrows=num_rows)
            return df
        except Exception as e:
            logger.error(f"Error getting CSV preview: {e}")
            return pd.DataFrame()

    def process_csv_batch(
        self, file_path: Path, batch_size: int = 1000, batch_callback: Callable | None = None
    ) -> tuple[int, int]:
        """
        Process CSV file in batches for large files.

        Args:
            file_path: Path to CSV file
            batch_size: Number of rows per batch
            batch_callback: Callback function for each batch

        Returns:
            Tuple of (total_processed, total_errors)
        """
        total_processed = 0
        total_errors = 0
        self.errors_log = []

        try:
            encoding = self.detect_encoding(file_path)

            # Process in chunks
            for chunk in pd.read_csv(file_path, encoding=encoding, chunksize=batch_size):
                try:
                    # Process chunk
                    chunk = self.standardize_columns(chunk)
                    chunk = self.parse_dates(chunk)
                    chunk = self.clean_data(chunk)

                    # Validate chunk
                    is_valid, missing = self.validate_columns(chunk)
                    if not is_valid:
                        logger.warning(f"Batch has missing columns: {missing}")

                    # Call callback if provided
                    if batch_callback:
                        batch_callback(chunk)

                    total_processed += len(chunk)

                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    total_errors += len(chunk)
                    self.errors_log.append({"batch_start": total_processed, "error": str(e)})

        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            raise

        return total_processed, total_errors

    def export_errors_report(self, output_path: Path):
        """
        Export error report to CSV file.

        Args:
            output_path: Path for error report file
        """
        if self.errors_log:
            pd.DataFrame(self.errors_log).to_csv(output_path, index=False)
            logger.info(f"Error report exported to {output_path}")


class CSVExporter:
    """
    Exports data to various CSV formats with customization options.
    """

    def __init__(self):
        """Initialize CSV exporter"""
        self.export_configs = {
            "standard": {
                "columns": [
                    "call_id",
                    "timestamp",
                    "phone_number",
                    "agent_id",
                    "campaign",
                    "outcome",
                    "duration",
                    "revenue",
                    "notes",
                ],
                "date_format": "%Y-%m-%d %H:%M:%S",
            },
            "detailed": {
                "columns": None,  # Include all columns
                "date_format": "%Y-%m-%d %H:%M:%S",
            },
            "summary": {
                "columns": ["call_id", "timestamp", "outcome", "duration", "revenue"],
                "date_format": "%Y-%m-%d",
            },
        }
        logger.info("CSVExporter initialized")

    def export_to_csv(
        self,
        df: pd.DataFrame,
        output_path: Path,
        format_type: str = "standard",
        encoding: str = "utf-8",
    ) -> Path:
        """
        Export DataFrame to CSV file.

        Args:
            df: DataFrame to export
            output_path: Path for output file
            format_type: Export format type ('standard', 'detailed', 'summary')
            encoding: File encoding

        Returns:
            Path to exported file
        """
        try:
            config = self.export_configs.get(format_type, self.export_configs["standard"])

            # Select columns
            if config["columns"]:
                # Only include columns that exist in the DataFrame
                columns = [col for col in config["columns"] if col in df.columns]
                export_df = df[columns].copy()
            else:
                export_df = df.copy()

            # Format dates
            date_columns = ["timestamp", "created_at", "updated_at", "date"]
            for col in date_columns:
                if col in export_df.columns and pd.api.types.is_datetime64_any_dtype(
                    export_df[col]
                ):
                    export_df[col] = export_df[col].dt.strftime(config["date_format"])

            # Export to CSV
            export_df.to_csv(output_path, index=False, encoding=encoding)

            logger.info(f"Exported {len(export_df)} rows to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            raise

    def export_to_excel(
        self,
        df: pd.DataFrame,
        output_path: Path,
        sheet_name: str = "Call Data",
        include_summary: bool = True,
    ) -> Path:
        """
        Export DataFrame to Excel file with optional summary sheet.

        Args:
            df: DataFrame to export
            output_path: Path for output file
            sheet_name: Name for the data sheet
            include_summary: Whether to include a summary sheet

        Returns:
            Path to exported file
        """
        try:
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                # Write main data
                df.to_excel(writer, sheet_name=sheet_name, index=False)

                # Add summary sheet if requested
                if include_summary:
                    summary = self._generate_summary(df)
                    summary.to_excel(writer, sheet_name="Summary", index=False)

                # Auto-adjust column widths
                for sheet in writer.sheets.values():
                    for column in sheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            value = cell.value
                            if value is None:
                                continue
                            cell_length = len(str(value))
                            if cell_length > max_length:
                                max_length = cell_length
                        adjusted_width = min(max_length + 2, 50)
                        sheet.column_dimensions[column_letter].width = adjusted_width

            logger.info(f"Exported {len(df)} rows to Excel: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            raise

    def export_to_json(
        self, df: pd.DataFrame, output_path: Path, orient: str = "records", indent: int = 2
    ) -> Path:
        """
        Export DataFrame to JSON file.

        Args:
            df: DataFrame to export
            output_path: Path for output file
            orient: JSON orientation ('records', 'index', 'columns', 'values')
            indent: Indentation level for pretty printing

        Returns:
            Path to exported file
        """
        try:
            # Convert datetime columns to string
            export_df = df.copy()
            for col in export_df.columns:
                if pd.api.types.is_datetime64_any_dtype(export_df[col]):
                    export_df[col] = export_df[col].dt.strftime("%Y-%m-%d %H:%M:%S")

            # Export to JSON
            export_df.to_json(output_path, orient=orient, indent=indent)

            logger.info(f"Exported {len(df)} rows to JSON: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            raise

    def export_for_analytics(
        self, df: pd.DataFrame, output_dir: Path, split_by: str | None = None
    ) -> list[Path]:
        """
        Export data optimized for analytics tools.

        Args:
            df: DataFrame to export
            output_dir: Directory for output files
            split_by: Optional column to split data by

        Returns:
            List of exported file paths
        """
        exported_files = []
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            if split_by and split_by in df.columns:
                # Split data by specified column
                for value in df[split_by].unique():
                    subset = df[df[split_by] == value]
                    filename = f"call_data_{split_by}_{value}.csv"
                    output_path = output_dir / filename
                    subset.to_csv(output_path, index=False)
                    exported_files.append(output_path)
                    logger.info(f"Exported {len(subset)} rows for {split_by}={value}")
            else:
                # Export as single file
                output_path = output_dir / "call_data_export.csv"
                df.to_csv(output_path, index=False)
                exported_files.append(output_path)
                logger.info(f"Exported {len(df)} rows to {output_path}")

            # Also create a metadata file
            has_timestamp = "timestamp" in df.columns
            metadata = {
                "export_date": datetime.now().isoformat(),
                "total_records": len(df),
                "columns": list(df.columns),
                "date_range": {
                    "start": df["timestamp"].min().isoformat() if has_timestamp else None,
                    "end": df["timestamp"].max().isoformat() if has_timestamp else None,
                },
                "files_created": [str(f) for f in exported_files],
            }

            metadata_path = output_dir / "export_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            exported_files.append(metadata_path)

        except Exception as e:
            logger.error(f"Error exporting for analytics: {e}")
            raise

        return exported_files

    def _generate_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for the data.

        Args:
            df: DataFrame to summarize

        Returns:
            DataFrame with summary statistics
        """
        summary_data = []

        # Basic counts
        summary_data.append({"Metric": "Total Calls", "Value": len(df)})

        # Date range
        if "timestamp" in df.columns:
            summary_data.append(
                {
                    "Metric": "Date Range",
                    "Value": f"{df['timestamp'].min()} to {df['timestamp'].max()}",
                }
            )

        # Unique values counts
        for col in ["agent_id", "campaign", "outcome", "call_type"]:
            if col in df.columns:
                summary_data.append(
                    {
                        "Metric": f'Unique {col.replace("_", " ").title()}s',
                        "Value": df[col].nunique(),
                    }
                )

        # Numeric statistics
        if "duration" in df.columns:
            summary_data.extend(
                [
                    {"Metric": "Average Duration (seconds)", "Value": df["duration"].mean()},
                    {"Metric": "Total Duration (hours)", "Value": df["duration"].sum() / 3600},
                ]
            )

        if "revenue" in df.columns:
            summary_data.extend(
                [
                    {"Metric": "Total Revenue", "Value": df["revenue"].sum()},
                    {"Metric": "Average Revenue", "Value": df["revenue"].mean()},
                ]
            )

        # Outcome distribution
        if "outcome" in df.columns:
            for outcome, count in df["outcome"].value_counts().items():
                summary_data.append({"Metric": f"Outcome: {outcome}", "Value": count})

        return pd.DataFrame(summary_data)
