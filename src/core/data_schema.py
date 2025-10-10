"""
Data Schema Module

This module defines the core data structures and validation logic
for the call analytics system. It ensures data consistency across
all processing stages.
"""

from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, validator


class CallConnectionStatus(str, Enum):
    """Enumeration for call connection status"""

    CONNECTED = "Connected"
    DISCONNECTED = "Disconnected"
    UNKNOWN = "Unknown"


class CallType(str, Enum):
    """Enumeration for call types"""

    INQUIRY = "Inquiry"
    BILLING_SALES = "Billing/Sales"
    SUPPORT = "Support"
    COMPLAINT = "Complaint"
    UNKNOWN = "Unknown"


class CallOutcome(str, Enum):
    """Enumeration for call outcomes"""

    RESOLVED = "Resolved"
    CALLBACK = "Callback"
    REFUND = "Refund"
    SALE_CLOSE = "Sale-close"
    UNKNOWN = "Unknown"


class CallRecord(BaseModel):
    """
    Data model for a single call record.
    This ensures all call data follows a consistent structure.
    """

    call_id: str = Field(..., description="Unique identifier for the call")
    start_time: datetime = Field(..., description="Call start timestamp")
    duration_seconds: float = Field(..., description="Call duration in seconds")
    transcript: str = Field(..., description="Call transcript text")

    # Optional fields that may come from CSV or be added during processing
    agent_id: str | None = Field(None, description="Agent identifier")
    campaign: str | None = Field(None, description="Campaign name")
    customer_name: str | None = Field(None, description="Customer name")
    product_name: str | None = Field(None, description="Product discussed")
    quantity: int | None = Field(None, description="Quantity if applicable")
    amount: float | None = Field(None, description="Transaction amount")
    order_id: str | None = Field(None, description="Order identifier")

    # Derived labels from analysis
    connection_status: CallConnectionStatus | None = None
    call_type: CallType | None = None
    outcome: CallOutcome | None = None

    @validator("duration_seconds")
    def validate_duration(cls, v):  # noqa: N805
        """Ensure duration is non-negative"""
        if v < 0:
            raise ValueError("Duration cannot be negative")
        return v

    @validator("transcript")
    def validate_transcript(cls, v):  # noqa: N805
        """Ensure transcript is not empty"""
        if not v or not v.strip():
            raise ValueError("Transcript cannot be empty")
        return v.strip()

    class Config:
        """Pydantic configuration"""

        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class CallDataFrame:
    """
    Wrapper class for managing call data in DataFrame format.
    Provides validation and transformation utilities.
    """

    # Define expected column names and types
    REQUIRED_COLUMNS = {
        "call_id": "string",
        "start_time": "datetime64[ns]",
        "duration_seconds": "float64",
        "transcript": "string",
    }

    OPTIONAL_COLUMNS = {
        "agent_id": "string",
        "campaign": "string",
        "customer_name": "string",
        "product_name": "string",
        "quantity": "Int64",  # Nullable integer
        "amount": "Float64",  # Nullable float
        "order_id": "string",
        "connection_status": "string",
        "call_type": "string",
        "outcome": "string",
    }

    def __init__(self, df: pd.DataFrame):
        """Initialize with a pandas DataFrame"""
        self.df = self._validate_and_normalize(df)

    def _validate_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate DataFrame structure and normalize data types.
        This ensures consistency across all operations.
        """
        df = df.copy()

        # Check for required columns
        missing_cols = set(self.REQUIRED_COLUMNS.keys()) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Add missing optional columns with appropriate NA values
        for col, _dtype in self.OPTIONAL_COLUMNS.items():
            if col not in df.columns:
                df[col] = pd.NA

        # Convert data types for all columns
        for col, dtype in {**self.REQUIRED_COLUMNS, **self.OPTIONAL_COLUMNS}.items():
            if col in df.columns:
                try:
                    if dtype == "datetime64[ns]":
                        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
                    elif dtype in ["Int64", "Float64"]:
                        df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
                    else:
                        df[col] = df[col].astype(dtype)
                except Exception as e:
                    print(f"Warning: Could not convert column {col} to {dtype}: {e}")

        # Sort columns in a consistent order
        column_order = list(self.REQUIRED_COLUMNS) + [
            col for col in self.OPTIONAL_COLUMNS if col in df.columns
        ]
        df = df[column_order]

        return df

    def add_labels(self, labels: dict[str, Any]) -> None:
        """
        Add derived labels to the DataFrame.

        Args:
            labels: Dictionary mapping call_id to label dictionaries
        """
        for call_id, label_dict in labels.items():
            mask = self.df["call_id"] == call_id
            for label_key, label_value in label_dict.items():
                if label_key in ["connection_status", "call_type", "outcome"]:
                    self.df.loc[mask, label_key] = label_value

    def filter_by_date_range(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Filter calls by date range.

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)

        Returns:
            Filtered DataFrame
        """
        mask = (self.df["start_time"] >= start_date) & (self.df["start_time"] <= end_date)
        return self.df[mask].copy()

    def get_statistics(self) -> dict[str, Any]:
        """
        Calculate basic statistics for the dataset.

        Returns:
            Dictionary containing various statistics
        """
        total_calls = len(self.df)

        # Connection statistics
        connection_counts = self.df["connection_status"].value_counts().to_dict()
        connected_pct = (
            connection_counts.get("Connected", 0) / total_calls * 100 if total_calls > 0 else 0
        )

        # Type distribution
        type_distribution = self.df["call_type"].value_counts().to_dict()

        # Outcome distribution
        outcome_distribution = self.df["outcome"].value_counts().to_dict()

        # Duration statistics
        duration_stats = {
            "mean": self.df["duration_seconds"].mean(),
            "median": self.df["duration_seconds"].median(),
            "min": self.df["duration_seconds"].min(),
            "max": self.df["duration_seconds"].max(),
        }

        return {
            "total_calls": total_calls,
            "connected_percentage": connected_pct,
            "connection_breakdown": connection_counts,
            "type_distribution": type_distribution,
            "outcome_distribution": outcome_distribution,
            "duration_statistics": duration_stats,
            "unique_agents": self.df["agent_id"].nunique(),
            "unique_campaigns": self.df["campaign"].nunique(),
        }

    def to_records(self) -> list[CallRecord]:
        """
        Convert DataFrame to list of CallRecord objects.

        Returns:
            List of validated CallRecord instances
        """
        records = []
        for _, row in self.df.iterrows():
            try:
                record = CallRecord(**row.to_dict())
                records.append(record)
            except Exception as e:
                print(f"Warning: Could not convert row to CallRecord: {e}")
                continue
        return records

    def export_to_csv(self, filepath: str) -> None:
        """
        Export the DataFrame to CSV file.

        Args:
            filepath: Path where CSV should be saved
        """
        self.df.to_csv(filepath, index=False)
        print(f"Data exported to {filepath}")

    def export_to_parquet(self, filepath: str) -> None:
        """
        Export the DataFrame to Parquet file for efficient storage.

        Args:
            filepath: Path where Parquet file should be saved
        """
        self.df.to_parquet(filepath, index=False)
        print(f"Data exported to {filepath}")
