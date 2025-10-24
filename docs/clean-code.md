# Clean Code

## Overview

These clean code guidelines help keep my project readable, maintainable, and scalable. They're based on industry standards like Robert C. Martin's "Clean Code" and Python best practices, with real examples from codebase.

## Table of Contents

- [Overview](#overview)
- [Build Management](#build-management)
  - [Ant workflow (`build.xml`)](#ant-workflow-buildxml)
  - [Maven workflow (`pom.xml`)](#maven-workflow-pomxml)
- [Naming Conventions](#naming-conventions)
  - [Use Descriptive Names](#use-descriptive-names)
  - [Follow PEP 8 Conventions](#follow-pep-8-conventions)
  - [Avoid Ambiguous Abbreviations](#avoid-ambiguous-abbreviations)
- [Function Design](#function-design)
  - [Single Responsibility Principle](#single-responsibility-principle)
  - [Keep Functions Short](#keep-functions-short)
  - [Use Type Hints](#use-type-hints)
  - [Minimize Function Parameters](#minimize-function-parameters)
- [Class Structure](#class-structure)
  - [Clear Responsibilities](#clear-responsibilities)
  - [Composition Over Inheritance](#composition-over-inheritance)
  - [Method Chaining Pattern](#method-chaining-pattern)
- [Error Handling](#error-handling)
  - [Use Specific Exceptions](#use-specific-exceptions)
  - [Fail Fast](#fail-fast)
  - [Log Errors Meaningfully](#log-errors-meaningfully)
- [Code Organization](#code-organization)
  - [Separate Concerns into Modules](#separate-concerns-into-modules)
  - [Use Helper Functions](#use-helper-functions)
  - [Avoid Deep Nesting](#avoid-deep-nesting)
- [Documentation](#documentation)
  - [Docstrings for Public APIs](#docstrings-for-public-apis)
  - [Use Type Annotations](#use-type-annotations)
  - [Comments for Complex Logic](#comments-for-complex-logic)
- [Testing](#testing)
  - [Write Testable Code](#write-testable-code)
  - [Use Dependency Injection](#use-dependency-injection)
- [Refactoring Examples](#refactoring-examples)
  - [Before and After: Extracted Functions](#before-and-after-extracted-functions)
  - [Before and After: Reduced Nesting](#before-and-after-reduced-nesting)
- [Best Practices Summary](#best-practices-summary)
  - [Code Review Checklist](#code-review-checklist)
  - [Refactoring Guidelines](#refactoring-guidelines)
- [Code Examples Index](#code-examples-index)
  - [By Topic](#by-topic)
- [Personal CCD Cheat Sheet](#personal-ccd-cheat-sheet)

### Build Management

I run the project with two build tools because each one solves a different problem for me. Ant keeps all the Python chores in one place, and Maven triggers Ant when I want a full build plus a release zip.

Before I call either tool I make sure Python 3.11, Ant, Maven, and a JDK are installed. The Ant `setup` target creates the virtual environment and installs the dev extras, so I do not have to manage that by hand.

#### Ant workflow (`build.xml`)

- `ant setup` creates `.venv` and installs dependencies.
- `ant lint` runs `ruff`, `black --check`, and `mypy` over `src/` and `scripts/`.
- `ant test` executes `pytest` and saves the JUnit report in `test-reports/junit.xml`.
- `ant docs` builds the Sphinx HTML docs into `docs/_build/html`.
- `ant wheel` produces a Python wheel using `python -m build`.
- `ant ci` chains `clean`, `setup`, `lint`, `test`, `docs`, and `wheel`, which covers the “run tests, generate docs, and build artifacts” requirement.

#### Maven workflow (`pom.xml`)

Maven wraps the Ant build so I can use standard Maven phases when I deploy.

- `mvn -B verify` triggers the Ant `ci` target through the `maven-antrun-plugin`.
- `mvn -B package` zips up `src`, `config`, and the top-level metadata into `artifacts/call-analytics-1.0.0.zip` via the `exec-maven-plugin`.

That way I can stay in Maven if I need to integrate with other JVM tooling, while still reusing the Python-focused Ant targets.

## Naming Conventions

### Use Descriptive Names

Names should clearly express intent without requiring comments.

**✅ Good Examples from Repository**:

*From: `src/utils/validators.py`, lines 27-36*

```python
def validate_email(email: str) -> bool:
    """Validate email address format."""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, str(email)))


def validate_phone(phone: str, region: str = "US") -> bool:
    """Validate phone number format."""
    # Remove common separators
    phone = re.sub(r"[\s\-\.\(\)]", "", str(phone))
```

Functions clearly state what they validate. No confusion about what `validate_email` does.

*From: `scripts/rebuild_index.py`, lines 246-305*

```python
def handle_stats_only(rebuilder: IndexRebuilder, logger: logging.Logger) -> None:
    """Handle the stats-only command."""
    stats = rebuilder.generate_index_stats()
    logger.info("Current index statistics:")
    logger.info(json.dumps(stats, indent=2))


def perform_backup_if_needed(
    rebuilder: IndexRebuilder,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> bool:
    """Backup the index if backup isn't disabled."""
    if args.no_backup:
        return True

    if rebuilder.backup_existing_index(args.backup_dir):
        return True

    logger.error("Backup failed, aborting rebuild")
    return False


def load_records_or_exit(rebuilder: IndexRebuilder, logger: logging.Logger) -> pd.DataFrame:
    """Load records from storage, exit if none found."""
    records = rebuilder.load_records()
    if records is None or records.empty:
        logger.warning("No records to index")
        sys.exit(0)
    return records
```

Each function name tells you exactly what it does. `load_records_or_exit` makes it clear the function might exit the program.

### Follow PEP 8 Conventions

*Functions and local variables | from `src/analysis/aggregations.py`, lines 39-75*:

```python
def calculate_basic_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
    metrics = {}
    metrics["total_calls"] = len(df)

    if "outcome" in df.columns:
        connected = df["outcome"].str.lower() == "connected"
        metrics["connection_rate"] = (connected.sum() / len(df) * 100) if len(df) > 0 else 0
        metrics["connected_calls"] = connected.sum()

    if "duration" in df.columns:
        metrics["avg_duration"] = df["duration"].mean()
        metrics["total_duration"] = df["duration"].sum()
```

*Classes | from `src/analysis/aggregations.py`, lines 28-37*:

```python
class MetricsCalculator:
    """Calculate various metrics and KPIs from call data."""

    def __init__(self):
        self.metrics_cache = {}
        logger.info("MetricsCalculator initialized")
```

*Constants | from `src/utils/file_handlers.py`, lines 151-160*:

```python
SECURE_HASHERS: dict[str, HashFactory] = {
    "sha256": hashlib.sha256,
    "sha384": hashlib.sha384,
    "sha512": hashlib.sha512,
    "blake2b": hashlib.blake2b,
    "blake2s": hashlib.blake2s,
}
```

### Avoid Ambiguous Abbreviations

**❌ Bad**:
```python
def proc_data(df):  # What does proc mean?
    temp = df  # Temp what?
    return temp
```

**✅ Good** (*from `src/analysis/filters.py`, lines 48-78*):
```python
def apply_date_range(
    self,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> "DataFilter":
    """Apply date range filter to the data."""
```

## Function Design

### Single Responsibility Principle

Each function should do one thing well.

**✅ Good Example** (*from `scripts/rebuild_index.py`, lines 246-305*):

```python
# Each function has ONE clear responsibility

def handle_stats_only(rebuilder: IndexRebuilder, logger: logging.Logger) -> None:
    """Only handles displaying stats."""
    stats = rebuilder.generate_index_stats()
    logger.info("Current index statistics:")
    logger.info(json.dumps(stats, indent=2))


def perform_backup_if_needed(...) -> bool:
    """Only handles backup logic."""
    if args.no_backup:
        return True
    if rebuilder.backup_existing_index(args.backup_dir):
        return True
    logger.error("Backup failed, aborting rebuild")
    return False


def load_records_or_exit(...) -> pd.DataFrame:
    """Only handles loading records."""
    records = rebuilder.load_records()
    if records is None or records.empty:
        logger.warning("No records to index")
        sys.exit(0)
    return records


def run_rebuild(...) -> None:
    """Coordinates the rebuild process."""
    records = load_records_or_exit(rebuilder, logger)
    indexed_count = rebuilder.rebuild_index(...)
    if args.verify:
        if rebuilder.verify_index():
            logger.info("✓ Index verification passed")
    # ... more coordination
```

Instead of one giant function, I have small focused functions that do one thing.

### Keep Functions Short

Aim for functions that fit on one screen (15-30 lines).

**✅ Good Example** (*from `src/utils/validators.py`, lines 65-98*):

```python
def validate_date(
    date_value: Any,
    format: str = "%Y-%m-%d",
    min_date: date | None = None,
    max_date: date | None = None,
) -> bool:
    """Validate date value and format."""
    try:
        if isinstance(date_value, str):
            parsed_date = datetime.strptime(date_value, format).date()
        elif isinstance(date_value, datetime):
            parsed_date = date_value.date()
        elif isinstance(date_value, date):
            parsed_date = date_value
        else:
            return False

        if min_date is not None and parsed_date < min_date:
            return False

        return not (max_date is not None and parsed_date > max_date)

    except (ValueError, TypeError):
        return False
```

Short, focused, easy to understand and test.

### Use Type Hints

**✅ Good Example** (*from `src/core/labeling_engine.py`, lines 17-24*):

```python
@dataclass
class LabelingResult:
    """Container for labeling results"""
    connection_status: str
    call_type: str
    outcome: str
    confidence_scores: dict[str, float]
    matched_keywords: list[str]
```

Type hints make code self-documenting and catch errors early.

### Minimize Function Parameters

**✅ Good Example** (*from `src/vectordb/indexer.py`, lines 63-71*):

```python
def index_dataframe(
    self, 
    df: pd.DataFrame, 
    batch_size: int = 100, 
    update_existing: bool = True
) -> int:
    """Index a DataFrame of calls into the vector database."""
```

Uses default values for optional parameters. Clear and manageable.

## Class Structure

### Clear Responsibilities

**✅ Good Example** (*from `scripts/rebuild_index.py`, lines 27-43*):

```python
class IndexRebuilder:
    """High-level helper for rebuilding the semantic search index."""

    def __init__(
        self,
        storage_manager: StorageManager,
        vector_client: ChromaClient,
        indexer: DocumentIndexer,
        logger: logging.Logger,
    ) -> None:
        self.storage_manager = storage_manager
        self.vector_client = vector_client
        self.indexer = indexer
        self.logger = logger
```

Class has a clear purpose: rebuild the index. Dependencies are injected, making it testable.

### Composition Over Inheritance

*From `src/analysis/filters.py`, lines 18-40*:

```python
class DataFilter:
    """Core filtering class for call data with chainable filter methods."""

    def __init__(self, df: pd.DataFrame):
        """Initialize with a DataFrame."""
        self.original_df = df.copy()
        self.filtered_df = df.copy()
        self.active_filters = {}
        logger.info(f"DataFilter initialized with {len(df)} records")

    def reset_filters(self) -> "DataFilter":
        """Reset all filters and restore original data."""
        self.filtered_df = self.original_df.copy()
        self.active_filters = {}
        logger.info("Filters reset")
        return self
```

Uses composition (has a DataFrame) and method chaining for clean API.

### Method Chaining Pattern

*From `src/analysis/filters.py`, lines 42-71*:

```python
def apply_date_range(
    self, 
    start_date: datetime | None = None, 
    end_date: datetime | None = None
) -> "DataFilter":
    """Apply date range filter to the data."""
    if "timestamp" not in self.filtered_df.columns:
        logger.warning("No timestamp column found")
        return self

    # Filter logic...
    
    logger.info(f"Date filter applied: {start_date} to {end_date}")
    return self  # Return self for chaining


def apply_call_type_filter(self, call_types: list[str]) -> "DataFilter":
    """Filter by call types."""
    if not call_types:
        return self
    
    # Filter logic...
    
    logger.info(f"Call type filter applied: {call_types}")
    return self  # Return self for chaining
```

**Usage**:
```python
# Can chain multiple filters
filtered_data = (
    DataFilter(df)
    .apply_date_range(start_date, end_date)
    .apply_call_type_filter(["inbound", "outbound"])
    .apply_outcome_filter(["connected"])
    .get_result()
)
```

Clean, readable, fluent interface.

## Error Handling

### Use Specific Exceptions

*From `src/vectordb/indexer.py`, lines 85-103*:

```python
def index_dataframe(...) -> int:
    """Index a DataFrame of calls into the vector database."""
    if df.empty:
        logger.warning("Empty DataFrame provided for indexing")
        return 0

    # Prepare documents
    documents, ids, metadatas = self._prepare_documents(df)

    if not documents:
        logger.warning("No valid documents to index")
        return 0

    # Index in batches
    for i in range(0, len(documents), batch_size):
        try:
            # Index the batch
            count = self.vector_db.add_documents(...)
            total_indexed += count
        except Exception as e:
            logger.error(f"Error indexing batch {i//batch_size + 1}: {e}")
            continue  # Continue with other batches
```

Handles errors gracefully, logs them, and continues processing.

### Fail Fast

*From `scripts/rebuild_index.py`, lines 265-272*:

```python
def load_records_or_exit(rebuilder: IndexRebuilder, logger: logging.Logger) -> pd.DataFrame:
    """Load records from storage, exit if none found."""
    records = rebuilder.load_records()
    if records is None or records.empty:
        logger.warning("No records to index")
        sys.exit(0)  # Exit immediately if no work to do
    return records
```

Exits early if there's nothing to process. No wasted computation.

### Log Errors Meaningfully

*From `scripts/rebuild_index.py`, lines 51-75*:

```python
def backup_existing_index(self, backup_dir: Path) -> bool:
    """Backup the current vector store files if they exist."""
    try:
        persist_dir: Path = self.vector_client.persist_dir
        if not persist_dir.exists() or not any(persist_dir.iterdir()):
            self.logger.info("No existing index to backup")
            return True

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"index_backup_{timestamp}"
        
        # Backup logic...
        
        self.logger.info("Backup created at %s", backup_path)
        return True

    except Exception as exc:
        self.logger.error("Failed to backup index: %s", exc)
        return False
```

Logs success and failure with context. Returns boolean for flow control.

## Code Organization

### Separate Concerns into Modules

Project structure:
```
src/
├── analysis/        # Data analysis and aggregation
├── core/           # Core business logic
├── ml/             # Machine learning components
├── ui/             # User interface
├── utils/          # Utilities (validators, formatters, etc.)
└── vectordb/       # Vector database operations
```

Each module has a clear purpose.

### Use Helper Functions

*From `src/core/labeling_engine.py`, lines 172-182*:

```python
def _score_entities(
    self,
    transcript: str,
    rules: dict[str, list[str]],
    scorer: Callable[[str], float],
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Return score and keyword matches for each rule group."""
    scores: dict[str, float] = {}
    keyword_matches: dict[str, list[str]] = {}

    for label, keywords in rules.items():
        matches = [kw for kw in keywords if self._keyword_match(transcript, kw)]
        keyword_matches[label] = matches
        scores[label] = float(sum(scorer(keyword) for keyword in matches))

    return scores, keyword_matches
```

Private helper methods (prefixed with `_`) keep public API clean.

### Avoid Deep Nesting

**❌ Bad**:
```python
def process():
    if condition1:
        if condition2:
            if condition3:
                if condition4:
                    # Do something
```

**✅ Good** (*from `src/core/labeling_engine.py`*):
```python
def _determine_connection_status(self, transcript: str, duration: float):
    """Determine if the call was connected."""
    confidence = 0.0
    matched_keywords = []

    # Early exit for disconnection
    disconnection_keywords = self.connection_rules.get("disconnection_keywords", [])
    for keyword in disconnection_keywords:
        if self._keyword_match(transcript, keyword):
            matched_keywords.append(keyword)
            return "Disconnected", 0.9, matched_keywords

    # Early exit for short calls
    if duration < min_duration or word_count < min_words:
        return "Disconnected", 0.8, []

    # Default case
    return "Connected", 0.7, []
```

Uses early returns to avoid nesting.

## Documentation

### Docstrings for Public APIs

*From `src/analysis/filters.py`, lines 42-58*:

```python
def apply_date_range(
    self, start_date: datetime | None = None, end_date: datetime | None = None
) -> "DataFilter":
    """
    Apply date range filter to the data.

    Args:
        start_date: Start date for filtering
        end_date: End date for filtering

    Returns:
        Self for method chaining
    """
    if "timestamp" not in self.filtered_df.columns:
        logger.warning("No timestamp column found")
        return self
    # ...
```

Clear documentation of parameters and return values.

### Use Type Annotations

*From `src/vectordb/indexer.py`, lines 119-128*:

```python
def _prepare_documents(
    self, df: pd.DataFrame
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """
    Prepare documents from DataFrame for indexing.

    Args:
        df: DataFrame with call data

    Returns:
        Tuple of (documents, ids, metadatas)
    """
```

Type annotations show exactly what the function returns.

### Comments for Complex Logic

*From `src/core/labeling_engine.py`, lines 145-156*:

```python
def _determine_connection_status(self, transcript: str, duration: float):
    # Check duration threshold
    min_duration = self.connection_rules.get("min_duration_seconds", 30)
    min_words = self.connection_rules.get("min_transcript_words", 40)

    # Check for disconnection keywords
    disconnection_keywords = self.connection_rules.get("disconnection_keywords", [])
    for keyword in disconnection_keywords:
        if self._keyword_match(transcript, keyword):
            matched_keywords.append(keyword)
            confidence = 0.9
            return "Disconnected", confidence, matched_keywords
```

Habbits of commenting in any logics

## Testing

### Write Testable Code

*From repository test structure:*

```
tests/
├── test_aggregations.py
└── test_text_processing.py
```

Functions with clear inputs/outputs and no hidden dependencies are easy to test.

**Example of Testable Function** (*from `src/utils/validators.py`*):

```python
def validate_email(email: str) -> bool:
    """Validate email address format."""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, str(email)))


# Easy to test:
def test_validate_email():
    assert validate_email("user@example.com") == True
    assert validate_email("invalid-email") == False
    assert validate_email("") == False
```

No dependencies, pure function, predictable behavior.

### Use Dependency Injection

*From `scripts/rebuild_index.py`, lines 27-43*:

```python
class IndexRebuilder:
    """High-level helper for rebuilding the semantic search index."""

    def __init__(
        self,
        storage_manager: StorageManager,
        vector_client: ChromaClient,
        indexer: DocumentIndexer,
        logger: logging.Logger,
    ) -> None:
        self.storage_manager = storage_manager
        self.vector_client = vector_client
        self.indexer = indexer
        self.logger = logger
```

Dependencies injected through constructor. Easy to mock in tests.

## Refactoring Examples

### Before and After: Extracted Functions

**Before** (monolithic main):
```python
def main():
    # 100+ lines of mixed concerns
    args = parse_raw_args()
    setup_logging()
    logger = get_logger()
    
    # Stats handling
    if args.stats_only:
        stats = get_stats()
        print(json.dumps(stats))
        return
    
    # Backup logic
    if not args.no_backup:
        if not backup_index():
            logger.error("Backup failed")
            sys.exit(1)
    
    # Load and process
    records = load_records()
    if not records:
        logger.warning("No records")
        sys.exit(0)
    # ... more mixed logic
```

**After** (*from `scripts/rebuild_index.py`, lines 308-334*):
```python
def main() -> None:
    """Main function to run the index rebuild script."""
    args = parse_arguments()
    setup_logging(log_level="INFO", console_output=True)
    logger = get_logger(__name__)

    try:
        rebuilder = initialize_components(args, logger)
        
        if args.stats_only:
            handle_stats_only(rebuilder, logger)
            return

        if not perform_backup_if_needed(rebuilder, args, logger):
            sys.exit(1)

        run_rebuild(rebuilder, args, logger)

    except Exception as e:
        logger.error(f"Script failed: {e}", exc_info=True)
        sys.exit(1)
```

Much cleaner! Each concern has its own function.

### Before and After: Reduced Nesting

**Before**:
```python
def process_call(call):
    if call:
        if call.transcript:
            if len(call.transcript) > 10:
                if call.duration > 30:
                    return analyze(call)
    return None
```

**After** (pattern from repository):
```python
def process_call(call):
    """Process call with early exits."""
    if not call:
        return None
    if not call.transcript:
        return None
    if len(call.transcript) <= 10:
        return None
    if call.duration <= 30:
        return None
        
    return analyze(call)
```

## Best Practices Summary

### Code Review Checklist

Before submitting code:

- [ ] Functions have clear, descriptive names
- [ ] Each function does one thing
- [ ] Functions are short (< 30 lines ideally)
- [ ] Type hints used throughout
- [ ] Docstrings on public functions/classes
- [ ] No deep nesting (max 3 levels)
- [ ] Error handling in place
- [ ] Logging for important operations
- [ ] No hardcoded values
- [ ] Tests written for new code
- [ ] PEP 8 compliant (checked with ruff/black)

### Refactoring Guidelines

1. **Small steps**: Refactor incrementally
2. **Keep tests passing**: Run tests after each change
3. **One change at a time**: Don't mix refactoring with features
4. **Document why**: Update comments and docs
5. **Use version control**: Commit frequently

## Code Examples Index

### By Topic

**Build System**:
- Ant Workflow: `build.xml`
- Maven Workflow: `pom.xml`

**Naming**:
- Email/Phone validation: `src/utils/validators.py:27-36`
- Helper functions: `scripts/rebuild_index.py:246-305`

**Function Design**:
- Single responsibility: `scripts/rebuild_index.py:246-305`
- Short functions: `src/utils/validators.py:65-98`
- Type hints: `src/core/labeling_engine.py:17-24`

**Class Structure**:
- Clear responsibilities: `scripts/rebuild_index.py:27-43`
- Method chaining: `src/analysis/filters.py:42-71`

**Error Handling**:
- Graceful errors: `src/vectordb/indexer.py:85-103`
- Fail fast: `scripts/rebuild_index.py:265-272`
- Logging: `scripts/rebuild_index.py:51-75`

**Refactoring**:
- Extracted functions: `scripts/rebuild_index.py:308-334`
- Reduced nesting: `src/core/labeling_engine.py:145-156`

## Personal CCD Cheat Sheet

**Naming & Intent**
- I name for the reader, not the compiler: verbs for actions, nouns for things.
- I avoid abbreviations unless project-wide (e.g., `df` only inside data utilities).
- If a function name needs "and", I probably split it.

**Functions & Flow**
- One screen, one purpose: functions stay in the 15–43 line window.
- Guard clause first: handle empties or invalids up top, then let the path read straight.
- Parameters default; optional args get safe, explicit defaults.

**Structure & Dependencies**
- One responsibility per module (`core/`, `analysis/`, `ui/`, etc.).
- Inject dependencies (clients, DBs, LLMs) via constructors or parameters.
- Prefer composition ("have-a") over inheritance unless a clear hierarchy exists.
- Provide seams around AI calls (adapters) so I can stub them in tests.

**Data & State**
- Immutable by default: copy incoming DataFrames before mutating.
- No magic numbers: thresholds live in config (TOML/YAML), never inline.
- Favour pure functions—inputs in, outputs out, no hidden I/O.

**Types, Docs, Comments**
- Type-annotate public APIs and return types.
- Docstrings explain the why; comments document non-obvious trade-offs.
- Delete stale comments—the code is the source of truth.

**Errors & Logging**
- Fail fast when there is nothing useful to do.
- Log once, precisely (who/what/where) to avoid noisy loops.
- Catch specific exceptions `except Exception`.

**Testing**
- Parametrised edge-case tests for empty, mixed, and invalid inputs.
- Unit tests rely on DI/mocks—not live external services.
- Measure coverage and runtime deltas after refactors to prove gains.

**Tooling & Process**
- Autoclean via CI: `ruff`, `black --check`, `mypy`, `pytest -q` on every PR.
- Refactor in small steps, keep tests green, and explain intent in commit messages.
- Review like a user: run the flow in the UI or CLI before merging.
