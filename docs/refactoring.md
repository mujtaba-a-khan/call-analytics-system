# Refactoring

This document have the major refactoring efforts I made to enhance the code quality, maintainability of the project

## Table of contents

1. [Environment Setup Script](#1-environment-setup-script)
2. [Analysis Page Metric Constants](#2-analysis-page-metric-constants)
3. [Streamlit App Component Setup](#3-streamlit-app-component-setup)
4. [Embedding Manager Cache Flow](#4-embedding-manager-cache-flow)
5. [CSV Processor Field Matching](#5-csv-processor-field-matching)

---

## 1. Environment Setup Script

### Original Snippet

*From: `scripts/setup_environment.py` (commit `d378056`), lines 520-640*

```python
def main():
    parser = argparse.ArgumentParser(
        description='Setup environment for Call Analytics System'
    )
    parser.add_argument('--base-dir', type=Path, default=Path.cwd(), help='Base directory')
    parser.add_argument('--skip-packages', action='store_true', help='Skip package installation')
    parser.add_argument('--skip-sample-data', action='store_true', help='Skip creating sample data')
    parser.add_argument('--upgrade-packages', action='store_true', help='Upgrade existing packages')
    parser.add_argument('--verify-only', action='store_true', help='Only verify existing installation')
    args = parser.parse_args()

    logger = setup_logging_simple()
    setup = EnvironmentSetup(args.base_dir, logger)

    if args.verify_only:
        overall, status = setup.verify_installation()
        logger.info("\nInstallation status:")
        for component, installed in status.items():
            icon = "✓" if installed else "✗"
            logger.info(f"  {icon} {component}")
        if overall:
            logger.info("\n✓ Installation verified successfully!")
            sys.exit(0)
        logger.error("\n✗ Installation incomplete")
        sys.exit(1)

    logger.info("Starting Call Analytics System environment setup...")
    logger.info(f"Base directory: {args.base_dir}")

    if not setup.check_python_version():
        sys.exit(1)
    if not setup.create_directories():
        sys.exit(1)
    if not args.skip_packages:
        if not setup.install_packages(upgrade=args.upgrade_packages):
            logger.error("Package installation failed")
            sys.exit(1)
    if not setup.create_config_files():
        sys.exit(1)
    if not setup.setup_streamlit_config():
        sys.exit(1)
    if not args.skip_sample_data:
        setup.create_sample_data()

    overall, status = setup.verify_installation()
    logger.info("\nInstallation summary:")
    for component, installed in status.items():
        icon = "✓" if installed else "✗"
        logger.info(f"  {icon} {component}")
    if overall:
        setup.print_next_steps()
    else:
        logger.warning("\nSetup completed with some optional components missing.")
        logger.info("The core system should work, but some features may be limited.")
        setup.print_next_steps()
```

* What I struggled with: `main()` owned parsing, logging, setup, verification, and exit handling, so every change forced me to edit the same giant block.
* Why it mattered: I could not unit test individual branches or reuse the logic because nothing was isolated.
* How it behaved: any small tweak (for example adding a flag) risked breaking unrelated code since the function mixed concerns everywhere.

### Refactored version

*From: `scripts/setup_environment.py`, lines 1115-1241*

```python
def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Setup environment for Call Analytics System")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd(), help="Base directory for the project")
    parser.add_argument("--skip-packages", action="store_true", help="Skip Python package installation")
    parser.add_argument("--skip-system-deps", action="store_true", help="Skip system dependencies")
    parser.add_argument("--skip-sample-data", action="store_true", help="Skip creating sample data")
    parser.add_argument("--upgrade-packages", action="store_true", help="Upgrade existing packages")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing installation")
    return parser.parse_args(argv)


def log_component_status(header: str, status: dict[str, bool], logger: logging.Logger) -> None:
    logger.info(header)
    for component, installed in status.items():
        icon = "✓" if installed else "✗"
        logger.info(f"  {icon} {component}")


def run_verification_only(setup: EnvironmentSetup, logger: logging.Logger) -> int:
    overall, status = setup.verify_installation()
    log_component_status("\nInstallation status:", status, logger)
    if overall:
        logger.info("\n✓ Installation verified successfully!")
        return 0
    logger.error("\n✗ Installation incomplete")
    return 1


def run_full_setup(args: argparse.Namespace, setup: EnvironmentSetup, logger: logging.Logger) -> int:
    logger.info("Starting Call Analytics System environment setup...")
    logger.info(f"Base directory: {args.base_dir}")

    if not setup.check_python_version():
        return 1
    if args.skip_system_deps:
        logger.info("Skipping system dependency installation (flag provided).")
    else:
        if not setup.install_system_dependencies():
            logger.error("System dependency installation failed.")
            return 1
    if not setup.create_directories():
        return 1
    if not args.skip_packages:
        if not setup.install_packages(upgrade=args.upgrade_packages):
            logger.error("Package installation failed")
            return 1
    if not setup.create_config_files():
        return 1
    if not setup.setup_streamlit_config():
        return 1
    if not args.skip_sample_data:
        setup.create_sample_data()

    overall, status = setup.verify_installation()
    log_component_status("\nInstallation summary:", status, logger)
    if overall:
        setup.print_next_steps()
    else:
        logger.warning("\nSetup completed with some optional components missing.")
        logger.info("The core system should work, but some features may be limited.")
        setup.print_next_steps()
    return 0


def main() -> None:
    args = parse_arguments()
    logger = setup_logging_simple()
    setup = EnvironmentSetup(args.base_dir, logger)

    exit_code = (
        run_verification_only(setup, logger)
        if args.verify_only
        else run_full_setup(args, setup, logger)
    )

    if exit_code:
        sys.exit(exit_code)
```

* What I changed: I split argument parsing, verification-only, and full setup into separate helpers and left `main()` to just orchestrate them.
* Why I changed it: Keeping everything in `main()` made it impossible for me to test individual branches or adjust a single step without scrolling through a huge block.
* How it improved: I now plug in a stubbed `EnvironmentSetup` to test `run_full_setup()` directly, and tweaking CLI flags means touching only `parse_arguments()`. The overall flow is easier for me to reason about.

## 2. Analysis Page Metric Constants

### Original Snippet

*From: `src/ui/pages/analysis.py` (commit `88724ed`), lines 300-329*

```python
with col2:
    metric = st.selectbox(
        "Metric to Track",
        ["Retention Rate", "Average Duration", "Revenue per Cohort", "Call Frequency"],
        key="cohort_metric",
    )

    periods_to_analyze = st.slider(
        "Periods to Analyze", min_value=3, max_value=12, value=6, key="cohort_periods"
    )

# Date range for cohort analysis
st.subheader("Analysis Period")
start_date, end_date = DateRangeFilter.render(
    key_prefix="cohort", default_range="Last 90 Days"
)

if st.button("Generate Cohort Analysis", type="primary", use_container_width=True):
    self._execute_cohort_analysis(
        cohort_type, cohort_period, metric, periods_to_analyze, start_date, end_date
    )
```

* What I struggled with: I repeated the same metric strings across dropdowns, legends, and tooltips, so a single rename meant hunting down every copy.
* Why it mattered: Duplicate literals made mistakes easy (typos, inconsistent labels) and the outdated `use_container_width` flag showed deprecation warnings.
* How it behaved: whenever I added a new chart I copied more strings, increasing the chance of mismatch between widgets.

### Refactored version

*From: `src/ui/pages/analysis.py`, lines 47-71 and 298-333*

```python
RETENTION_RATE = "Retention Rate"
AVERAGE_DURATION = "Average Duration"
REVENUE_PER_COHORT = "Revenue per Cohort"
CALL_FREQUENCY = "Call Frequency"

COHORT_METRIC_OPTIONS = [
    RETENTION_RATE,
    AVERAGE_DURATION,
    REVENUE_PER_COHORT,
    CALL_FREQUENCY,
]

COHORT_HEATMAP_COLORBARS = {
    RETENTION_RATE: "Retention (%)",
    AVERAGE_DURATION: "Minutes",
    REVENUE_PER_COHORT: "Revenue per Cohort",
    CALL_FREQUENCY: "Call Frequency",
}
```

```python
with col2:
    metric = st.selectbox(
        "Metric to Track",
        COHORT_METRIC_OPTIONS,
        key="cohort_metric",
    )

    periods_to_analyze = st.slider(
        "Periods to Analyze", min_value=3, max_value=12, value=6, key="cohort_periods"
    )

st.subheader("Analysis Period")
start_date, end_date = DateRangeFilter.render(
    key_prefix="cohort", default_range="Last 90 Days"
)

if st.button("Generate Cohort Analysis", type="primary", width="stretch"):
    self._execute_cohort_analysis(
        cohort_type, cohort_period, metric, periods_to_analyze, start_date, end_date
    )
```

* What I changed: I introduced named constants for cohort metrics and reused the shared list when populating the select box plus related charts. I also switched the button call to `width="stretch"`.
* Why I changed it: Copying literal strings everywhere made me fix the same typo multiple times and the deprecated `use_container_width` flag kept generating warnings.
* How it improved: I now update the metric vocabulary in one location and every widget stays in sync. The layout code is cleaner and aligned with the current Streamlit API, so I avoid the warning spam.

## 3. Streamlit App Component Setup

### Original Snippet

*From: `src/ui/app.py` (commit `4204b9b`), lines 449-520*

```python
def setup_components(self) -> None:
    try:
        # Create required directories
        for _path_key, path_value in self.config.get("paths", {}).items():
            Path(path_value).mkdir(parents=True, exist_ok=True)

        # Initialize storage manager if not already done
        if st.session_state.storage_manager is None:
            from core.storage_manager import StorageManager

            st.session_state.storage_manager = StorageManager(
                base_path=Path(self.config["paths"]["data"])
            )
            logger.info("Storage manager initialized")
        from vectordb.chroma_client import ChromaClient

        if st.session_state.vector_store is None:
            vector_cfg = self.config.get("vectorstore", {})
            st.session_state.vector_store = ChromaClient(vector_cfg)
            try:
                stats = st.session_state.vector_store.get_statistics()
                if stats.get("total_documents", 0) == 0:
                    data_df = st.session_state.storage_manager.load_all_records()
                    if data_df is not None and not data_df.empty:
                        from vectordb.indexer import DocumentIndexer

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
                        indexer = DocumentIndexer(
                            st.session_state.vector_store, config=indexing_config
                        )
```

* What I struggled with: `setup_components()` mixed directory creation, storage bootstrap, vector-store wiring, and optional reindexing in one long block.
* Why it mattered: when something failed, the session state ended up half-initialised and the logs did not tell me which step broke.
* How it behaved: any adjustment around the vector store risked touching directory or storage logic because everything was interleaved.

### Refactored version

*From: `src/ui/app.py`, lines 449-551*

```python
def setup_components(self) -> None:
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
    for path_value in self.config.get("paths", {}).values():
        Path(path_value).mkdir(parents=True, exist_ok=True)

def _ensure_storage_manager(self) -> None:
    if st.session_state.storage_manager is not None:
        return
    from core.storage_manager import StorageManager
    st.session_state.storage_manager = StorageManager(
        base_path=Path(self.config["paths"]["data"])
    )

def _ensure_vector_store(self) -> None:
    if st.session_state.vector_store is not None:
        return
    from vectordb.chroma_client import ChromaClient
    vector_cfg = self.config.get("vectorstore", {})
    try:
        st.session_state.vector_store = ChromaClient(vector_cfg)
    except RuntimeError as vector_error:
        logger.warning("Vector store disabled: %s", vector_error)
        st.session_state.vector_store = None
        return
    try:
        self._populate_vector_store_if_needed(vector_cfg)
    except Exception as index_error:
        logger.warning("Vector store initialization skipped: %s", index_error)
```

* What I changed: I broke the old mega method into `_ensure_directories`, `_ensure_storage_manager`, `_ensure_vector_store`, and `_ensure_llm_client`, leaving `setup_components()` to call them in sequence.
* Why I changed it: Mixing directory creation, storage setup, vector boot, and reindexing in one place kept leaving me with half-initialised state whenever something went wrong, and the logs did not tell me which step failed.
* How it improved: I run the helpers independently in tests, I can short-circuit optional pieces such as the vector store, and the logs now point at the precise stage that needs attention.

## 4. Embedding Manager Cache Flow

### Original Snippet

*From: `src/ml/embeddings.py` (commit `02b8125`), lines 166-336*

```python
class HashEmbeddingProvider(EmbeddingProvider):
    ...
    def generate(self, texts: list[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            text_hash = hashlib.sha256(f"{self.seed}:{text}".encode()).digest()
            np.random.seed(int.from_bytes(text_hash[:4], "little"))
            embedding = np.random.randn(self.dimension)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        return np.array(embeddings)

def generate_embeddings(
    self,
    texts: list[str],
    use_cache: bool = True,
) -> np.ndarray:
    if not texts:
        return np.array([])

    embeddings = []
    texts_to_generate = []
    text_indices = []

    if use_cache and self.cache_enabled:
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                embeddings.append(self.cache[cache_key])
            else:
                texts_to_generate.append(text)
                text_indices.append(i)
    else:
        texts_to_generate = texts
        text_indices = list(range(len(texts)))

    if texts_to_generate:
        new_embeddings = self.provider.generate(texts_to_generate)
        if self.cache_enabled:
            for text, embedding in zip(texts_to_generate, new_embeddings, strict=False):
                cache_key = self._get_cache_key(text)
                self.cache[cache_key] = embedding
        if use_cache and self.cache_enabled:
            result = np.zeros((len(texts), self.provider.get_dimension()))
            cache_idx = 0
            for i, _text in enumerate(texts):
                if i not in text_indices:
                    result[i] = embeddings[cache_idx]
                    cache_idx += 1
            for i, idx in enumerate(text_indices):
                result[idx] = new_embeddings[i]
            return result
        else:
            return new_embeddings
    else:
        return np.array(embeddings)
```

* What I struggled with: the fallback provider reset NumPy’s global RNG for every text and the cache path collected embeddings in temporary lists.
* Why it mattered: resetting the global RNG changed randomness in other modules, and the list juggling made me unsure the output order stayed aligned with the inputs.
* How it behaved: batches with mixed cached and fresh embeddings were hard to reason about and fragile under tests.

### Refactored version

*From: `src/ml/embeddings.py`, lines 166-360*

```python
class HashEmbeddingProvider(EmbeddingProvider):
    def generate(self, texts: list[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            text_hash = hashlib.sha256(f"{self.seed}:{text}".encode()).digest()
            seed_value = int.from_bytes(text_hash[:4], "little")
            generator = np.random.default_rng(seed_value)
            embedding = generator.standard_normal(self.dimension)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        return np.array(embeddings)

def generate_embeddings(
    self,
    texts: list[str],
    use_cache: bool = True,
) -> np.ndarray:
    if not texts:
        return np.array([])

    cache_active = use_cache and self.cache_enabled

    if cache_active:
        cached_embeddings, missing_indices, texts_to_generate = self._split_cache_hits(texts)
        if not texts_to_generate:
            return self._assemble_cached_results(len(texts), cached_embeddings)
    else:
        cached_embeddings = {}
        missing_indices = list(range(len(texts)))
        texts_to_generate = list(texts)

    new_embeddings = self.provider.generate(texts_to_generate)

    if self.cache_enabled:
        self._store_in_cache(texts_to_generate, new_embeddings)

    if cache_active:
        return self._assemble_results(
            len(texts),
            cached_embeddings,
            missing_indices,
            new_embeddings,
        )

    return new_embeddings
```

* What I changed: I replaced the global `np.random.seed` usage with a per-call `np.random.default_rng`, and I extracted `_split_cache_hits`, `_store_in_cache`, `_assemble_results`, and `_assemble_cached_results`.
* Why I changed it: Resetting the global RNG corrupted other parts of the pipeline, and the old caching path used ad-hoc lists that made me nervous about mixing up the order of embeddings.
* How it improved: Each batch now generates embeddings deterministically without leaking state, and the separate helpers keep the cache logic easy for me to trace and assert in tests.

## 5. CSV Processor Field Matching

### Original Snippet

*From: `src/core/csv_processor.py` (commit `8fe2fbe2`), lines 270-420*

```python
def _match_field_to_header(
    self,
    field: dict[str, Any],
    headers: list[str],
    normalized_headers: dict[str, str],
    used_headers: set[str],
) -> str | None:
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

def parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
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

            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col])
                    logger.info(f"Parsed {col} with auto-detection")
                except (TypeError, ValueError):
                    logger.warning(f"Could not parse dates in column {col}")

    return df
```

* What I struggled with: I handled schema aliases, regex patterns, and header fallbacks inside one method, so I could not tell which rule picked a column.
* Why it mattered: debugging customer CSV files meant stepping through the entire function and guessing where the decision happened.
* How it behaved: the date parsing loop also mixed manual formats with the fallback, so errors were hard for me to isolate.

### Refactored version

*From: `src/core/csv_processor.py`, lines 308-520*

```python
def _match_field_to_header(
    self,
    field: dict[str, Any],
    headers: list[str],
    normalized_headers: dict[str, str],
    used_headers: set[str],
) -> str | None:
    name = field.get("name")
    if not name:
        return None

    candidates = self._build_candidate_headers(field, name)
    available_headers = self._available_headers(headers, used_headers)

    direct_match = self._find_direct_header_match(
        available_headers, normalized_headers, candidates
    )
    if direct_match:
        return direct_match

    return self._find_pattern_header_match(name, available_headers, normalized_headers)

def _build_candidate_headers(self, field: dict[str, Any], name: str) -> set[str]:
    candidates = {self._normalize_header(name)}
    label = field.get("label")
    if label:
        candidates.add(self._normalize_header(label))
    aliases = self.field_alias_map.get(name, [])
    candidates.update(self._normalize_header(alias) for alias in aliases)
    return candidates

def _find_pattern_header_match(
    self,
    name: str,
    headers: list[str],
    normalized_headers: dict[str, str],
) -> str | None:
    for pattern in self._iter_compiled_patterns(name):
        match = self._match_header_by_pattern(pattern, headers, normalized_headers)
        if match:
            return match
    return None

def parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
    for column in self._iter_date_columns(df):
        self._parse_date_column(df, column)
    return df

def _iter_date_columns(df: pd.DataFrame) -> list[str]:
    desired_columns = ["timestamp", "start_time", "end_time", "created_at", "updated_at"]
    return [column for column in desired_columns if column in df.columns]

def _parse_date_column(self, df: pd.DataFrame, column: str) -> None:
    if pd.api.types.is_datetime64_any_dtype(df[column]):
        return
    if self._parse_with_known_formats(df, column):
        return
    self._parse_with_auto_detection(df, column)
```

* What I changed: I refactored field matching into small helpers (for candidate headers, direct matches, regex matches) and broke the date parsing flow into `_iter_date_columns`, `_parse_date_column`, `_parse_with_known_formats`, and `_parse_with_auto_detection`.
* Why I changed it: When customers sent odd CSV headers I could not see which branch failed, and the mix of format-specific parsing with the fallback made debugging inconsistent.
* How it improved: I now log the exact helper that resolves each field, test each branch independently, and understand whether a date column used a configured format or fell back to automatic parsing.
