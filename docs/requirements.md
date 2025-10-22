# Requirements

This document outlines the detailed functional and non-functional requirements for the Call Analytics System, a locally-hosted solution designed to process, analyze, and query call data using Python 3.11, Streamlit, and ChromaDB. The requirements ensure the system delivers robust audio transcription, semantic search, natural language querying, and comprehensive analytics while prioritizing performance, privacy, and maintainability.

## Functional Requirements

1. **Audio Transcription Capability**  
   The system shall provide automatic transcription of audio call recordings in supported formats (WAV, MP3, M4A, FLAC) using the OpenAI Whisper speech-to-text model.

   - Users shall be able to configure the Whisper model size (tiny, base, small, medium, large) in `config/models.toml` to balance transcription accuracy and processing speed.
   - The system shall support configurable compute types (int8, float16, float32) and device selection (auto, CPU, CUDA) to optimize performance based on hardware capabilities.
   - Audio files shall be processed with a maximum duration of 60 minutes, a sample rate of 16,000 Hz, and single-channel (mono) audio, as specified in `config/app.toml`.
   - The system shall validate audio files for compatibility and provide clear error messages if unsupported formats or corrupted files are detected.

2. **Semantic Search Functionality**  
   The system shall enable vector-based semantic search using ChromaDB and sentence transformers to allow users to query call transcripts with natural language inputs.

   - Search queries shall support filtering by metadata fields such as `agent_id`, `call_type`, `campaign`, or `start_time`, with results limited to a configurable top-k (default: 10) matches.
   - The system shall use the `all-MiniLM-L6-v2` sentence transformer model (configurable in `config/vectorstore.toml`) for generating text embeddings, with cosine distance as the default metric for similarity scoring.
   - Search results shall be accessible via the Streamlit UI and programmatically through the `vector_db.search` API, returning call metadata and relevant transcript snippets.
   - The system shall handle queries efficiently, with an average response time of under 2 seconds for datasets containing up to 1,000 transcripts.

3. **Natural Language Query Interface**  
   The system shall provide a natural language query interface allowing users to ask questions in plain English (e.g., “What were the main complaints last week?” or “Show refund requests from agent John”).

   - Queries shall be processed using a local language model integrated via `ml/llm_interface.py`, with results displayed in the Streamlit UI under the Q&A interface (`ui/pages/qa_interface.py`).
   - The system shall interpret queries by mapping them to structured filters and semantic search operations, supporting complex queries involving time ranges, call types, outcomes, or specific agents.
   - Query results shall include summarized answers, relevant call metadata, and links to full transcripts where applicable.
   - The system shall provide example queries in the UI to guide users in formulating effective questions.

4. **Automated Call Labeling**  
   The system shall automatically categorize calls by type (e.g., inquiry, support, billing, complaint) and outcome (e.g., resolved, escalated, unresolved) based on configurable rules.

   - Labeling rules shall be defined in `config/rules.toml`, using keyword-based classification (e.g., “unhappy” or “poor service” for complaints) and optional machine learning models in `ml/labeling_engine.py`.
   - Users shall be able to customize and extend rules to support additional call types or outcomes without modifying the core codebase.
   - The system shall apply labels during audio transcription or CSV import, storing results in the `data/processed/` directory alongside call metadata.
   - Labeling accuracy shall be validated through periodic manual review, with a target accuracy of at least 90% for keyword-based classification.

5. **Interactive Analytics Dashboard**  
   The system shall provide a Streamlit-based dashboard (`ui/pages/dashboard.py`) for visualizing call analytics and key performance indicators (KPIs).
   - The dashboard shall display metrics such as total calls, average call duration, call type distribution, agent performance, and peak call hours, with interactive charts and tables (`ui/components/charts.py`, `ui/components/tables.py`).
   - Users shall be able to apply filters by date range, call type, outcome, agent, or campaign, with filter controls implemented in `ui/components/filters.py`.
   - Analytics shall be computed using pre-aggregated data (`analysis/aggregations.py`) to ensure fast rendering, with an average dashboard load time of under 3 seconds.
   - The system shall support exporting dashboard data as CSV files for external analysis.

## Non-Functional Requirements

6. **Local Processing and Privacy**  
   The system shall perform all data processing tasks (audio transcription, embedding generation, query execution, and analytics) locally on the user’s machine to ensure data privacy.

   - No external API calls shall be made by default, and all dependencies (e.g., Whisper, sentence transformers) shall use locally stored models in the `models/` directory.
   - The system shall support optional PII (Personally Identifiable Information) masking in transcripts, configurable via `config/app.toml`, to further enhance privacy.
   - Data shall be stored locally in the `data/` and `vectorstore/` directories, with optional encryption support for sensitive datasets.
   - The system shall comply with privacy regulations by ensuring no data is transmitted over the internet unless explicitly configured by the user.

7. **Performance Optimization**  
   The system shall be optimized for efficient processing on modern workstations with 8GB+ RAM and optional CUDA-capable GPUs.

   - Audio transcription with Whisper shall leverage GPU acceleration when available, achieving transcription speeds of at least 1x real-time for the `small.en` model on CUDA-enabled hardware.
   - The system shall implement caching for embeddings and analytics results, configurable in `config/app.toml`, to reduce redundant computations.
   - Batch processing (`scripts/rebuild_index.py`) shall support configurable batch sizes (default: 50) to handle large datasets without memory exhaustion.
   - Semantic search and query operations shall maintain an average response time of under 2 seconds for datasets with up to 1,000 call records.

8. **Scalability and Robustness**  
   The system shall scale to process up to 1,000 call recordings (each up to 60 minutes) in a single batch without performance degradation.

   - The system shall handle datasets with up to 10,000 total call records in the `data/` directory, with periodic vector index rebuilding to maintain search performance.
   - Error handling shall ensure graceful recovery from invalid audio files, corrupted CSVs, or memory constraints, logging issues to `logs/` for debugging.
   - The system shall support concurrent processing of multiple audio files or CSV imports, utilizing multi-threading where appropriate to maximize CPU utilization.
   - Robustness shall be validated through automated tests in `tests/`, targeting at least 90% code coverage for critical modules (`src/core/`, `src/analysis/`).

9. **Ease of Installation and Setup**  
   The system shall provide a streamlined installation process compatible with macOS, Ubuntu/Debian, and Windows environments.

   - Installation shall require Python 3.11, FFmpeg, and build tools (Ant, Maven), with setup instructions documented in `README.md`.
   - Scripts (`scripts/setup_environment.py`, `scripts/download_models.py`) shall automate virtual environment creation, dependency installation, and model downloading.
   - The system shall validate dependencies (e.g., `python3.11 --version`, `ffmpeg -version`) during setup and provide clear error messages if prerequisites are missing.
   - The Streamlit application shall launch automatically at `http://localhost:8501` after setup, with a first-time setup time of under 10 minutes on a modern workstation.

10. **Maintainability and Extensibility**  
    The system shall be designed with a modular architecture to facilitate maintenance and future enhancements.
    - Source code in `src/` shall be organized into clear modules (`core/`, `analysis/`, `ml/`, `vectordb/`, `ui/`) with well-defined interfaces and documentation.
    - Configuration files (`config/*.toml`) shall allow users to customize settings (e.g., model sizes, labeling rules, vector store parameters) without code changes.
    - The system shall support development workflows with tools like pytest, black, ruff, and mypy, integrated via `build.xml` and `pom.xml` for CI/CD pipelines.
    - Documentation in `docs/` shall be generated using Sphinx, covering setup, usage, and API details, with HTML output accessible in `docs/_build/html`.
    - Developers shall be able to extend functionality (e.g., adding new call types, metrics, or UI components) with minimal changes to existing code, supported by clear contribution guidelines in `README.md`.
