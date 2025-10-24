> 📘 Read the docs on GitHub Pages: https://mujtaba-a-khan.github.io/call-analytics-system/

# 📞 Call Analytics System

A professional, locally-hosted call analytics system with speech-to-text, semantic search, and natural language Q&A capabilities. Built with Python 3.11, Streamlit, and ChromaDB.

📚 Documentation: https://mujtaba-a-khan.github.io/call-analytics-system/

[![Python 3.11](https://img.shields.io/badge/Python-3.11-informational?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-informational?logo=streamlit)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vectorDB-informational)](https://www.trychroma.com/)
[![Whisper](https://img.shields.io/badge/Whisper-STT-informational?logo=openai)](https://github.com/openai/whisper)
[![Docs](https://img.shields.io/badge/Docs-Latest-blue?logo=readthedocs)](https://mujtaba-a-khan.github.io/call-analytics-system/)
[![License: MIT](https://img.shields.io/badge/License-MIT-success)](LICENSE)

## Table of Contents

<details>
  <summary><strong>✨ Features</strong></summary>

<a id="features"></a>

- **🎵 Audio Processing**: Automatic transcription of call recordings using Whisper STT  
- **📄 CSV Import**: Bulk import of existing call transcripts  
- **🏷️ Intelligent Labeling**: Automatic categorization of calls by type and outcome  
- **🔍 Semantic Search**: Vector-based search using ChromaDB and sentence transformers  
- **❓ Natural Language Q&A**: Query your data using plain English  
- **📊 Rich Analytics**: Interactive dashboards with metrics and visualizations  
- **🔒 Privacy-First**: All processing happens locally—no data leaves your machine  
- **⚡ High Performance**: Efficient caching and batch processing capabilities  
- **✅ Requirements Tracker**: FastAPI-powered backlog manager.

[↑ Back to top](#readme)
</details>

<details>
  <summary><strong>🚀 Quick Start</strong></summary>

<a id="quick-start"></a>

### Prerequisites
- Python 3.11 or higher  
- FFmpeg (for audio processing)  
- 8GB+ RAM recommended  
- CUDA-capable GPU (optional, for faster processing)

### Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/mujtaba-a-khan/call-analytics-system.git
   cd call-analytics-system
   ```
2. **Install system tooling**
   ```bash
   # macOS (Homebrew)
   brew install python@3.11 ffmpeg maven ant zip gcc

   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install -y python3.11 python3.11-venv python3.11-dev ffmpeg build-essential zip maven ant

   # Windows: download from https://ffmpeg.org/download.html
   # - Python 3.11: https://www.python.org/downloads/
   # - Maven: https://maven.apache.org/install.html
   # - Ant: https://ant.apache.org/bindownload.cgi
   ```
   Confirm your installations:
   ```bash
   python3.11 --version
   ffmpeg -version
   ant -version
   mvn -version
   ```
3. **Create virtual environment**
   ```bash
   python3.11 -m venv .venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies**
   ```bash
   pip install -e .
   # For development
   pip install -e ".[dev,tools]"
   # For documentation
   pip install -e ".[docs]"
   ```
   The `tools` extra installs FastAPI, SQLModel, and Jinja so the embedded requirements tracker and its tests run out of the box.
5. **Run scripts to bootstrap assets**
   ```bash
   python scripts/setup_environment.py
   python scripts/download_models.py
   ```
6. **Run the application**
   ```bash
   call-analytics-ui
   ```
App opens at `http://localhost:8501`.

[↑ Back to top](#readme)
</details>

<details>
  <summary><strong>🛠️ Build & CI Tooling</strong></summary>

<a id="build-ci"></a>

Run these commands from the repository root once the virtual environment is activated. The Ant tasks expect `python3.11`, `python3.11-venv`, `python3.11-dev`, `build-essential`, `zip`, and FFmpeg to be available on the host (see Quick Start).

### Apache Ant (`build.xml`)
```bash
ant -noinput -buildfile build.xml clean   # wipe build outputs
ant -noinput -buildfile build.xml setup   # create .venv & install project deps
ant -noinput -buildfile build.xml lint    # ruff, black --check, mypy
ant -noinput -buildfile build.xml test    # pytest with junit report
ant -noinput -buildfile build.xml docs    # sphinx html docs in docs/_build/html
ant -noinput -buildfile build.xml wheel   # build Python wheel into dist/
ant -noinput -buildfile build.xml ci      # run the full clean→wheel pipeline
```

Outputs land in `dist/`, `docs/_build`, and `test-reports/`.

Need a clean Codespaces setup? The dev container (`.devcontainer/devcontainer.json`) pulls in system packages via apt-get and auto-launches the Streamlit UI once the workspace is ready.

### Apache Maven (`pom.xml`)
```bash
mvn -B -V verify    # runs the Ant ci target via maven-antrun-plugin
mvn -B package      # zips release assets to artifacts/call-analytics-1.0.0.zip
```

Maven requires Ant to be on the PATH because the `verify` phase delegates to `build.xml`. The generated ZIP bundles source, configs, and documentation for deployment.

[↑ Back to top](#readme)
</details>

<details>
  <summary><strong>📁 Project Structure</strong></summary>

<a id="project-structure"></a>

```text
call-analytics-system/
│
├── pyproject.toml                 # Project configuration and dependencies
├── README.md                      # Project documentation
├── .gitignore                     # Git ignore file
├── Jenkinsfile                    # Jenkins CI pipeline definition
├── build.xml                      # Ant build configuration for Jenkins agents
├── pom.xml                        # Maven build configuration for Jenkins agents
│
├── config/                        # Configuration files for the analytics engine
│   ├── app.toml                   # Main application settings
│   ├── fields.toml                # Column mapping for ingestion
│   ├── models.toml                # LLM and STT model configurations
│   ├── rules.toml                 # Call labeling rules
│   └── vectorstore.toml           # Vector database settings
│
├── docs/                          # Sphinx documentation sources
│   └── ...                        # See docs/index.md for table of contents
│
├── scripts/                       # Utility scripts for local workflows
│   ├── setup_environment.py       # Bootstrap virtual environments and deps
│   ├── download_models.py         # Fetch Whisper/LLM assets
│   ├── rebuild_index.py           # Rebuild Chroma vector indexes
│   └── launch_ui.py               # Convenience launcher for the Streamlit UI
│
├── src/                           # Application source code
│   ├── cli.py                     # Command-line entry point
│   ├── core/
│   │   ├── audio_processor.py     # Audio file processing
│   │   ├── csv_processor.py       # CSV file processing
│   │   ├── data_schema.py         # Data models and schemas
│   │   ├── labeling_engine.py     # Call labeling logic
│   │   └── storage_manager.py     # Data persistence
│   ├── analysis/
│   │   ├── aggregations.py        # KPIs and metrics
│   │   ├── filters.py             # Data filtering logic
│   │   ├── query_interpreter.py   # Natural language query processing
│   │   └── semantic_search.py     # Semantic search implementation
│   ├── ml/
│   │   ├── embeddings.py          # Text embedding generation
│   │   ├── llm_interface.py       # Local LLM integration
│   │   └── whisper_stt.py         # Speech-to-text engine
│   ├── vectordb/
│   │   ├── chroma_client.py       # ChromaDB interface
│   │   ├── indexer.py             # Document indexing
│   │   └── retriever.py           # Document retrieval
│   ├── ui/
│   │   ├── app.py                 # Main Streamlit application
│   │   ├── pages/
│   │   │   ├── analysis.py        # Analysis view
│   │   │   ├── dashboard.py       # Main dashboard
│   │   │   ├── qa_interface.py    # Q&A interface
│   │   │   └── upload.py          # File upload interface
│   │   └── components/
│   │       ├── charts.py          # Chart components
│   │       ├── filters.py         # Filter components
│   │       ├── metrics.py         # Metric display components
│   │       └── tables.py          # Table components
│   └── utils/
│       ├── file_handlers.py       # File I/O utilities
│       ├── formatters.py          # Formatting utilities
│       ├── logger.py              # Logging configuration
│       ├── text_processing.py     # Text utilities
│       └── validators.py          # Data validation
│
├── data/                          # Working datasets and cached artifacts
│   ├── raw/                       # Uploaded raw sources
│   ├── processed/                 # Normalized transcripts
│   └── vectorstore/               # Persisted embeddings
│
├── models/                        # Model registry and downloaded weights
├── logs/                          # Aggregated runtime logs
├── test-reports/                  # Collected junit/coverage artifacts
└── tests/                         # Automated test suite
    ├── test_aggregations.py       # KPI aggregation coverage
    └── test_text_processing.py    # Text utility coverage
```

[↑ Back to top](#readme)
</details>

<details>
  <summary><strong>🔧 Configuration</strong></summary>

<a id="configuration"></a>

### Audio Settings (`config/app.toml`)
```toml
[audio]
supported_formats = ["wav", "mp3", "m4a", "flac"]
max_duration_minutes = 60
sample_rate = 16000
channels = 1
```

### Model Settings (`config/models.toml`)
```toml
[whisper]
model_size = "small.en"  # tiny, base, small, medium, large
compute_type = "int8"    # int8, float16, float32
device = "auto"          # auto, cpu, cuda
```

### Vector Store (`config/vectorstore.toml`)
```toml
[vectorstore]
provider = "chromadb"
collection_name = "call_transcripts"
distance_metric = "cosine"

[embeddings]
provider = "sentence-transformers"
model_name = "all-MiniLM-L6-v2"
```

[↑ Back to top](#readme)
</details>


<details>
  <summary><strong>📊 Usage Guide</strong></summary>

<a id="usage-guide"></a>

### 1) Upload Files
- **Audio**: WAV, MP3, M4A, FLAC  
- **CSV**: transcripts with metadata

Required columns: `call_id`, `start_time`, `duration_seconds`, `transcript`  
Optional: `agent_id`, `campaign`, `customer_name`, `product_name`, `amount`

### 2) Apply Filters
By date, type (Inquiry/Support/Billing/Complaint), outcome, agent, campaign.

### 3) View Analytics
KPIs, distributions, agent performance, peak hours.

### 4) Natural Language Q&A
Examples:
- “What were the main complaints last week?”
- “Show refund requests from agent John”
- “Calls about billing issues over 5 minutes”

[↑ Back to top](#readme)
</details>


<details>
  <summary><strong>🔬 Advanced Features</strong></summary>

<a id="advanced-features"></a>

**Semantic Search**
```python
results = vector_db.search(
    query_text="customer asking for refund",
    top_k=10,
    filter_dict={"agent_id": "john_doe"}
)
```

**Custom Labeling Rules** (`config/rules.toml`)
```toml
[rules.call_types]
inquiry = ["information", "question", "how to"]
support = ["not working", "error", "problem", "issue"]
billing = ["invoice", "payment", "charge", "bill"]
complaint = ["unhappy", "disappointed", "poor service"]
```

**Batch Processing**
```bash
python scripts/rebuild_index.py --batch-size 50
```

**Requirements Tracker API**
```bash
uvicorn tools.requirements_tracker.app:app
```
Seeds default backlog items even if the JSON fixture is missing and exposes CRUD endpoints at `http://localhost:8000/requirements`.

[↑ Back to top](#readme)
</details>


<details>
  <summary><strong>🛠️ Development</strong></summary>

<a id="development"></a>

```bash
pytest tests/ -v --cov=src      # tests
black src/ tests/               # formatting
ruff check src/ tests/          # lint
mypy src/                       # types
# docs
cd docs && make html
```

[↑ Back to top](#readme)
</details>

<details>
  <summary><strong>📈 Performance Tips</strong></summary>

<a id="performance-tips"></a>

1. Use CUDA/GPU for Whisper  
2. Enable caching in `config/app.toml`  
3. Batch processing for large imports  
4. Rebuild vector index periodically  

[↑ Back to top](#readme)
</details>


<details>
  <summary><strong>🔒 Security & Privacy</strong></summary>

<a id="security--privacy"></a>

- Local processing only  
- No external APIs by default  
- Optional PII masking  
- Local storage with optional encryption  

[↑ Back to top](#readme)
</details>


<details>
  <summary><strong>🐛 Troubleshooting</strong></summary>

<a id="troubleshooting"></a>

**FFmpeg not found**
```bash
ffmpeg -version
# Add to PATH if needed:
export PATH=$PATH:/path/to/ffmpeg
```

**Out of memory**  
Reduce batch size; process smaller groups; increase swap.

**Slow transcription**  
Use smaller Whisper model; enable GPU; reduce audio quality.

[↑ Back to top](#readme)
</details>


<details>
  <summary><strong>📝 License</strong></summary>

<a id="license"></a>

MIT License — see [LICENSE](LICENSE).

[↑ Back to top](#readme)
</details>


<details>
  <summary><strong>🤝 Contributing</strong></summary>

<a id="contributing"></a>

Contributions welcome!

[↑ Back to top](#readme)
</details>

<details>
  <summary><strong>📧 Support</strong></summary>

<a id="support"></a>

- Open an issue on GitHub  
- Check `/docs`  
- Review closed issues

[↑ Back to top](#readme)
</details>


<details>
  <summary><strong>🙏 Acknowledgments</strong></summary>

<a id="acknowledgments"></a>

- OpenAI Whisper for speech-to-text  
- ChromaDB for vector storage  
- Streamlit for the UI  
- The open-source community

[↑ Back to top](#readme)
</details>
