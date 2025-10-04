# 📞 Call Analytics System

A professional, locally-hosted call analytics system with speech-to-text, semantic search, and natural language Q&A capabilities. Built with Python 3.13, Streamlit, and ChromaDB.

[![Python 3.13](https://img.shields.io/badge/Python-3.13-informational?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-informational?logo=streamlit)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vectorDB-informational)](https://www.trychroma.com/)
[![Whisper](https://img.shields.io/badge/Whisper-STT-informational?logo=openai)](https://github.com/openai/whisper)
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

[↑ Back to top](#readme)
</details>

<details>
  <summary><strong>🚀 Quick Start</strong></summary>

<a id="quick-start"></a>

### Prerequisites
- Python 3.13 or higher  
- FFmpeg (for audio processing)  
- 8GB+ RAM recommended  
- CUDA-capable GPU (optional, for faster processing)

### Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/mujtaba-a-khan/call-analytics-system.git
   cd call-analytics-system
   ```
2. **Install FFmpeg**
   ```bash
   # macOS
   brew install ffmpeg

   # Ubuntu/Debian
   sudo apt-get install ffmpeg

   # Windows: download from https://ffmpeg.org/download.html
   ```
3. **Create virtual environment**
   ```bash
   python3.13 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install dependencies**
   ```bash
   pip install -e .
   # For development
   pip install -e ".[dev]"
   # For documentation
   pip install -e ".[docs]"
   ```
5. **Run Scripts to Setup Enviroment**
   ```bash
   python scripts/setup_enviroment.py

   python scripts/download_models.py
   ```
6. **Run the application**
   ```bash
   streamlit run src/ui/app.py
   ```
App opens at `http://localhost:8501`.

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
├── requirements.txt               # Alternative dependency list
│
├── config/                        # Configuration files
│   ├── app.toml                   # Main application settings
│   ├── models.toml                # LLM and STT model configurations
│   ├── vectorstore.toml           # Vector database settings
│   └── rules.toml                 # Call labeling rules
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── audio_processor.py     # Audio file processing
│   │   ├── csv_processor.py       # CSV file processing
│   │   ├── data_schema.py         # Data models and schemas
│   │   ├── labeling_engine.py     # Call labeling logic
│   │   └── storage_manager.py     # Data persistence
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── filters.py             # Data filtering logic
│   │   ├── aggregations.py        # KPIs and metrics
│   │   ├── semantic_search.py     # Semantic search implementation
│   │   └── query_interpreter.py   # Natural language query processing
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── whisper_stt.py         # Speech-to-text engine
│   │   ├── llm_interface.py       # Local LLM integration
│   │   └── embeddings.py          # Text embedding generation
│   ├── vectordb/
│   │   ├── __init__.py
│   │   ├── chroma_client.py       # ChromaDB interface
│   │   ├── indexer.py             # Document indexing
│   │   └── retriever.py           # Document retrieval
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── app.py                 # Main Streamlit application
│   │   ├── pages/
│   │   │   ├── __init__.py
│   │   │   ├── dashboard.py       # Main dashboard
│   │   │   ├── upload.py          # File upload interface
│   │   │   ├── analysis.py        # Analysis view
│   │   │   └── qa_interface.py    # Q&A interface
│   │   └── components/
│   │       ├── __init__.py
│   │       ├── charts.py          # Chart components
│   │       ├── filters.py         # Filter components
│   │       ├── tables.py          # Table components
│   │       └── metrics.py         # Metric display components
│   └── utils/
│       ├── __init__.py
│       ├── text_processing.py     # Text utilities
│       ├── file_handlers.py       # File I/O utilities
│       ├── formatters.py          # Formatting Utilities
│       ├── validators.py          # Data validation
│       └── logger.py              # Logging configuration
│
├── data/                          # Data directory
│   ├── uploads/                   # User uploaded files
│   ├── processed/                 # Processed data
│   ├── cache/                     # Cache directory
│   └── vectorstore/               # Vector database storage
├── models/                        # Model storage
│   └── whisper/                   # Whisper models
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── test_core/
│   ├── test_analysis/
│   └── test_ml/
└── scripts/
    ├── setup_environment.py
    ├── download_models.py
    └── rebuild_index.py
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

Contributions welcome! Please read `CONTRIBUTING.md`.

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
