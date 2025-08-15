# 📞 Call Analytics System

A professional, locally-hosted call analytics system with speech-to-text, semantic search, and natural language Q&A capabilities. Built with Python 3.13, Streamlit, and ChromaDB.

## ✨ Features

- **🎵 Audio Processing**: Automatic transcription of call recordings using Whisper STT
- **📄 CSV Import**: Bulk import of existing call transcripts
- **🏷️ Intelligent Labeling**: Automatic categorization of calls by type and outcome
- **🔍 Semantic Search**: Vector-based search using ChromaDB and sentence transformers
- **❓ Natural Language Q&A**: Query your data using plain English
- **📊 Rich Analytics**: Interactive dashboards with metrics and visualizations
- **🔒 Privacy-First**: All processing happens locally - no data leaves your machine
- **⚡ High Performance**: Efficient caching and batch processing capabilities

## 🚀 Quick Start

### Prerequisites

- Python 3.13 or higher
- FFmpeg (for audio processing)
- 8GB+ RAM recommended
- CUDA-capable GPU (optional, for faster processing)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/call-analytics-system.git
cd call-analytics-system
```

2. **Install FFmpeg**
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
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

5. **Download Whisper model**
```bash
python scripts/download_models.py
```

6. **Run the application**
```bash
streamlit run src/ui/app.py
```

The application will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
call-analytics-system/
│
├── pyproject.toml                 # Project configuration and dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore file
├── requirements.txt                # Alternative dependency list
│
├── config/                         # Configuration files
│   ├── app.toml                   # Main application settings
│   ├── models.toml                # LLM and STT model configurations
│   ├── vectorstore.toml           # Vector database settings
│   └── rules.toml                 # Call labeling rules
│
├── src/                           # Source code
│   ├── __init__.py
│   │
│   ├── core/                      # Core business logic
│   │   ├── __init__.py
│   │   ├── audio_processor.py    # Audio file processing
│   │   ├── csv_processor.py      # CSV file processing
│   │   ├── data_schema.py        # Data models and schemas
│   │   ├── labeling_engine.py    # Call labeling logic
│   │   └── storage_manager.py    # Data persistence
│   │
│   ├── analysis/                  # Analysis modules
│   │   ├── __init__.py
│   │   ├── filters.py            # Data filtering logic
│   │   ├── aggregations.py       # KPIs and metrics
│   │   ├── semantic_search.py    # Semantic search implementation
│   │   └── query_interpreter.py  # Natural language query processing
│   │
│   ├── ml/                        # Machine learning components
│   │   ├── __init__.py
│   │   ├── whisper_stt.py        # Speech-to-text engine
│   │   ├── llm_interface.py      # Local LLM integration
│   │   └── embeddings.py         # Text embedding generation
│   │
│   ├── vectordb/                  # Vector database
│   │   ├── __init__.py
│   │   ├── chroma_client.py      # ChromaDB interface
│   │   ├── indexer.py            # Document indexing
│   │   └── retriever.py          # Document retrieval
│   │
│   ├── ui/                        # User interface
│   │   ├── __init__.py
│   │   ├── app.py                # Main Streamlit application
│   │   ├── pages/                # Streamlit pages
│   │   │   ├── __init__.py
│   │   │   ├── dashboard.py      # Main dashboard
│   │   │   ├── upload.py         # File upload interface
│   │   │   ├── analysis.py       # Analysis view
│   │   │   └── qa_interface.py   # Q&A interface
│   │   └── components/            # Reusable UI components
│   │       ├── __init__.py
│   │       ├── charts.py         # Chart components
│   │       ├── filters.py        # Filter components
│   │       ├── tables.py         # Table components
│   │       └── metrics.py        # Metric display components
│   │
│   └── utils/                     # Utility functions
│       ├── __init__.py
│       ├── text_processing.py    # Text utilities
│       ├── file_handlers.py      # File I/O utilities
│       ├── validators.py         # Data validation
│       └── logger.py             # Logging configuration
│
├── data/                          # Data directory
│   ├── uploads/                   # User uploaded files
│   ├── processed/                 # Processed data
│   ├── cache/                     # Cache directory
│   └── vectorstore/               # Vector database storage
│
├── models/                        # Model storage
│   └── whisper/                   # Whisper models
│
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── test_core/                # Core module tests
│   ├── test_analysis/            # Analysis module tests
│   └── test_ml/                  # ML module tests
│
└── scripts/                       # Utility scripts
    ├── setup_environment.py       # Environment setup
    ├── download_models.py         # Model download script
    └── rebuild_index.py          # Vector index rebuilding
```

## 🔧 Configuration

### Audio Settings (config/app.toml)
```toml
[audio]
supported_formats = ["wav", "mp3", "m4a", "flac"]
max_duration_minutes = 60
sample_rate = 16000
channels = 1
```

### Model Settings (config/models.toml)
```toml
[whisper]
model_size = "small.en"  # Options: tiny, base, small, medium, large
compute_type = "int8"    # Options: int8, float16, float32
device = "auto"          # Options: auto, cpu, cuda
```

### Vector Store Settings (config/vectorstore.toml)
```toml
[vectorstore]
provider = "chromadb"
collection_name = "call_transcripts"
distance_metric = "cosine"

[embeddings]
provider = "sentence-transformers"
model_name = "all-MiniLM-L6-v2"
```

## 📊 Usage Guide

### 1. Upload Files

- **Audio Files**: Upload WAV, MP3, M4A, or FLAC files
- **CSV Files**: Import existing transcripts with metadata

Required CSV columns:
- `call_id`: Unique identifier
- `start_time`: Timestamp (ISO format)
- `duration_seconds`: Call duration
- `transcript`: Call transcript text

Optional columns:
- `agent_id`, `campaign`, `customer_name`, `product_name`, `amount`

### 2. Apply Filters

Use the sidebar to filter calls by:
- Date range
- Call type (Inquiry, Support, Billing/Sales, Complaint)
- Outcome (Resolved, Callback, Refund, Sale-close)
- Agent
- Campaign

### 3. View Analytics

The dashboard provides:
- Key metrics (total calls, connection rate, average duration)
- Distribution charts (types, outcomes, timeline)
- Agent performance metrics
- Peak hours and busy days analysis

### 4. Natural Language Q&A

Ask questions like:
- "What were the main complaints last week?"
- "Show me all refund requests from agent John"
- "Find calls about billing issues that lasted over 5 minutes"

## 🔬 Advanced Features

### Semantic Search

The system uses ChromaDB with sentence transformers to enable semantic search:

```python
# Example: Find similar calls
results = vector_db.search(
    query_text="customer asking for refund",
    top_k=10,
    filter_dict={"agent_id": "john_doe"}
)
```

### Custom Labeling Rules

Edit `config/rules.toml` to customize call categorization:

```toml
[rules.call_types]
inquiry = ["information", "question", "how to"]
support = ["not working", "error", "problem", "issue"]
billing = ["invoice", "payment", "charge", "bill"]
complaint = ["unhappy", "disappointed", "poor service"]
```

### Batch Processing

Process multiple files efficiently:

```bash
python scripts/rebuild_index.py --batch-size 50
```

## 🛠️ Development

### Running Tests
```bash
pytest tests/ -v --cov=src
```

### Code Formatting
```bash
black src/ tests/
ruff check src/ tests/
```

### Type Checking
```bash
mypy src/
```

### Building Documentation
```bash
cd docs/
make html
```

## 📈 Performance Tips

1. **GPU Acceleration**: Install CUDA for faster Whisper processing
2. **Caching**: Enable caching in `config/app.toml` for repeated queries
3. **Batch Processing**: Process files in batches for better efficiency
4. **Index Optimization**: Rebuild vector index periodically

## 🔒 Security & Privacy

- **Local Processing**: All data processing happens on your machine
- **No External APIs**: No data is sent to external services
- **Configurable PII Masking**: Optional customer name masking
- **Secure Storage**: Data stored locally with configurable encryption

## 🐛 Troubleshooting

### Common Issues

**FFmpeg not found**
```bash
# Verify installation
ffmpeg -version

# Add to PATH if needed
export PATH=$PATH:/path/to/ffmpeg
```

**Out of memory errors**
- Reduce batch size in configuration
- Process files in smaller groups
- Increase system swap space

**Slow transcription**
- Use smaller Whisper model (tiny or base)
- Enable GPU acceleration
- Reduce audio quality settings

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please read CONTRIBUTING.md for guidelines.

## 📧 Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation at `/docs`
- Review closed issues for solutions

## 🙏 Acknowledgments

- OpenAI Whisper for speech-to-text
- ChromaDB for vector storage
- Streamlit for the UI framework
- The open-source community

---

Built with ❤️ for efficient call analytics