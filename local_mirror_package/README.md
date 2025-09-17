# Enhanced Stock Analysis - Local Mirror System

A comprehensive local stock analysis platform with document management, AI-powered analysis, and enhanced prediction modeling. This system operates entirely offline after initial setup, providing continuous local deployment capabilities with persistent storage of all analysis results.

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Key Features

### 📄 **Comprehensive Document Management**
- **Multi-source document discovery** from ASX announcements, company websites, investor presentations
- **Async downloading with rate limiting** to respect server resources
- **Duplicate detection via content hashing** to avoid redundant storage
- **Local document storage** with persistent metadata tracking
- **Support for multiple formats**: PDF, DOCX, HTML, TXT

### 🧠 **AI-Powered Document Analysis**
- **FinBERT financial sentiment analysis** for specialized financial text understanding
- **spaCy NLP processing** for advanced text analysis and entity recognition  
- **Transformers-based summarization** for key insights extraction
- **Risk factor analysis** with automated identification of business risks
- **Business insights extraction** for strategic investment considerations
- **Persistent analysis storage** with versioning and confidence scoring

### 📈 **Enhanced Prediction Modeling**
- **Ensemble approach** combining document analysis (30%), technical analysis (40%), and market context (30%)
- **Uses stored document analysis results** for comprehensive market understanding
- **Configurable prediction weights** based on user preferences or market conditions
- **Multiple timeframe support**: 5min, 30min, 1hr, 1day, 5day, 1month, 3month
- **Confidence intervals** with statistical validation
- **Prediction accuracy tracking** for model performance monitoring

### 🎯 **Offline-First Architecture**
- **Complete local operation** after initial setup and model downloads
- **SQLite database** for persistent storage without external dependencies
- **Local document cache** for fast repeated access
- **Background task processing** for non-blocking operations
- **Continuous operation capability** suitable for unattended local deployment

## 🚀 Quick Start

### Prerequisites

- **Python 3.8 or higher**
- **4GB+ RAM recommended** (for AI models)
- **2GB+ disk space** (for documents and models)
- **Internet connection** (initial setup only)

### Automated Installation

1. **Download the local mirror package** to your desired location
2. **Navigate to the package directory**:
   ```bash
   cd local_mirror_package
   ```
3. **Run the installation script**:
   ```bash
   chmod +x scripts/install_local_mirror.sh
   ./scripts/install_local_mirror.sh
   ```

The installer will:
- ✅ Check system dependencies
- ✅ Create Python virtual environment  
- ✅ Install all required packages
- ✅ Download NLP models (NLTK, spaCy)
- ✅ Initialize local SQLite database
- ✅ Create startup/stop scripts
- ✅ Run installation validation tests

### Manual Installation

If you prefer manual setup:

1. **Create virtual environment**:
   ```bash
   python3 -m venv stock_analysis_env
   source stock_analysis_env/bin/activate  # Linux/macOS
   # OR
   stock_analysis_env\Scripts\activate     # Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLP models**:
   ```bash
   python -m nltk.downloader punkt stopwords vader_lexicon wordnet
   python -m spacy download en_core_web_sm
   ```

4. **Initialize system**:
   ```bash
   python setup.py
   ```

## 📋 System Usage

### Starting the System

After installation, start the local mirror system:

```bash
cd enhanced_stock_analysis_local  # Your installation directory
./start.sh
```

Or with custom options:
```bash
./start.sh --host 0.0.0.0 --port 8080 --debug
```

### Accessing the Interface

Once started, access the system at:
- **Main Dashboard**: http://127.0.0.1:8000
- **API Documentation**: http://127.0.0.1:8000/api/docs
- **Interactive API**: http://127.0.0.1:8000/api/redoc

### Basic Workflow

1. **Navigate to the dashboard** to see system overview
2. **Select a stock symbol** (e.g., CBA.AX for Commonwealth Bank)
3. **Download documents** using the Documents page
4. **Run analysis** on downloaded documents  
5. **Generate predictions** using enhanced modeling
6. **Review results** and export data as needed

### Example: Analyzing CBA.AX

```python
# Via web interface:
# 1. Go to Documents → Enter "CBA.AX" 
# 2. Click "Download New Documents"
# 3. Go to Analysis → Click "New Analysis"  
# 4. Go to Predictions → Click "New Prediction"

# Via API:
import requests

base_url = "http://127.0.0.1:8000/api"

# Download documents
response = requests.post(f"{base_url}/documents/CBA.AX/download")
task_id = response.json()['task_id']

# Run analysis  
response = requests.post(f"{base_url}/analysis/CBA.AX/analyze")

# Generate prediction
response = requests.post(f"{base_url}/predictions/CBA.AX/predict", 
                        json={"timeframe": "5d", "include_analysis": True})
```

## 🏗️ Architecture Overview

```
Enhanced Stock Analysis - Local Mirror
├── 📁 Document Management Layer
│   ├── Multi-source document discovery (ASX, company sites)
│   ├── Async download with rate limiting
│   └── Content deduplication and local storage
│
├── 🧠 AI Analysis Layer  
│   ├── FinBERT sentiment analysis
│   ├── spaCy NLP processing
│   ├── Transformers summarization
│   └── Risk and insights extraction
│
├── 📈 Enhanced Prediction Layer
│   ├── Document sentiment integration (30%)
│   ├── Technical analysis features (40%)  
│   ├── Market context analysis (30%)
│   └── Ensemble modeling with confidence intervals
│
├── 💾 Persistent Storage Layer
│   ├── SQLite database for all data
│   ├── Local document file storage
│   └── Analysis results caching
│
└── 🌐 Web Interface Layer
    ├── FastAPI backend with async support
    ├── Jinja2 templated frontend
    └── RESTful API with OpenAPI docs
```

## 🔧 Configuration

### Main Configuration File

Edit `config/local_config.json` to customize system behavior:

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8000,
    "debug": false
  },
  "document_management": {
    "max_concurrent_downloads": 3,
    "download_timeout_seconds": 300,
    "document_retention_days": 365
  },
  "analysis": {
    "max_concurrent_analysis": 2,
    "finbert_model": "ProsusAI/finbert",
    "enable_gpu_acceleration": false
  },
  "prediction": {
    "prediction_weights": {
      "document_analysis": 0.30,
      "technical_analysis": 0.40,
      "market_context": 0.30
    }
  }
}
```

### Environment Variables

Set these environment variables for additional configuration:

```bash
export STOCK_ANALYSIS_DEBUG=true          # Enable debug logging
export STOCK_ANALYSIS_GPU=false           # Use GPU acceleration
export STOCK_ANALYSIS_WORKERS=1           # Number of worker processes
export STOCK_ANALYSIS_LOG_LEVEL=INFO      # Logging level
```

## 📊 API Reference

### Document Management Endpoints

```http
POST   /api/documents/{symbol}/download     # Download documents for symbol
GET    /api/documents/{symbol}             # Get documents for symbol  
DELETE /api/documents/{document_id}        # Delete specific document
```

### Analysis Endpoints

```http
POST   /api/analysis/{symbol}/analyze      # Analyze documents for symbol
GET    /api/analysis/{symbol}              # Get analysis results
GET    /api/analysis/document/{doc_id}     # Get specific document analysis
```

### Prediction Endpoints

```http
POST   /api/predictions/{symbol}/predict   # Generate new prediction
GET    /api/predictions/{symbol}           # Get prediction history
GET    /api/predictions/{symbol}/latest    # Get latest prediction
```

### System Endpoints

```http
GET    /api/health                         # System health check
GET    /api/system/stats                   # System statistics
GET    /api/tasks                          # Task queue status
GET    /api/config                         # Configuration details
```

## 🛠️ Troubleshooting

### Common Issues

**Installation Fails with "Missing Dependencies"**
```bash
# Ubuntu/Debian:
sudo apt-get update && sudo apt-get install python3-dev python3-venv gcc curl

# macOS:
brew install python gcc

# CentOS/RHEL:
sudo yum install python3-devel gcc curl
```

**NLP Models Download Fails**
```bash
# Manual NLTK download:
python -c "import nltk; nltk.download('all')"

# Manual spaCy download:
python -m spacy download en_core_web_sm
```

**Database Issues**
```bash
# Reset database:
rm data/local_mirror.db
python setup.py --init-db-only
```

**Memory Issues with AI Models**
```json
// In config/local_config.json, reduce concurrent processing:
{
  "analysis": {
    "max_concurrent_analysis": 1,
    "enable_gpu_acceleration": false
  }
}
```

### Performance Optimization

**For better performance on low-resource systems:**

1. **Reduce concurrent operations** in config file
2. **Disable GPU acceleration** if causing issues
3. **Increase timeouts** for slow document downloads
4. **Use smaller batch sizes** for analysis processing

**For high-performance systems:**

1. **Enable GPU acceleration** if CUDA available
2. **Increase concurrent operations**  
3. **Use multiple worker processes**
4. **Enable analysis caching**

### Logs and Debugging

**View application logs:**
```bash
tail -f logs/startup.log           # Startup logs
tail -f logs/local_mirror_app.log  # Application logs  
tail -f logs/document_downloader.log # Download logs
tail -f logs/analysis.log          # Analysis logs
```

**Enable debug mode:**
```bash
./start.sh --debug                 # Debug mode with verbose logging
```

## 🔄 System Management

### Starting/Stopping the System

```bash
# Start system
./start.sh

# Start with custom options
./start.sh --host 0.0.0.0 --port 8080 --workers 2

# Check status
./status.sh

# Stop system
./stop.sh
```

### Database Management

```bash
# Backup database
cp data/local_mirror.db backups/backup_$(date +%Y%m%d).db

# View database statistics
sqlite3 data/local_mirror.db "SELECT 
  (SELECT COUNT(*) FROM documents) as documents,
  (SELECT COUNT(*) FROM document_analysis) as analyses,
  (SELECT COUNT(*) FROM predictions) as predictions;"
```

### Updates and Maintenance

```bash
# Update Python dependencies
source stock_analysis_env/bin/activate
pip install --upgrade -r requirements.txt

# Clean old cache files
find data/cache -type f -mtime +30 -delete

# Clean old log files  
find logs -name "*.log" -mtime +30 -delete
```

## 📈 Performance Metrics

**Typical performance on modern hardware:**

| Operation | Time | Notes |
|-----------|------|-------|
| Document Download | 2-10s per document | Depends on file size |
| FinBERT Analysis | 5-30s per document | CPU-dependent |
| Enhanced Prediction | 10-60s | Includes all analysis |
| Database Query | <1s | SQLite is very fast |
| Web Interface Load | <2s | Local network only |

**Resource Requirements:**

| Component | CPU | RAM | Storage |
|-----------|-----|-----|---------|
| Base System | 1 core | 1GB | 500MB |
| With AI Models | 2+ cores | 4GB+ | 2GB+ |
| Heavy Analysis | 4+ cores | 8GB+ | 5GB+ |

## 🤝 Contributing

This is a local mirror system designed for individual use. However, you can extend functionality by:

1. **Adding new document sources** in `src/document_management/`
2. **Implementing additional analysis models** in `src/analysis/`  
3. **Creating new prediction algorithms** in `src/prediction/`
4. **Enhancing the web interface** in `templates/`

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **FinBERT** - Financial sentiment analysis model
- **spaCy** - Industrial-strength NLP library
- **Transformers** - State-of-the-art NLP models
- **FastAPI** - Modern web framework for Python
- **yfinance** - Yahoo Finance data access
- **SQLite** - Reliable embedded database

---

## 📞 Support

If you encounter issues:

1. **Check the troubleshooting section** above
2. **Review log files** in the `logs/` directory
3. **Verify configuration** in `config/local_config.json`
4. **Test individual components** using the API documentation

---

**Enhanced Stock Analysis - Local Mirror v1.0.0**  
*Comprehensive offline stock analysis with AI-powered insights*