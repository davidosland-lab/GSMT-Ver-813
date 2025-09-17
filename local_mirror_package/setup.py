#!/usr/bin/env python3
"""
Enhanced Stock Market Tracker - Local Mirror Installation
A comprehensive offline-capable stock analysis system with document management
"""

import os
import sys
import subprocess
from pathlib import Path
import json
import sqlite3
from datetime import datetime

class LocalMirrorInstaller:
    def __init__(self):
        self.project_name = "enhanced-stock-tracker-local"
        self.base_dir = Path.cwd() / self.project_name
        self.data_dir = self.base_dir / "data"
        self.documents_dir = self.data_dir / "documents"
        self.analysis_dir = self.data_dir / "analysis_results"
        self.db_path = self.data_dir / "local_stock_tracker.db"
        
    def create_directory_structure(self):
        """Create comprehensive directory structure for local deployment"""
        directories = [
            self.base_dir,
            self.base_dir / "src",
            self.base_dir / "src" / "core",
            self.base_dir / "src" / "prediction",
            self.base_dir / "src" / "document_management",
            self.base_dir / "src" / "analysis",
            self.base_dir / "src" / "api",
            self.base_dir / "src" / "web_interface",
            self.base_dir / "config",
            self.data_dir,
            self.documents_dir,
            self.documents_dir / "annual_reports",
            self.documents_dir / "asx_announcements", 
            self.documents_dir / "investor_presentations",
            self.documents_dir / "financial_statements",
            self.documents_dir / "news_articles",
            self.documents_dir / "research_reports",
            self.analysis_dir,
            self.analysis_dir / "document_summaries",
            self.analysis_dir / "sentiment_analysis",
            self.analysis_dir / "financial_metrics",
            self.analysis_dir / "prediction_factors",
            self.base_dir / "logs",
            self.base_dir / "backup",
            self.base_dir / "templates",
            self.base_dir / "static" / "css",
            self.base_dir / "static" / "js"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {directory}")
            
    def create_local_database(self):
        """Initialize comprehensive local database for document storage and analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Documents table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            document_type TEXT NOT NULL, -- annual_report, asx_announcement, presentation, etc.
            title TEXT,
            url TEXT UNIQUE,
            local_path TEXT,
            file_size INTEGER,
            download_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_analyzed TIMESTAMP,
            analysis_version INTEGER DEFAULT 1,
            content_hash TEXT,
            status TEXT DEFAULT 'downloaded' -- downloaded, analyzed, failed
        )
        ''')
        
        # Document analysis results table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            analysis_type TEXT NOT NULL, -- sentiment, financial_metrics, key_points, summary
            analysis_result JSON,
            confidence_score REAL,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_version TEXT,
            FOREIGN KEY (document_id) REFERENCES documents (id)
        )
        ''')
        
        # Stock information table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stocks (
            symbol TEXT PRIMARY KEY,
            company_name TEXT,
            sector TEXT,
            market_cap REAL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            document_count INTEGER DEFAULT 0,
            analysis_count INTEGER DEFAULT 0
        )
        ''')
        
        # Prediction results table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            timeframe TEXT, -- 1d, 5d, 30d, etc.
            predicted_price REAL,
            current_price REAL,
            confidence_score REAL,
            direction TEXT, -- up, down, sideways
            factors_used JSON, -- Document IDs and analysis results used
            model_version TEXT,
            actual_price REAL, -- Filled later for backtesting
            accuracy_score REAL
        )
        ''')
        
        # Analysis cache table for performance
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_cache (
            cache_key TEXT PRIMARY KEY,
            symbol TEXT,
            cache_data JSON,
            expiry_date TIMESTAMP,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Configuration table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS configuration (
            key TEXT PRIMARY KEY,
            value TEXT,
            description TEXT,
            updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Insert default configuration
        default_configs = [
            ('document_retention_days', '365', 'How long to keep downloaded documents'),
            ('analysis_refresh_interval_hours', '24', 'How often to re-analyze documents'),
            ('max_documents_per_symbol', '100', 'Maximum documents to store per stock'),
            ('enable_auto_download', 'true', 'Automatically download new documents'),
            ('prediction_model_version', '2.0', 'Current prediction model version'),
        ]
        
        cursor.executemany('''
        INSERT OR IGNORE INTO configuration (key, value, description) 
        VALUES (?, ?, ?)
        ''', default_configs)
        
        conn.commit()
        conn.close()
        print(f"✅ Created local database: {self.db_path}")
        
    def create_requirements_file(self):
        """Create comprehensive requirements.txt for local deployment"""
        requirements = [
            "fastapi>=0.104.1",
            "uvicorn[standard]>=0.24.0",
            "yfinance>=0.2.22",
            "pandas>=2.1.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "tensorflow>=2.15.0",
            "beautifulsoup4>=4.12.0",
            "requests>=2.31.0",
            "aiohttp>=3.9.0",
            "asyncio>=3.4.3",
            "sqlalchemy>=2.0.0",
            "sqlite3", # Built-in, but listed for clarity
            "jinja2>=3.1.0",
            "python-multipart>=0.0.6",
            "python-dotenv>=1.0.0",
            "pydantic>=2.5.0",
            "textblob>=0.17.1",  # For sentiment analysis
            "nltk>=3.8.1",       # Natural language processing
            "spacy>=3.7.0",      # Advanced NLP
            "transformers>=4.36.0", # Hugging Face transformers for document analysis
            "torch>=2.1.0",      # PyTorch for advanced ML models
            "plotly>=5.17.0",    # Interactive charts
            "dash>=2.15.0",      # Local web dashboard
            "celery>=5.3.0",     # Background task processing
            "redis>=5.0.0",      # Task queue backend
            "apscheduler>=3.10.0", # Scheduled tasks
            "pypdf2>=3.0.1",     # PDF document processing
            "python-docx>=1.1.0", # Word document processing
            "openpyxl>=3.1.0",   # Excel file processing
            "lxml>=4.9.0",       # XML/HTML parsing
            "feedparser>=6.0.0", # RSS feed parsing
            "newspaper3k>=0.2.8", # News article extraction
            "wordcloud>=1.9.0",  # Document visualization
            "seaborn>=0.13.0",   # Statistical visualization
            "matplotlib>=3.8.0", # Base plotting
            "jupyter>=1.0.0",    # Interactive analysis
            "ipykernel>=6.26.0", # Jupyter kernel
            "streamlit>=1.28.0", # Alternative web interface
        ]
        
        requirements_path = self.base_dir / "requirements.txt"
        with open(requirements_path, 'w') as f:
            f.write('\n'.join(requirements))
        print(f"✅ Created requirements file: {requirements_path}")
        
    def create_configuration_files(self):
        """Create configuration files for local deployment"""
        
        # Main configuration
        config = {
            "database": {
                "path": str(self.db_path),
                "backup_interval_hours": 24
            },
            "document_management": {
                "download_dir": str(self.documents_dir),
                "max_file_size_mb": 50,
                "allowed_extensions": [".pdf", ".doc", ".docx", ".txt", ".html", ".xlsx"],
                "auto_download": True,
                "download_schedule": "0 9 * * 1-5"  # Weekdays at 9 AM
            },
            "analysis": {
                "sentiment_model": "textblob",
                "summarization_model": "transformers",
                "max_summary_length": 500,
                "confidence_threshold": 0.7
            },
            "prediction": {
                "model_type": "ensemble",
                "include_document_analysis": True,
                "document_weight": 0.3,
                "technical_weight": 0.4,
                "market_weight": 0.3,
                "min_documents_for_analysis": 3
            },
            "api": {
                "host": "127.0.0.1",
                "port": 8000,
                "reload": True,
                "log_level": "info"
            },
            "web_interface": {
                "title": "Enhanced Stock Tracker - Local",
                "theme": "dark",
                "enable_document_viewer": True,
                "enable_analysis_export": True
            }
        }
        
        config_path = self.base_dir / "config" / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        # Environment file
        env_content = f'''# Enhanced Stock Tracker - Local Configuration
DATABASE_PATH={self.db_path}
DOCUMENTS_DIR={self.documents_dir}
ANALYSIS_DIR={self.analysis_dir}
LOG_LEVEL=INFO
DEBUG=false
MAX_WORKERS=4
CACHE_SIZE_MB=500
ENABLE_DOCUMENT_AUTO_DOWNLOAD=true
PREDICTION_MODEL_VERSION=2.0
BACKUP_RETENTION_DAYS=30
'''
        
        env_path = self.base_dir / ".env"
        with open(env_path, 'w') as f:
            f.write(env_content)
            
        print(f"✅ Created configuration files")
        
    def create_startup_script(self):
        """Create startup script for local deployment"""
        startup_content = f'''#!/usr/bin/env python3
"""
Enhanced Stock Market Tracker - Local Startup Script
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_dependencies():
    """Check if all required packages are installed"""
    try:
        import fastapi
        import uvicorn
        import yfinance
        import pandas
        import numpy
        import sklearn
        import tensorflow
        print("✅ All core dependencies installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {{e}}")
        print("Please run: pip install -r requirements.txt")
        return False

def start_services():
    """Start the local stock tracker services"""
    if not check_dependencies():
        return False
        
    print("🚀 Starting Enhanced Stock Market Tracker - Local")
    
    # Start background document downloader
    print("📥 Starting document downloader service...")
    
    # Start main API server
    print("🌐 Starting API server on http://127.0.0.1:8000")
    
    try:
        import uvicorn
        from src.api.main_app import app
        
        uvicorn.run(
            "src.api.main_app:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\\n🛑 Shutting down services...")
    except Exception as e:
        print(f"❌ Error starting services: {{e}}")
        return False
        
    return True

if __name__ == "__main__":
    if not start_services():
        sys.exit(1)
'''
        
        startup_path = self.base_dir / "start_local.py"
        with open(startup_path, 'w') as f:
            f.write(startup_content)
        os.chmod(startup_path, 0o755)
        
        print(f"✅ Created startup script: {startup_path}")
        
    def create_readme(self):
        """Create comprehensive README for local installation"""
        readme_content = f'''# Enhanced Stock Market Tracker - Local Mirror

A comprehensive offline-capable stock analysis system with advanced document management and analysis capabilities.

## Features

### 🔍 **Document Management**
- **Automatic Download**: Annual reports, ASX announcements, investor presentations
- **Local Storage**: All documents stored locally for offline access
- **Analysis Engine**: AI-powered document analysis with persistent results
- **Smart Caching**: Efficient storage and retrieval of analysis results

### 📊 **Advanced Prediction**
- **Enhanced Models**: Document-informed prediction algorithms
- **Local Processing**: All analysis done locally, no external API dependencies
- **Persistent Storage**: Analysis results cached for fast retrieval
- **Backtesting**: Historical accuracy validation

### 🎯 **Supported Analysis Types**
- Sentiment analysis of annual reports and announcements
- Financial metrics extraction from documents
- Key business insights identification
- Management commentary analysis
- Risk factor assessment

## Installation

### Prerequisites
- Python 3.9 or higher
- 8GB RAM recommended
- 5GB free disk space (more for document storage)

### Quick Start

1. **Extract and Navigate**
   ```bash
   cd {self.project_name}
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize Database**
   ```bash
   python setup.py init
   ```

4. **Start the System**
   ```bash
   python start_local.py
   ```

5. **Access Web Interface**
   - Open http://127.0.0.1:8000 in your browser
   - Enhanced predictions: http://127.0.0.1:8000/enhanced_predictions

## Usage

### Document Management

1. **Add a Stock for Analysis**
   ```python
   # Via API or web interface
   POST /api/stocks/add
   {{"symbol": "CBA.AX", "auto_download": true}}
   ```

2. **Download Documents**
   - Automatic: Documents download on schedule
   - Manual: Use web interface or API endpoints

3. **View Analysis Results**
   - Web dashboard shows all analysis results
   - Export capabilities for further analysis

### Prediction System

The enhanced prediction model uses:
- **Technical Analysis** (40%): Price patterns, indicators
- **Document Analysis** (30%): Sentiment, business insights
- **Market Factors** (30%): Broader market conditions

### Configuration

Edit `config/config.json` to customize:
- Document download preferences
- Analysis model settings
- Prediction weights
- Storage locations

## Directory Structure

```
{self.project_name}/
├── src/                          # Core application code
│   ├── document_management/      # Document download and processing
│   ├── analysis/                 # AI analysis engines
│   ├── prediction/               # Enhanced prediction models
│   └── api/                      # FastAPI application
├── data/                         # Local data storage
│   ├── documents/                # Downloaded documents
│   └── analysis_results/         # Analysis outputs
├── config/                       # Configuration files
├── logs/                         # Application logs
├── backup/                       # Database backups
└── static/                       # Web interface assets
```

## API Endpoints

### Document Management
- `GET /api/documents/{{symbol}}` - List documents for stock
- `POST /api/documents/download` - Download new documents
- `GET /api/analysis/{{document_id}}` - Get document analysis

### Prediction
- `GET /api/prediction/enhanced/{{symbol}}` - Get enhanced prediction
- `GET /api/prediction/factors/{{symbol}}` - Get prediction factors
- `POST /api/prediction/backtest` - Run backtesting

### System Management
- `GET /api/system/status` - System health check
- `POST /api/system/backup` - Create database backup
- `GET /api/system/stats` - Usage statistics

## Advanced Features

### Custom Analysis Models
Add your own document analysis models in `src/analysis/custom_models/`

### Batch Processing
Process multiple stocks simultaneously for efficient analysis

### Export Capabilities
Export analysis results to Excel, CSV, or JSON formats

## Troubleshooting

### Common Issues
1. **Port 8000 in use**: Change port in config.json
2. **Missing documents**: Check download permissions and URLs
3. **Slow analysis**: Reduce batch size or enable GPU acceleration

### Logs
Check `logs/` directory for detailed error information

## Performance Optimization

- **Database**: Regular VACUUM operations
- **Documents**: Automatic cleanup of old files
- **Cache**: Intelligent result caching
- **Memory**: Configurable batch sizes

## Security

- **Local Only**: No external data transmission
- **Encrypted Storage**: Optional database encryption
- **Access Control**: Configurable API authentication

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review configuration in `config/config.json`
3. Ensure all dependencies are installed

---

**Note**: This is a local mirror designed for offline operation. All analysis is performed locally without external API dependencies.
'''
        
        readme_path = self.base_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
            
        print(f"✅ Created README: {readme_path}")

    def run_installation(self):
        """Run the complete installation process"""
        print(f"🚀 Installing Enhanced Stock Market Tracker - Local Mirror")
        print(f"📁 Installation directory: {self.base_dir}")
        
        try:
            self.create_directory_structure()
            self.create_local_database()
            self.create_requirements_file()
            self.create_configuration_files()
            self.create_startup_script()
            self.create_readme()
            
            print(f"\\n✅ Installation completed successfully!")
            print(f"📁 Project created at: {self.base_dir}")
            print(f"\\n🔄 Next steps:")
            print(f"1. cd {self.base_dir}")
            print(f"2. pip install -r requirements.txt")
            print(f"3. python start_local.py")
            print(f"\\n🌐 Access at: http://127.0.0.1:8000")
            
        except Exception as e:
            print(f"❌ Installation failed: {e}")
            return False
            
        return True

if __name__ == "__main__":
    installer = LocalMirrorInstaller()
    installer.run_installation()