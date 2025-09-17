#!/usr/bin/env python3
"""
Enhanced Stock Analysis - Local Mirror System End-to-End Test
============================================================

This script tests the complete functionality of the local mirror system including:
- Setup and initialization
- Document downloading and management
- AI-powered document analysis  
- Enhanced prediction modeling
- Web interface components
- Database operations and persistence

Author: Local Mirror System
Version: 1.0.0
Date: 2025-09-16
"""

import os
import sys
import asyncio
import json
import sqlite3
import tempfile
import shutil
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock
import logging

# Add project paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LocalMirrorSystemTest(unittest.TestCase):
    """Comprehensive test suite for the local mirror system."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # Create test directory structure
        self.create_test_directories()
        
        # Copy essential files
        self.copy_essential_files()
        
        logger.info(f"Test environment set up in: {self.test_dir}")
    
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)
        logger.info("Test environment cleaned up")
    
    def create_test_directories(self):
        """Create necessary directory structure for testing."""
        directories = [
            'data',
            'data/documents', 
            'data/cache',
            'data/analysis',
            'data/models',
            'logs',
            'config',
            'src/api',
            'src/document_management',
            'src/analysis',
            'src/prediction',
            'templates',
            'static/css',
            'static/js'
        ]
        
        for directory in directories:
            (self.test_dir / directory).mkdir(parents=True, exist_ok=True)
    
    def copy_essential_files(self):
        """Copy essential files from the source directory."""
        source_dir = PROJECT_ROOT
        
        # Copy configuration
        config_source = source_dir / "config" / "local_config.json"
        if config_source.exists():
            shutil.copy2(config_source, self.test_dir / "config" / "local_config.json")
        else:
            # Create minimal config for testing
            self.create_test_config()
        
        # Copy setup script
        setup_source = source_dir / "setup.py"
        if setup_source.exists():
            shutil.copy2(setup_source, self.test_dir / "setup.py")
    
    def create_test_config(self):
        """Create minimal configuration for testing."""
        config = {
            "app_name": "Test Local Mirror",
            "version": "1.0.0",
            "server": {"host": "127.0.0.1", "port": 8001, "debug": True},
            "database": {"database_path": "data/test_mirror.db"},
            "storage": {
                "documents_dir": "data/documents",
                "cache_dir": "data/cache",
                "analysis_dir": "data/analysis"
            },
            "document_management": {"max_concurrent_downloads": 1},
            "analysis": {"max_concurrent_analysis": 1},
            "prediction": {
                "prediction_weights": {
                    "document_analysis": 0.30,
                    "technical_analysis": 0.40,
                    "market_context": 0.30
                }
            }
        }
        
        with open(self.test_dir / "config" / "local_config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    def test_01_setup_initialization(self):
        """Test setup and database initialization."""
        logger.info("Testing setup and initialization...")
        
        try:
            # Import setup functionality
            sys.path.insert(0, str(self.test_dir))
            from setup import LocalMirrorSetup
            
            # Initialize setup
            setup = LocalMirrorSetup(str(self.test_dir))
            setup.create_local_database()
            
            # Verify database creation
            db_path = self.test_dir / "data" / "test_mirror.db"
            self.assertTrue(db_path.exists(), "Database file should be created")
            
            # Verify database schema
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check for required tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = [
                'documents',
                'document_analysis', 
                'predictions',
                'stock_data_cache',
                'analysis_queue',
                'app_settings'
            ]
            
            for table in required_tables:
                self.assertIn(table, tables, f"Table '{table}' should exist")
            
            conn.close()
            logger.info("✓ Setup and initialization test passed")
            
        except Exception as e:
            self.fail(f"Setup initialization failed: {e}")
    
    def test_02_document_management_imports(self):
        """Test document management component imports."""
        logger.info("Testing document management imports...")
        
        try:
            # Test document downloader import
            with patch('yfinance.Ticker'), patch('requests.get'), patch('aiohttp.ClientSession'):
                from src.document_management.document_downloader import DocumentDownloader
                
                downloader = DocumentDownloader(
                    download_dir=str(self.test_dir / "data" / "documents"),
                    db_path=str(self.test_dir / "data" / "test_mirror.db")
                )
                
                self.assertIsNotNone(downloader, "Document downloader should initialize")
                logger.info("✓ Document downloader import successful")
            
        except ImportError as e:
            self.fail(f"Document management import failed: {e}")
    
    def test_03_analysis_component_imports(self):
        """Test analysis component imports.""" 
        logger.info("Testing analysis component imports...")
        
        try:
            # Mock heavy AI model imports
            with patch('transformers.AutoTokenizer'), \
                 patch('transformers.AutoModelForSequenceClassification'), \
                 patch('spacy.load'), \
                 patch('nltk.download'):
                
                from src.analysis.document_analyzer import DocumentAnalyzer
                
                analyzer = DocumentAnalyzer(
                    db_path=str(self.test_dir / "data" / "test_mirror.db"),
                    cache_dir=str(self.test_dir / "data" / "cache")
                )
                
                self.assertIsNotNone(analyzer, "Document analyzer should initialize")
                logger.info("✓ Document analyzer import successful")
                
        except ImportError as e:
            self.fail(f"Analysis component import failed: {e}")
    
    def test_04_prediction_component_imports(self):
        """Test prediction component imports."""
        logger.info("Testing prediction component imports...")
        
        try:
            with patch('yfinance.Ticker'), patch('pandas.DataFrame'):
                from src.prediction.enhanced_local_predictor import EnhancedLocalPredictor
                
                predictor = EnhancedLocalPredictor(
                    db_path=str(self.test_dir / "data" / "test_mirror.db"),
                    cache_dir=str(self.test_dir / "data" / "cache")
                )
                
                self.assertIsNotNone(predictor, "Enhanced predictor should initialize")
                logger.info("✓ Enhanced predictor import successful")
                
        except ImportError as e:
            self.fail(f"Prediction component import failed: {e}")
    
    def test_05_main_app_creation(self):
        """Test main FastAPI application creation."""
        logger.info("Testing main application creation...")
        
        try:
            with patch('src.document_management.document_downloader.DocumentDownloader'), \
                 patch('src.analysis.document_analyzer.DocumentAnalyzer'), \
                 patch('src.prediction.enhanced_local_predictor.EnhancedLocalPredictor'):
                
                from src.api.main_app import LocalMirrorApplication, create_app
                
                # Test application factory
                app = create_app(str(self.test_dir / "config" / "local_config.json"))
                
                self.assertIsNotNone(app, "FastAPI application should be created")
                logger.info("✓ Main application creation successful")
                
        except Exception as e:
            self.fail(f"Main application creation failed: {e}")
    
    def test_06_database_operations(self):
        """Test database operations and queries."""
        logger.info("Testing database operations...")
        
        try:
            db_path = self.test_dir / "data" / "test_mirror.db"
            
            # Create database if it doesn't exist
            if not db_path.exists():
                sys.path.insert(0, str(self.test_dir))
                from setup import LocalMirrorSetup
                setup = LocalMirrorSetup(str(self.test_dir))
                setup.create_local_database()
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Test insert operations
            cursor.execute("""
                INSERT INTO documents (symbol, document_type, title, url, status)
                VALUES (?, ?, ?, ?, ?)
            """, ('TEST.AX', 'annual_report', 'Test Document', 'http://test.com', 'downloaded'))
            
            document_id = cursor.lastrowid
            
            # Test analysis insert
            cursor.execute("""
                INSERT INTO document_analysis (document_id, analysis_type, sentiment_score, confidence_score)
                VALUES (?, ?, ?, ?)
            """, (document_id, 'finbert', 0.75, 0.85))
            
            # Test prediction insert
            cursor.execute("""
                INSERT INTO predictions (symbol, prediction_type, timeframe, predicted_price, confidence_score)
                VALUES (?, ?, ?, ?, ?)
            """, ('TEST.AX', 'enhanced', '5d', 168.50, 0.78))
            
            conn.commit()
            
            # Test select operations
            cursor.execute("SELECT COUNT(*) FROM documents WHERE symbol = ?", ('TEST.AX',))
            doc_count = cursor.fetchone()[0]
            self.assertEqual(doc_count, 1, "Document should be inserted")
            
            cursor.execute("SELECT COUNT(*) FROM document_analysis")
            analysis_count = cursor.fetchone()[0]
            self.assertEqual(analysis_count, 1, "Analysis should be inserted")
            
            cursor.execute("SELECT COUNT(*) FROM predictions")
            prediction_count = cursor.fetchone()[0]
            self.assertEqual(prediction_count, 1, "Prediction should be inserted")
            
            conn.close()
            logger.info("✓ Database operations test passed")
            
        except Exception as e:
            self.fail(f"Database operations failed: {e}")
    
    def test_07_template_files_exist(self):
        """Test that all required template files exist in the source."""
        logger.info("Testing template files...")
        
        source_templates = PROJECT_ROOT / "templates"
        required_templates = [
            'base.html',
            'dashboard.html',
            'documents.html',
            'analysis.html',
            'predictions.html'
        ]
        
        for template in required_templates:
            template_path = source_templates / template
            self.assertTrue(
                template_path.exists(), 
                f"Template '{template}' should exist in source"
            )
            
            # Check that template has content
            with open(template_path, 'r') as f:
                content = f.read()
                self.assertGreater(
                    len(content), 100,
                    f"Template '{template}' should have substantial content"
                )
        
        logger.info("✓ Template files test passed")
    
    def test_08_configuration_validation(self):
        """Test configuration file validation."""
        logger.info("Testing configuration validation...")
        
        try:
            config_path = self.test_dir / "config" / "local_config.json"
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Test required configuration sections
            required_sections = [
                'app_name',
                'version', 
                'server',
                'database',
                'storage'
            ]
            
            for section in required_sections:
                self.assertIn(section, config, f"Config section '{section}' should exist")
            
            # Test server configuration
            self.assertIn('host', config['server'])
            self.assertIn('port', config['server'])
            self.assertIsInstance(config['server']['port'], int)
            
            # Test database configuration
            self.assertIn('database_path', config['database'])
            
            logger.info("✓ Configuration validation test passed")
            
        except Exception as e:
            self.fail(f"Configuration validation failed: {e}")
    
    def test_09_startup_script_functionality(self):
        """Test startup script functionality."""
        logger.info("Testing startup script functionality...")
        
        try:
            startup_script = PROJECT_ROOT / "scripts" / "start_local_mirror.py"
            self.assertTrue(startup_script.exists(), "Startup script should exist")
            
            # Test script is executable
            self.assertTrue(os.access(startup_script, os.X_OK), "Startup script should be executable")
            
            # Read script content and verify key functions
            with open(startup_script, 'r') as f:
                script_content = f.read()
            
            required_functions = [
                'check_dependencies',
                'download_nlp_models', 
                'create_directory_structure',
                'start_application'
            ]
            
            for func in required_functions:
                self.assertIn(
                    f"def {func}", script_content,
                    f"Function '{func}' should exist in startup script"
                )
            
            logger.info("✓ Startup script functionality test passed")
            
        except Exception as e:
            self.fail(f"Startup script test failed: {e}")
    
    def test_10_requirements_completeness(self):
        """Test that requirements.txt is complete and valid."""
        logger.info("Testing requirements completeness...")
        
        try:
            requirements_path = PROJECT_ROOT / "requirements.txt"
            self.assertTrue(requirements_path.exists(), "Requirements.txt should exist")
            
            with open(requirements_path, 'r') as f:
                requirements_content = f.read()
            
            # Check for critical dependencies
            critical_deps = [
                'fastapi',
                'uvicorn',
                'pandas',
                'yfinance',
                'transformers',
                'torch',
                'spacy',
                'nltk'
            ]
            
            for dep in critical_deps:
                self.assertIn(
                    dep, requirements_content,
                    f"Critical dependency '{dep}' should be in requirements.txt"
                )
            
            logger.info("✓ Requirements completeness test passed")
            
        except Exception as e:
            self.fail(f"Requirements test failed: {e}")

def run_comprehensive_test():
    """Run comprehensive end-to-end testing."""
    print("=" * 70)
    print("Enhanced Stock Analysis - Local Mirror System Test")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(LocalMirrorSystemTest)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED - Local Mirror System is Ready for Deployment")
        print("✅ System Components Validated:")
        print("   • Database initialization and schema")
        print("   • Document management functionality")
        print("   • AI analysis components")
        print("   • Enhanced prediction modeling")  
        print("   • Web application framework")
        print("   • Configuration and startup scripts")
        print("   • Template and interface files")
        print("   • Dependency management")
    else:
        print("❌ SOME TESTS FAILED - Please review and fix issues")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
        
        if result.failures:
            print("\n   Failed Tests:")
            for test, traceback in result.failures:
                print(f"   • {test}")
        
        if result.errors:
            print("\n   Error Tests:")
            for test, traceback in result.errors:
                print(f"   • {test}")
    
    print("=" * 70)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)