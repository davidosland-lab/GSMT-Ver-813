#!/bin/bash

# Enhanced Stock Analysis - Local Mirror Installation Script
# =========================================================
# 
# This script installs and configures the local mirror system for offline stock analysis
# with comprehensive document management and AI-powered insights.
#
# Usage: chmod +x install_local_mirror.sh && ./install_local_mirror.sh
#
# Author: Local Mirror System  
# Version: 1.0.0
# Date: 2025-09-16

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="$HOME/enhanced_stock_analysis_local"
PYTHON_MIN_VERSION="3.8"
VENV_NAME="stock_analysis_env"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
check_python_version() {
    if command_exists python3; then
        local python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        local required_version="3.8"
        
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            log "✓ Python $python_version detected (>= $required_version required)"
            return 0
        else
            error "Python $python_version detected, but >= $required_version is required"
            return 1
        fi
    else
        error "Python 3 is not installed. Please install Python 3.8 or higher."
        return 1
    fi
}

# Check system dependencies
check_system_dependencies() {
    log "Checking system dependencies..."
    
    local missing_deps=()
    
    # Check for essential tools
    if ! command_exists curl; then
        missing_deps+=("curl")
    fi
    
    if ! command_exists git; then
        missing_deps+=("git")
    fi
    
    if ! command_exists gcc; then
        missing_deps+=("gcc")
    fi
    
    # Platform-specific package manager checks
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command_exists apt-get; then
            # Debian/Ubuntu
            if ! dpkg -l python3-dev >/dev/null 2>&1; then
                missing_deps+=("python3-dev")
            fi
            if ! dpkg -l python3-venv >/dev/null 2>&1; then
                missing_deps+=("python3-venv")
            fi
        elif command_exists yum; then
            # CentOS/RHEL
            if ! rpm -q python3-devel >/dev/null 2>&1; then
                missing_deps+=("python3-devel")
            fi
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if ! command_exists brew; then
            warning "Homebrew is not installed. Some dependencies might need manual installation."
        fi
    fi
    
    if [ ${#missing_deps[@]} -eq 0 ]; then
        log "✓ All system dependencies are available"
        return 0
    else
        error "Missing system dependencies: ${missing_deps[*]}"
        info "Please install the missing dependencies and run this script again."
        
        if [[ "$OSTYPE" == "linux-gnu"* ]] && command_exists apt-get; then
            info "On Ubuntu/Debian, run: sudo apt-get update && sudo apt-get install ${missing_deps[*]}"
        elif [[ "$OSTYPE" == "linux-gnu"* ]] && command_exists yum; then
            info "On CentOS/RHEL, run: sudo yum install ${missing_deps[*]}"
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            info "On macOS, run: brew install ${missing_deps[*]}"
        fi
        
        return 1
    fi
}

# Create installation directory
create_install_directory() {
    log "Creating installation directory at $INSTALL_DIR..."
    
    if [ -d "$INSTALL_DIR" ]; then
        warning "Installation directory already exists. Backing up..."
        mv "$INSTALL_DIR" "${INSTALL_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    fi
    
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    
    log "✓ Installation directory created"
}

# Copy local mirror files
copy_local_mirror_files() {
    log "Copying local mirror system files..."
    
    # Get the directory where this script is located
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local package_dir="$(dirname "$script_dir")"
    
    log "Script directory: $script_dir"
    log "Package directory: $package_dir"
    
    # Verify source directory exists and has content
    if [ ! -d "$package_dir" ]; then
        error "Package directory not found: $package_dir"
        return 1
    fi
    
    # Copy all files from the local mirror package
    if cp -r "$package_dir"/* "$INSTALL_DIR/" 2>/dev/null; then
        log "✓ Files copied successfully"
    else
        warning "Standard copy failed, trying alternative method..."
        # Alternative copying method
        cd "$package_dir"
        find . -type f -exec cp --parents {} "$INSTALL_DIR/" \;
        find . -type d -exec mkdir -p "$INSTALL_DIR/{}" \;
    fi
    
    # Make scripts executable (with error checking)
    if [ -d "$INSTALL_DIR/scripts" ]; then
        chmod +x "$INSTALL_DIR/scripts/"*.py 2>/dev/null || true
        chmod +x "$INSTALL_DIR/scripts/"*.sh 2>/dev/null || true
    fi
    
    log "✓ Local mirror files copied"
}

# Create Python virtual environment
create_virtual_environment() {
    log "Creating Python virtual environment..."
    
    python3 -m venv "$VENV_NAME"
    source "$VENV_NAME/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    log "✓ Virtual environment created and activated"
}

# Install Python dependencies
install_python_dependencies() {
    log "Installing Python dependencies..."
    
    # Ensure we're in the virtual environment
    source "$VENV_NAME/bin/activate"
    
    # Install requirements
    pip install -r requirements.txt
    
    log "✓ Python dependencies installed"
}

# Download NLP models
download_nlp_models() {
    log "Downloading NLP models..."
    
    source "$VENV_NAME/bin/activate"
    
    # Download NLTK data
    python3 -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('wordnet', quiet=True)
print('NLTK data downloaded successfully')
"
    
    # Download spaCy model
    python3 -m spacy download en_core_web_sm
    
    log "✓ NLP models downloaded"
}

# Initialize database
initialize_database() {
    log "Initializing local database..."
    
    source "$VENV_NAME/bin/activate"
    
    # Run the setup script to initialize the database
    python3 setup.py --init-db-only
    
    log "✓ Database initialized"
}

# Create startup scripts
create_startup_scripts() {
    log "Creating startup scripts..."
    
    # Create convenience startup script
    cat > "start.sh" << 'EOF'
#!/bin/bash

# Enhanced Stock Analysis - Local Mirror Startup
cd "$(dirname "$0")"
source stock_analysis_env/bin/activate
python3 scripts/start_local_mirror.py "$@"
EOF
    
    # Create stop script
    cat > "stop.sh" << 'EOF'
#!/bin/bash

# Enhanced Stock Analysis - Local Mirror Shutdown
echo "Stopping Enhanced Stock Analysis - Local Mirror..."

# Find and kill the process
pkill -f "start_local_mirror.py" || echo "No running processes found"
echo "Local mirror stopped"
EOF
    
    # Create status script
    cat > "status.sh" << 'EOF'
#!/bin/bash

# Enhanced Stock Analysis - Local Mirror Status
echo "Enhanced Stock Analysis - Local Mirror Status:"
echo "=============================================="

if pgrep -f "start_local_mirror.py" > /dev/null; then
    echo "Status: RUNNING"
    echo "PID: $(pgrep -f start_local_mirror.py)"
    echo "URL: http://127.0.0.1:8000"
else
    echo "Status: STOPPED"
fi
EOF
    
    chmod +x start.sh stop.sh status.sh
    
    log "✓ Startup scripts created"
}

# Create desktop shortcut (Linux/macOS)
create_desktop_shortcut() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        log "Creating desktop shortcut..."
        
        local desktop_file="$HOME/Desktop/Enhanced_Stock_Analysis.desktop"
        
        cat > "$desktop_file" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Enhanced Stock Analysis - Local Mirror
Comment=Comprehensive local stock analysis with AI-powered insights
Exec=$INSTALL_DIR/start.sh
Icon=$INSTALL_DIR/static/images/icon.png
Terminal=true
Categories=Office;Finance;
EOF
        
        chmod +x "$desktop_file"
        log "✓ Desktop shortcut created"
        
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        log "Creating application bundle for macOS..."
        
        local app_dir="$HOME/Applications/Enhanced Stock Analysis.app"
        mkdir -p "$app_dir/Contents/MacOS"
        mkdir -p "$app_dir/Contents/Resources"
        
        cat > "$app_dir/Contents/MacOS/Enhanced Stock Analysis" << EOF
#!/bin/bash
cd "$INSTALL_DIR"
./start.sh
EOF
        
        chmod +x "$app_dir/Contents/MacOS/Enhanced Stock Analysis"
        
        cat > "$app_dir/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>Enhanced Stock Analysis</string>
    <key>CFBundleIdentifier</key>
    <string>com.localproj.stockanalysis</string>
    <key>CFBundleName</key>
    <string>Enhanced Stock Analysis</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
</dict>
</plist>
EOF
        
        log "✓ macOS application bundle created"
    fi
}

# Run installation tests
run_installation_tests() {
    log "Running installation tests..."
    
    source "$VENV_NAME/bin/activate"
    
    # Test imports
    python3 -c "
import sys
sys.path.insert(0, 'src')

try:
    from api.main_app import create_app
    print('✓ Main application imports successfully')
except ImportError as e:
    print(f'✗ Main application import failed: {e}')
    sys.exit(1)

try:
    from document_management.document_downloader import DocumentDownloader
    print('✓ Document downloader imports successfully')
except ImportError as e:
    print(f'✗ Document downloader import failed: {e}')
    sys.exit(1)

try:
    from analysis.document_analyzer import DocumentAnalyzer  
    print('✓ Document analyzer imports successfully')
except ImportError as e:
    print(f'✗ Document analyzer import failed: {e}')
    sys.exit(1)

try:
    from prediction.enhanced_local_predictor import EnhancedLocalPredictor
    print('✓ Enhanced predictor imports successfully')
except ImportError as e:
    print(f'✗ Enhanced predictor import failed: {e}')
    sys.exit(1)

print('All core components imported successfully')
"
    
    # Test database initialization
    if [ -f "data/local_mirror.db" ]; then
        log "✓ Database file exists"
    else
        error "Database file not found"
        return 1
    fi
    
    log "✓ Installation tests passed"
}

# Main installation function
main() {
    echo ""
    echo "=================================================================="
    echo "Enhanced Stock Analysis - Local Mirror Installation"  
    echo "=================================================================="
    echo ""
    
    log "Starting installation process..."
    
    # Pre-installation checks
    if ! check_python_version; then
        exit 1
    fi
    
    if ! check_system_dependencies; then
        exit 1
    fi
    
    # Installation steps
    create_install_directory
    copy_local_mirror_files  
    create_virtual_environment
    install_python_dependencies
    download_nlp_models
    initialize_database
    create_startup_scripts
    create_desktop_shortcut
    
    # Post-installation validation
    if run_installation_tests; then
        echo ""
        echo "=================================================================="
        log "Installation completed successfully!"
        echo "=================================================================="
        echo ""
        
        info "Installation directory: $INSTALL_DIR"
        info "To start the system: cd $INSTALL_DIR && ./start.sh"
        info "To stop the system: cd $INSTALL_DIR && ./stop.sh"
        info "To check status: cd $INSTALL_DIR && ./status.sh"
        echo ""
        info "Once started, access the system at: http://127.0.0.1:8000"
        info "API documentation will be available at: http://127.0.0.1:8000/api/docs"
        echo ""
        
        # Offer to start the system
        read -p "Would you like to start the system now? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log "Starting Enhanced Stock Analysis - Local Mirror..."
            cd "$INSTALL_DIR"
            ./start.sh
        else
            info "You can start the system later by running: cd $INSTALL_DIR && ./start.sh"
        fi
        
    else
        error "Installation tests failed. Please check the logs and try again."
        exit 1
    fi
}

# Run main installation if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi