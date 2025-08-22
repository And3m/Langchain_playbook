# Installation Guide 🚀

Complete setup instructions for the LangChain Playbook. Follow these steps to get started quickly and efficiently.

## 📋 Quick Start (5 Minutes)

### 1. Prerequisites Check
```bash
# Check Python version (3.8+ required)
python --version

# Check Git installation
git --version

# Check pip is available
pip --version
```

### 2. Clone Repository
```bash
git clone https://github.com/your-username/Langchain-Playbook.git
cd Langchain-Playbook
```

### 3. Setup Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure API Keys
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys
# OPENAI_API_KEY=your_openai_key_here
```

### 5. Verify Installation
```bash
# Run health check
python tests/test_suite.py --mode health

# Test first example
python basics/01_getting_started/hello_langchain.py
```

🎉 **You're ready to go!** Start with the [basics](../basics/) or jump into [interactive notebooks](../notebooks/).

---

## 🖥️ Platform-Specific Setup

### Windows Setup

#### Using Command Prompt
```cmd
# Clone repository
git clone https://github.com/your-username/Langchain-Playbook.git
cd Langchain-Playbook

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt

# Set API key (session only)
set OPENAI_API_KEY=your_key_here
```

#### Using PowerShell
```powershell
# Clone repository
git clone https://github.com/your-username/Langchain-Playbook.git
Set-Location Langchain-Playbook

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# If execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install dependencies
pip install -r requirements.txt

# Set API key (session only)
$env:OPENAI_API_KEY="your_key_here"
```

#### Windows Troubleshooting
```cmd
# If pip install fails, try:
python -m pip install --upgrade pip
python -m pip install --user -r requirements.txt

# If virtual environment activation fails:
python -m venv --clear .venv
```

### macOS Setup

#### Using Terminal
```bash
# Install Xcode command line tools (if needed)
xcode-select --install

# Clone repository
git clone https://github.com/your-username/Langchain-Playbook.git
cd Langchain-Playbook

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set API key permanently
echo 'export OPENAI_API_KEY="your_key_here"' >> ~/.zshrc
source ~/.zshrc
```

#### Using Homebrew Python
```bash
# Install Python via Homebrew
brew install python

# Use Homebrew Python
/opt/homebrew/bin/python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### macOS Troubleshooting
```bash
# If SSL certificate errors:
/Applications/Python\ 3.x/Install\ Certificates.command

# If permission errors:
pip install --user -r requirements.txt

# If M1 Mac compatibility issues:
arch -arm64 pip install -r requirements.txt
```

### Linux Setup

#### Ubuntu/Debian
```bash
# Update package list
sudo apt update

# Install Python and pip
sudo apt install python3 python3-pip python3-venv git

# Clone repository
git clone https://github.com/your-username/Langchain-Playbook.git
cd Langchain-Playbook

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set API key permanently
echo 'export OPENAI_API_KEY="your_key_here"' >> ~/.bashrc
source ~/.bashrc
```

#### CentOS/RHEL/Fedora
```bash
# Install Python and development tools
sudo dnf install python3 python3-pip python3-venv git gcc python3-devel

# Or for CentOS/RHEL 7:
sudo yum install python3 python3-pip git gcc python3-devel

# Follow standard setup
git clone https://github.com/your-username/Langchain-Playbook.git
cd Langchain-Playbook
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Linux Troubleshooting
```bash
# If build tools missing:
sudo apt install build-essential  # Ubuntu/Debian
sudo dnf groupinstall "Development Tools"  # Fedora
sudo yum groupinstall "Development Tools"  # CentOS/RHEL

# If SSL/TLS errors:
pip install --trusted-host pypi.org --trusted-host pypi.python.org -r requirements.txt
```

---

## 🔑 API Key Configuration

### Supported Providers

#### OpenAI (Primary)
```bash
# Get API key from: https://platform.openai.com/api-keys
export OPENAI_API_KEY="sk-..."
```

#### Anthropic (Optional)
```bash
# Get API key from: https://console.anthropic.com/
export ANTHROPIC_API_KEY="sk-ant-..."
```

#### Google (Optional)
```bash
# Get API key from: https://makersuite.google.com/app/apikey
export GOOGLE_API_KEY="AI..."
```

### Configuration Methods

#### Method 1: Environment Variables (Recommended)
```bash
# Linux/macOS - Add to ~/.bashrc or ~/.zshrc
export OPENAI_API_KEY="your_key_here"
export ANTHROPIC_API_KEY="your_key_here"
export GOOGLE_API_KEY="your_key_here"

# Windows - Add to environment variables or use:
setx OPENAI_API_KEY "your_key_here"
```

#### Method 2: .env File (Project-specific)
```bash
# Copy template
cp .env.example .env

# Edit .env file
cat > .env << EOF
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here
EOF
```

#### Method 3: Runtime Configuration
```python
# In Python scripts
import os
os.environ['OPENAI_API_KEY'] = 'your_key_here'

# Or using utils
from utils.config import set_api_key
set_api_key('openai', 'your_key_here')
```

### Security Best Practices

#### ✅ Do This
- Use environment variables or .env files
- Keep API keys in secure password managers
- Rotate keys regularly
- Use different keys for development/production
- Monitor API usage and costs

#### ❌ Never Do This
```python
# Don't hardcode keys in source code
api_key = "sk-actual-key-here"  # NEVER!

# Don't commit .env files to version control
git add .env  # DANGEROUS!
```

#### Securing .env Files
```bash
# Add to .gitignore
echo ".env" >> .gitignore
echo "*.key" >> .gitignore

# Set proper permissions (Linux/macOS)
chmod 600 .env
```

---

## 🐳 Docker Setup (Optional)

### Quick Docker Start
```bash
# Build image
docker build -t langchain-playbook .

# Run with API key
docker run -e OPENAI_API_KEY=your_key_here -p 8888:8888 langchain-playbook

# Or use docker-compose
docker-compose up
```

### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'
services:
  langchain-playbook:
    build: .
    ports:
      - "8888:8888"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - .:/app
      - ./notebooks:/app/notebooks
```

---

## 📦 Development Setup

### For Contributors

#### 1. Fork and Clone
```bash
# Fork repository on GitHub, then:
git clone https://github.com/your-username/Langchain-Playbook.git
cd Langchain-Playbook

# Add upstream remote
git remote add upstream https://github.com/original-repo/Langchain-Playbook.git
```

#### 2. Development Environment
```bash
# Create development environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

#### 3. Code Quality Tools
```bash
# Install linting tools
pip install black flake8 isort mypy

# Format code
black .
isort .

# Check code quality
flake8 .
mypy .
```

#### 4. Testing Setup
```bash
# Run tests
python tests/test_suite.py --mode full

# Run specific tests
python tests/test_suite.py --mode specific --test-class TestBasicExamples

# Validate examples
python tests/validate_examples.py
```

---

## 🎓 IDE and Editor Setup

### VS Code Setup

#### Recommended Extensions
```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-toolsai.jupyter",
    "ms-python.black-formatter",
    "ms-python.isort",
    "ms-python.flake8"
  ]
}
```

#### VS Code Settings
```json
{
  "python.defaultInterpreterPath": "./.venv/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "jupyter.askForKernelRestart": false
}
```

### PyCharm Setup

#### 1. Open Project
- File → Open → Select Langchain-Playbook directory
- Configure Python interpreter: Settings → Project → Python Interpreter → Add → Existing environment → Select .venv/bin/python

#### 2. Configure Jupyter
- Settings → Languages & Frameworks → Jupyter
- Enable Jupyter support
- Set Jupyter server URL: http://localhost:8888

#### 3. Code Style
- Settings → Editor → Code Style → Python
- Import Black formatter configuration

### Jupyter Lab Setup
```bash
# Install JupyterLab
pip install jupyterlab

# Install extensions
pip install jupyterlab-git
pip install jupyterlab-lsp
pip install python-lsp-server

# Start JupyterLab
jupyter lab
```

---

## 🔧 Troubleshooting Installation

### Common Issues

#### Python Version Issues
```bash
# Check available Python versions
python --version
python3 --version
python3.11 --version

# Use specific Python version
python3.11 -m venv .venv
```

#### Package Installation Failures
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Clear pip cache
pip cache purge

# Install with verbose output
pip install -v langchain

# Use specific index
pip install -i https://pypi.org/simple/ langchain
```

#### Virtual Environment Issues
```bash
# Remove and recreate environment
rm -rf .venv  # or rmdir /s .venv on Windows
python -m venv .venv

# Ensure activation
which python  # Should show .venv path
```

#### Import Errors
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Verify installation
python -c "import langchain; print(langchain.__version__)"

# Check utils import
python -c "import sys; sys.path.append('utils'); from config import get_api_key; print('OK')"
```

#### Permission Errors
```bash
# Linux/macOS - Use user install
pip install --user -r requirements.txt

# Windows - Run as administrator or use:
pip install --user -r requirements.txt
```

### Platform-Specific Fixes

#### Windows Issues
```cmd
# PowerShell execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Long path support
git config --system core.longpaths true

# UTF-8 encoding
set PYTHONIOENCODING=utf-8
```

#### macOS Issues
```bash
# Xcode command line tools
sudo xcode-select --install

# Homebrew Python path
export PATH="/opt/homebrew/bin:$PATH"

# SSL certificates
/Applications/Python\ 3.x/Install\ Certificates.command
```

#### Linux Issues
```bash
# Missing development packages
sudo apt install python3-dev build-essential  # Ubuntu/Debian
sudo dnf install python3-devel gcc  # Fedora

# Locale issues
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
```

---

## ✅ Verification Checklist

### Installation Verification
- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed successfully
- [ ] API keys configured
- [ ] Health check passes
- [ ] First example runs

### Environment Test
```bash
# Complete verification script
echo "🔍 Running installation verification..."

# Check Python
python --version || echo "❌ Python not found"

# Check virtual environment
python -c "import sys; print('✅ Virtual env' if 'venv' in sys.executable else '❌ Not in virtual env')"

# Check LangChain
python -c "import langchain; print(f'✅ LangChain {langchain.__version__}')" || echo "❌ LangChain not installed"

# Check API key
python -c "import os; print('✅ API key configured' if os.getenv('OPENAI_API_KEY') else '⚠️ No API key (demo mode)')"

# Check utils
python -c "import sys; sys.path.append('utils'); from config import get_api_key; print('✅ Utils working')" || echo "❌ Utils not working"

# Run health check
python tests/test_suite.py --mode health

echo "🎉 Installation verification complete!"
```

### Ready to Start?
If all checks pass, you're ready to explore:
- 📚 [Basic Examples](../basics/) - Start here if you're new to LangChain
- 📓 [Interactive Notebooks](../notebooks/) - Hands-on learning experience
- 🚀 [Projects](../projects/) - Real-world applications
- 📖 [Documentation](../docs/) - Comprehensive guides

---

## 🆘 Getting Help

### Self-Help Resources
1. **Check [Troubleshooting Guide](../docs/troubleshooting.md)**
2. **Review [FAQ](../docs/faq.md)**
3. **Run diagnostic tools**: `python tests/test_suite.py --mode health`

### Community Support
- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share solutions
- **Discord/Slack**: Real-time community help

### When Reporting Issues
Include:
- Operating system and version
- Python version
- Error messages (full traceback)
- Steps to reproduce

**Happy Learning! 🚀**