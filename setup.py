#!/usr/bin/env python3
"""
LangChain Playbook Setup Script

Automated setup and configuration script for the LangChain Playbook.
This script handles environment setup, dependency installation, and initial configuration.

Usage:
    python setup.py              # Interactive setup
    python setup.py --auto       # Automated setup with defaults
    python setup.py --dev        # Development setup
    python setup.py --docker     # Docker setup
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse


class Colors:
    """Terminal colors for pretty output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class SetupManager:
    """Manages the setup process for LangChain Playbook."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent
        self.venv_path = self.project_root / '.venv'
        self.env_file = self.project_root / '.env'
        self.requirements_file = self.project_root / 'requirements.txt'
        
    def print_header(self, text: str):
        """Print formatted header."""
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")
    
    def print_success(self, text: str):
        """Print success message."""
        print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")
    
    def print_warning(self, text: str):
        """Print warning message."""
        print(f"{Colors.WARNING}⚠️ {text}{Colors.ENDC}")
    
    def print_error(self, text: str):
        """Print error message."""
        print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")
    
    def print_info(self, text: str):
        """Print info message."""
        print(f"{Colors.OKBLUE}ℹ️ {text}{Colors.ENDC}")
    
    def run_command(self, command: List[str], cwd: Path = None, check: bool = True) -> Tuple[bool, str, str]:
        """Run a command and return success status and output."""
        try:
            result = subprocess.run(
                command,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                check=check
            )
            return True, result.stdout, result.stderr
        except subprocess.CalledProcessError as e:
            return False, e.stdout, e.stderr
        except FileNotFoundError:
            return False, "", f"Command not found: {command[0]}"
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            self.print_error(f"Python 3.8+ required, found {version.major}.{version.minor}")
            return False
        
        self.print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    
    def check_git(self) -> bool:
        """Check if Git is available."""
        success, stdout, stderr = self.run_command(['git', '--version'], check=False)
        if success:
            self.print_success(f"Git available: {stdout.strip()}")
            return True
        else:
            self.print_warning("Git not found - some features may not work")
            return False
    
    def create_virtual_environment(self) -> bool:
        """Create virtual environment."""
        if self.venv_path.exists():
            self.print_info("Virtual environment already exists")
            return True
        
        self.print_info("Creating virtual environment...")
        success, stdout, stderr = self.run_command([
            sys.executable, '-m', 'venv', str(self.venv_path)
        ])
        
        if success:
            self.print_success("Virtual environment created")
            return True
        else:
            self.print_error(f"Failed to create virtual environment: {stderr}")
            return False
    
    def get_python_executable(self) -> str:
        """Get Python executable in virtual environment."""
        if os.name == 'nt':  # Windows
            return str(self.venv_path / 'Scripts' / 'python.exe')
        else:  # Unix-like
            return str(self.venv_path / 'bin' / 'python')
    
    def get_pip_executable(self) -> str:
        """Get pip executable in virtual environment."""
        if os.name == 'nt':  # Windows
            return str(self.venv_path / 'Scripts' / 'pip.exe')
        else:  # Unix-like
            return str(self.venv_path / 'bin' / 'pip')
    
    def install_dependencies(self, dev_mode: bool = False) -> bool:
        """Install Python dependencies."""
        if not self.requirements_file.exists():
            self.print_error(f"Requirements file not found: {self.requirements_file}")
            return False
        
        pip_executable = self.get_pip_executable()
        
        # Upgrade pip first
        self.print_info("Upgrading pip...")
        success, stdout, stderr = self.run_command([
            pip_executable, 'install', '--upgrade', 'pip'
        ])
        
        if not success:
            self.print_warning(f"Pip upgrade failed: {stderr}")
        
        # Install requirements
        self.print_info("Installing dependencies...")
        success, stdout, stderr = self.run_command([
            pip_executable, 'install', '-r', str(self.requirements_file)
        ])
        
        if success:
            self.print_success("Dependencies installed")
            
            # Install development dependencies if requested
            if dev_mode:
                dev_requirements = self.project_root / 'requirements-dev.txt'
                if dev_requirements.exists():
                    self.print_info("Installing development dependencies...")
                    success, stdout, stderr = self.run_command([
                        pip_executable, 'install', '-r', str(dev_requirements)
                    ])
                    if success:
                        self.print_success("Development dependencies installed")
                    else:
                        self.print_warning(f"Development dependencies failed: {stderr}")
            
            return True
        else:
            self.print_error(f"Failed to install dependencies: {stderr}")
            return False
    
    def create_env_file(self, interactive: bool = True) -> bool:
        """Create .env file with API keys."""
        if self.env_file.exists():
            self.print_info(".env file already exists")
            if interactive:
                response = input("Do you want to update it? (y/n): ").lower()
                if response != 'y':
                    return True
        
        env_content = []
        
        if interactive:
            self.print_info("Setting up environment variables...")
            print("You can skip any API key by pressing Enter (you can add them later)")
            
            # API Keys
            api_keys = {
                'OPENAI_API_KEY': 'OpenAI API key (sk-...)',
                'ANTHROPIC_API_KEY': 'Anthropic API key (sk-ant-...)',
                'GOOGLE_API_KEY': 'Google API key (AI...)',
                'HUGGINGFACE_API_TOKEN': 'Hugging Face token (hf_...)'
            }
            
            for key, description in api_keys.items():
                value = input(f"Enter {description}: ").strip()
                if value:
                    env_content.append(f"{key}={value}")
                else:
                    env_content.append(f"# {key}=your_key_here")
        else:
            # Create template
            env_content = [
                "# LangChain Playbook Environment Variables",
                "# Add your API keys here",
                "",
                "# OpenAI API Key (required for most examples)",
                "# Get from: https://platform.openai.com/api-keys",
                "# OPENAI_API_KEY=sk-your_openai_key_here",
                "",
                "# Anthropic API Key (optional, for Claude models)",
                "# Get from: https://console.anthropic.com/",
                "# ANTHROPIC_API_KEY=sk-ant-your_anthropic_key_here",
                "",
                "# Google API Key (optional, for Gemini models)",
                "# Get from: https://makersuite.google.com/app/apikey",
                "# GOOGLE_API_KEY=your_google_key_here",
                "",
                "# Hugging Face Token (optional, for HF models)",
                "# Get from: https://huggingface.co/settings/tokens",
                "# HUGGINGFACE_API_TOKEN=hf_your_token_here",
                "",
                "# Application Settings",
                "# DEBUG=false",
                "# LOG_LEVEL=INFO"
            ]
        
        try:
            with open(self.env_file, 'w') as f:
                f.write('\n'.join(env_content))
            
            self.print_success(f"Environment file created: {self.env_file}")
            if not interactive:
                self.print_info("Remember to add your API keys to the .env file")
            
            return True
        except Exception as e:
            self.print_error(f"Failed to create .env file: {e}")
            return False
    
    def validate_installation(self) -> bool:
        """Validate the installation."""
        self.print_info("Validating installation...")
        
        python_executable = self.get_python_executable()
        
        # Test basic imports
        test_script = '''
import sys
sys.path.append("utils")

try:
    import langchain
    print(f"✅ LangChain {langchain.__version__}")
except ImportError as e:
    print(f"❌ LangChain import failed: {e}")
    sys.exit(1)

try:
    from utils.config import get_api_key
    print("✅ Utils module working")
except ImportError:
    print("⚠️ Utils module not available (you may need to run from project directory)")

print("✅ Basic validation passed")
'''
        
        success, stdout, stderr = self.run_command([
            python_executable, '-c', test_script
        ])
        
        if success:
            print(stdout)
            return True
        else:
            self.print_error(f"Validation failed: {stderr}")
            return False
    
    def run_health_check(self) -> bool:
        """Run comprehensive health check."""
        self.print_info("Running health check...")
        
        python_executable = self.get_python_executable()
        test_script_path = self.project_root / 'tests' / 'test_suite.py'
        
        if test_script_path.exists():
            success, stdout, stderr = self.run_command([
                python_executable, str(test_script_path), '--mode', 'health'
            ])
            
            if success:
                print(stdout)
                return True
            else:
                self.print_warning(f"Health check had issues: {stderr}")
                return False
        else:
            self.print_warning("Health check script not found")
            return True
    
    def setup_development_environment(self) -> bool:
        """Setup development environment with additional tools."""
        pip_executable = self.get_pip_executable()
        
        dev_packages = [
            'jupyter',
            'jupyterlab',
            'black',
            'flake8',
            'isort',
            'mypy',
            'pytest',
            'pre-commit'
        ]
        
        self.print_info("Installing development tools...")
        for package in dev_packages:
            success, stdout, stderr = self.run_command([
                pip_executable, 'install', package
            ], check=False)
            
            if success:
                self.print_success(f"Installed {package}")
            else:
                self.print_warning(f"Failed to install {package}: {stderr}")
        
        # Setup pre-commit hooks
        if (self.project_root / '.git').exists():
            success, stdout, stderr = self.run_command([
                self.get_python_executable(), '-m', 'pre_commit', 'install'
            ], check=False)
            
            if success:
                self.print_success("Pre-commit hooks installed")
            else:
                self.print_warning("Failed to install pre-commit hooks")
        
        return True
    
    def setup_docker_environment(self) -> bool:
        """Setup Docker environment."""
        # Check if Docker is available
        success, stdout, stderr = self.run_command(['docker', '--version'], check=False)
        if not success:
            self.print_error("Docker not found. Please install Docker first.")
            return False
        
        self.print_success(f"Docker available: {stdout.strip()}")
        
        # Create Dockerfile if not exists
        dockerfile_path = self.project_root / 'Dockerfile'
        if not dockerfile_path.exists():
            dockerfile_content = '''FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["python", "-m", "uvicorn", "projects.api_service.api_service:app", "--host", "0.0.0.0", "--port", "8000"]
'''
            
            try:
                with open(dockerfile_path, 'w') as f:
                    f.write(dockerfile_content)
                self.print_success("Dockerfile created")
            except Exception as e:
                self.print_error(f"Failed to create Dockerfile: {e}")
                return False
        
        # Create docker-compose.yml if not exists
        compose_path = self.project_root / 'docker-compose.yml'
        if not compose_path.exists():
            compose_content = '''version: '3.8'

services:
  langchain-playbook:
    build: .
    ports:
      - "8000:8000"
      - "8888:8888"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    volumes:
      - .:/app
      - ./notebooks:/app/notebooks
    command: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''
'''
            
            try:
                with open(compose_path, 'w') as f:
                    f.write(compose_content)
                self.print_success("docker-compose.yml created")
            except Exception as e:
                self.print_error(f"Failed to create docker-compose.yml: {e}")
                return False
        
        self.print_info("To start with Docker:")
        self.print_info("  docker-compose up")
        
        return True
    
    def print_next_steps(self, setup_type: str = "standard"):
        """Print next steps for the user."""
        self.print_header("🎉 Setup Complete!")
        
        print(f"{Colors.OKGREEN}Your LangChain Playbook is ready to use!{Colors.ENDC}\n")
        
        if setup_type == "docker":
            print(f"{Colors.OKBLUE}🐳 Docker Setup:{Colors.ENDC}")
            print("   docker-compose up")
            print("   Open: http://localhost:8888 (Jupyter)")
            print("   Open: http://localhost:8000 (API)")
        else:
            print(f"{Colors.OKBLUE}🚀 Next Steps:{Colors.ENDC}")
            
            # Activation command
            if os.name == 'nt':  # Windows
                print(f"   {Colors.BOLD}1. Activate environment:{Colors.ENDC}")
                print("      .venv\\Scripts\\activate")
            else:  # Unix-like
                print(f"   {Colors.BOLD}1. Activate environment:{Colors.ENDC}")
                print("      source .venv/bin/activate")
            
            print(f"\n   {Colors.BOLD}2. Add API keys to .env file{Colors.ENDC}")
            
            print(f"\n   {Colors.BOLD}3. Start learning:{Colors.ENDC}")
            print("      python basics/01_getting_started/hello_langchain.py")
            print("      jupyter notebook  # For interactive learning")
            
            if setup_type == "dev":
                print(f"\n   {Colors.BOLD}4. Development tools:{Colors.ENDC}")
                print("      black .          # Format code")
                print("      flake8 .         # Check code quality")
                print("      pytest           # Run tests")
        
        print(f"\n{Colors.OKBLUE}📚 Resources:{Colors.ENDC}")
        print("   • README.md - Project overview")
        print("   • docs/installation.md - Detailed setup guide")
        print("   • docs/troubleshooting.md - Common issues")
        print("   • notebooks/ - Interactive tutorials")
        
        print(f"\n{Colors.WARNING}⚠️ Remember:{Colors.ENDC}")
        print("   • Add your API keys to .env file")
        print("   • Keep your API keys secure")
        print("   • Check docs/ for additional guides")
        
        print(f"\n{Colors.OKGREEN}Happy learning with LangChain! 🦜🔗{Colors.ENDC}")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="LangChain Playbook Setup Script")
    parser.add_argument('--auto', action='store_true', help='Automated setup with defaults')
    parser.add_argument('--dev', action='store_true', help='Development setup with additional tools')
    parser.add_argument('--docker', action='store_true', help='Docker setup')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency installation')
    parser.add_argument('--skip-env', action='store_true', help='Skip .env file creation')
    
    args = parser.parse_args()
    
    setup = SetupManager()
    
    # Welcome message
    setup.print_header("🦜🔗 LangChain Playbook Setup")
    print(f"{Colors.OKBLUE}Welcome to the LangChain Playbook setup!{Colors.ENDC}")
    print(f"{Colors.OKBLUE}This script will help you get started quickly.{Colors.ENDC}\n")
    
    # Determine setup type
    if args.docker:
        setup_type = "docker"
    elif args.dev:
        setup_type = "dev"
    else:
        setup_type = "standard"
    
    print(f"Setup type: {Colors.BOLD}{setup_type}{Colors.ENDC}\n")
    
    try:
        # Step 1: Check prerequisites
        setup.print_header("Step 1: Checking Prerequisites")
        
        if not setup.check_python_version():
            sys.exit(1)
        
        setup.check_git()
        
        # Step 2: Docker setup (if requested)
        if args.docker:
            setup.print_header("Step 2: Docker Setup")
            if not setup.setup_docker_environment():
                sys.exit(1)
            setup.print_next_steps("docker")
            return
        
        # Step 3: Virtual environment
        setup.print_header("Step 2: Virtual Environment")
        if not setup.create_virtual_environment():
            sys.exit(1)
        
        # Step 4: Dependencies
        if not args.skip_deps:
            setup.print_header("Step 3: Installing Dependencies")
            if not setup.install_dependencies(dev_mode=args.dev):
                sys.exit(1)
        
        # Step 5: Environment configuration
        if not args.skip_env:
            setup.print_header("Step 4: Environment Configuration")
            if not setup.create_env_file(interactive=not args.auto):
                sys.exit(1)
        
        # Step 6: Development tools (if requested)
        if args.dev:
            setup.print_header("Step 5: Development Tools")
            setup.setup_development_environment()
        
        # Step 7: Validation
        setup.print_header("Step 6: Validation")
        if not setup.validate_installation():
            setup.print_warning("Validation had issues, but setup may still work")
        
        # Step 8: Health check
        setup.run_health_check()
        
        # Final steps
        setup.print_next_steps(setup_type)
        
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Setup interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        setup.print_error(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()