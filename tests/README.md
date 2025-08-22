# Testing and Validation 🧪

Comprehensive testing suite for the LangChain Playbook to ensure all examples, projects, and utilities work correctly.

## 📋 Test Suite Overview

### 🔍 Health Check
Quick validation that the environment is properly configured:
```bash
python tests/test_suite.py --mode health
```

### 🧪 Full Test Suite
Comprehensive testing of all components:
```bash
python tests/test_suite.py --mode full
```

### 📝 Example Validation
Validates all examples can be imported and executed:
```bash
python tests/validate_examples.py
```

## 🎯 What Gets Tested

### Environment Setup
- ✅ Python version compatibility (3.8+)
- ✅ Required package availability
- ✅ Project structure completeness
- ✅ Utility module imports
- ✅ Configuration management

### Basic Examples
- ✅ Hello LangChain functionality
- ✅ Prompt template creation
- ✅ Chain composition
- ✅ Memory integration
- ✅ Output parsing

### Intermediate Examples
- ✅ RAG implementation
- ✅ Agent functionality
- ✅ Vector database integration
- ✅ Async operations
- ✅ Callback handling

### Project Functionality
- ✅ Chatbot application
- ✅ Document Q&A system
- ✅ Code assistant
- ✅ Research assistant
- ✅ API service

### Jupyter Notebooks
- ✅ Notebook structure validation
- ✅ Cell content verification
- ✅ JSON format compliance
- ✅ Educational content presence

## 🚀 Running Tests

### Prerequisites
```bash
# Install test dependencies
pip install pytest unittest2

# Ensure you're in the project root
cd /path/to/Langchain-Playbook
```

### Quick Health Check
```bash
# Basic system validation
python tests/test_suite.py --mode health
```

**Expected Output:**
```
🔍 Running LangChain Playbook Health Check...
✅ Python 3.11
✅ LangChain 0.2.0
✅ basics/ directory
✅ intermediate/ directory
✅ projects/ directory
✅ utils/ directory
✅ Utils module

🎉 All systems go! LangChain Playbook is ready.
```

### Comprehensive Testing
```bash
# Run all tests
python tests/test_suite.py --mode full
```

**Test Categories:**
- **Environment Setup**: System compatibility
- **Utility Functions**: Configuration and logging
- **Basic Examples**: Core LangChain concepts
- **Intermediate Examples**: Advanced patterns
- **Project Functionality**: Real-world applications
- **Integration Tests**: End-to-end workflows
- **Validation Tests**: Code quality checks

### Example Validation
```bash
# Validate all examples work
python tests/validate_examples.py
```

**Validation Process:**
1. **Import Testing**: Verify modules import correctly
2. **Execution Testing**: Run examples in controlled environment
3. **Notebook Validation**: Check Jupyter notebook structure
4. **Project Testing**: Validate project classes and functions

## 📊 Test Results Interpretation

### Success Indicators
- ✅ **All tests pass**: System fully functional
- ⚠️ **Warnings only**: Minor issues, generally safe to proceed
- ❌ **Errors present**: Critical issues requiring attention

### Common Test Scenarios

#### With API Keys
```bash
# Set your API key
export OPENAI_API_KEY=your_actual_key

# Run tests - will test actual API integration
python tests/validate_examples.py
```

#### Without API Keys (Demo Mode)
```bash
# Tests run in demo mode with mocked responses
python tests/validate_examples.py
```

#### Specific Test Classes
```bash
# Run only environment tests
python tests/test_suite.py --mode specific --test-class TestEnvironmentSetup

# Run only project tests
python tests/test_suite.py --mode specific --test-class TestProjectFunctionality
```

## 🔧 Test Configuration

### Environment Variables
```bash
# API keys for testing
export OPENAI_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key
export GOOGLE_API_KEY=your_key

# Test configuration
export PYTHONPATH=/path/to/Langchain-Playbook
export TEST_MODE=demo  # or 'live' for real API testing
```

### Test Modes

#### Demo Mode (Default)
- Uses mocked API responses
- Tests code structure and logic
- Safe to run without API keys
- Fast execution

#### Live Mode
- Uses real API calls
- Tests actual integrations
- Requires valid API keys
- Slower execution, consumes API credits

## 🐛 Troubleshooting Tests

### Common Issues

#### "ModuleNotFoundError"
```bash
# Solution: Ensure correct Python path
export PYTHONPATH=/path/to/Langchain-Playbook
cd /path/to/Langchain-Playbook
python tests/test_suite.py --mode health
```

#### "Import errors in projects"
```bash
# Solution: Check project structure
ls -la projects/
# Ensure all project directories have required files
```

#### "API key errors"
```bash
# Solution: Use demo mode or set valid API key
unset OPENAI_API_KEY  # For demo mode
# or
export OPENAI_API_KEY=your_valid_key
```

#### "Timeout errors"
```bash
# Solution: Increase timeout or use demo mode
# Edit validate_examples.py timeout parameter
timeout=60  # Increase from default 30 seconds
```

### Debug Mode
```python
# Enable verbose output
python tests/test_suite.py --mode full -v

# Add debug prints
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Continuous Integration

### GitHub Actions Integration
```yaml
# .github/workflows/test.yml
name: Test LangChain Playbook
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run health check
      run: python tests/test_suite.py --mode health
    - name: Validate examples
      run: python tests/validate_examples.py
```

### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Set up hooks
cat > .pre-commit-config.yaml << EOF
repos:
- repo: local
  hooks:
  - id: test-health
    name: Health Check
    entry: python tests/test_suite.py --mode health
    language: system
    pass_filenames: false
EOF

# Install hooks
pre-commit install
```

## 🎯 Custom Test Development

### Adding New Tests

#### 1. Environment Tests
```python
class TestNewFeature(unittest.TestCase):
    def test_new_functionality(self):
        # Your test code here
        self.assertTrue(True)
```

#### 2. Project Tests
```python
def test_new_project_import(self):
    try:
        import new_project_module
        self.assertTrue(hasattr(new_project_module, 'MainClass'))
    except ImportError as e:
        self.skipTest(f"New project import failed: {e}")
```

#### 3. Integration Tests
```python
@patch('langchain.chat_models.ChatOpenAI')
def test_new_integration(self, mock_chat):
    # Mock and test integration
    mock_chat.return_value.predict.return_value = "Test response"
    # Test your integration
```

### Test Best Practices

#### Mocking External Services
```python
# Mock LLM calls
@patch('langchain.chat_models.ChatOpenAI')
def test_with_mock_llm(self, mock_chat):
    mock_llm = Mock()
    mock_llm.predict.return_value = "Mocked response"
    mock_chat.return_value = mock_llm
    # Your test code
```

#### Error Handling
```python
def test_error_handling(self):
    with self.assertRaises(ExpectedException):
        # Code that should raise exception
        pass
```

#### Resource Cleanup
```python
def setUp(self):
    self.temp_file = tempfile.NamedTemporaryFile(delete=False)

def tearDown(self):
    os.unlink(self.temp_file.name)
```

## 📊 Performance Testing

### Execution Time Monitoring
```python
import time

def test_performance(self):
    start_time = time.time()
    # Execute code under test
    execution_time = time.time() - start_time
    self.assertLess(execution_time, 5.0, "Code should execute within 5 seconds")
```

### Memory Usage Testing
```python
import psutil
import os

def test_memory_usage(self):
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss
    # Execute code under test
    memory_after = process.memory_info().rss
    memory_increase = memory_after - memory_before
    self.assertLess(memory_increase, 100 * 1024 * 1024, "Memory increase should be < 100MB")
```

## 🔄 Test Automation

### Automated Test Execution
```bash
#!/bin/bash
# run_tests.sh
set -e

echo "🧪 Running automated test suite..."

# Health check
echo "1. Health check..."
python tests/test_suite.py --mode health

# Example validation
echo "2. Example validation..."
python tests/validate_examples.py

# Full test suite
echo "3. Full test suite..."
python tests/test_suite.py --mode full

echo "✅ All tests completed successfully!"
```

### Scheduled Testing
```bash
# Add to crontab for daily testing
# Run tests daily at 2 AM
0 2 * * * cd /path/to/Langchain-Playbook && ./run_tests.sh
```

---

## 🎉 Success Criteria

The LangChain Playbook passes validation when:
- ✅ All environment checks pass
- ✅ All examples import successfully
- ✅ All projects have required classes
- ✅ All notebooks have valid structure
- ✅ No critical errors in execution tests
- ✅ Success rate > 90%

**Ready to validate? Run the tests and ensure your LangChain Playbook is production-ready! 🚀**