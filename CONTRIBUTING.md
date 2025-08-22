# Contributing to LangChain Playbook

Thank you for your interest in contributing to the LangChain Playbook! This guide will help you get started.

## 🤝 How to Contribute

### Types of Contributions
- **Examples and Tutorials**: Add new examples or improve existing ones
- **Documentation**: Improve README files, add explanations, fix typos
- **Bug Fixes**: Fix issues in existing code
- **New Features**: Add new modules or enhance existing functionality
- **Testing**: Add or improve test cases

### Getting Started

1. **Fork the Repository**
   ```bash
   # Click the "Fork" button on GitHub
   git clone https://github.com/your-username/Langchain-Playbook.git
   cd Langchain-Playbook
   ```

2. **Set Up Development Environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # macOS/Linux
   pip install -r requirements.txt
   ```

3. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Development Guidelines

#### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Include type hints where appropriate

#### Example Format
Each example should include:
```python
#!/usr/bin/env python3
"""
Brief description of what this example demonstrates.

Key concepts:
- List key concepts
- That are demonstrated
- In this example
"""

import sys
from pathlib import Path

# Add utils to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import setup_logging, get_logger, get_api_key

def main():
    """Main function demonstrating the concept."""
    setup_logging()
    logger = get_logger(__name__)
    
    # Your example code here
    print("✅ Example completed successfully!")

if __name__ == "__main__":
    main()
```

#### Documentation Standards
- Each module must have a README.md
- Include clear learning objectives
- Provide step-by-step instructions
- Add troubleshooting sections
- Include practical exercises

#### Testing Requirements
- Add tests for new functionality
- Ensure existing tests pass
- Include both unit and integration tests
- Test with and without API keys (demo mode)

### Submission Process

1. **Make Your Changes**
   - Follow the coding guidelines
   - Test your changes thoroughly
   - Update documentation as needed

2. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add example for custom tool development"
   ```

3. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a Pull Request**
   - Go to the original repository on GitHub
   - Click "New Pull Request"
   - Select your feature branch
   - Fill out the PR template

### Pull Request Guidelines

#### PR Title Format
Use conventional commits format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions
- `refactor:` for code refactoring

#### PR Description Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (please describe):

## Testing
- [ ] Tested locally
- [ ] Added new tests
- [ ] All existing tests pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes
```

### Review Process

1. **Automated Checks**: CI/CD will run tests and style checks
2. **Maintainer Review**: A maintainer will review your code
3. **Feedback**: Address any requested changes
4. **Approval**: Once approved, your PR will be merged

### Community Guidelines

#### Be Respectful
- Use inclusive language
- Be constructive in feedback
- Help newcomers learn
- Follow the code of conduct

#### Best Practices
- Start small with your first contribution
- Ask questions if you're unsure
- Reference issues in your PRs
- Keep PRs focused on a single change

## 🐛 Reporting Issues

### Bug Reports
Include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages or logs

### Feature Requests
Include:
- Use case description
- Proposed solution
- Alternative solutions considered
- Additional context

## 📚 Development Setup

### Required Tools
- Python 3.8+
- Git
- Code editor (VS Code recommended)
- API keys for testing (optional)

### Recommended Extensions (VS Code)
- Python
- Pylance
- Python Docstring Generator
- GitLens
- Jupyter

### Environment Variables
Create a `.env` file for development:
```env
# Development mode
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# API keys (optional for development)
OPENAI_API_KEY=your_key_here
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_basics/

# Run with coverage
pytest --cov=. tests/

# Run integration tests
pytest tests/integration/
```

### Test Structure
```
tests/
├── test_basics/
├── test_intermediate/
├── test_advanced/
├── test_projects/
├── integration/
└── fixtures/
```

### Writing Tests
- Use descriptive test names
- Test both success and failure cases
- Mock external APIs when possible
- Include integration tests for complete workflows

## 📖 Documentation

### Adding Examples
1. Create appropriately numbered directory
2. Add main example file
3. Include README with learning objectives
4. Add to parent section's README
5. Update main learning guide

### Documentation Structure
```
section/
├── README.md (section overview)
├── 01_topic/
│   ├── example.py
│   └── README.md
└── 02_next_topic/
    ├── example.py
    └── README.md
```

## 🏷️ Release Process

### Versioning
We follow semantic versioning:
- MAJOR: Breaking changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes, backward compatible

### Release Notes
- Highlight new features
- List bug fixes
- Include upgrade instructions
- Thank contributors

## 📞 Getting Help

### Channels
- **GitHub Discussions**: General questions and ideas
- **Issues**: Bug reports and feature requests
- **Discord**: Real-time community chat
- **Email**: maintainer@langchain-playbook.com

### Maintainers
- **@username1**: Lead maintainer
- **@username2**: Documentation lead
- **@username3**: Testing and CI/CD

## 🎉 Recognition

### Contributors
All contributors will be:
- Listed in CONTRIBUTORS.md
- Thanked in release notes
- Invited to join maintainer team (for significant contributions)

### Hall of Fame
Special recognition for:
- First-time contributors
- Outstanding examples
- Significant improvements
- Community building

---

**Thank you for making LangChain Playbook better for everyone! 🚀**