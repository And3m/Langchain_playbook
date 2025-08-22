# LangChain Playbook 🦜🔗

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-latest-green.svg)](https://github.com/langchain-ai/langchain)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

A comprehensive, hands-on learning resource for mastering LangChain - from basic concepts to advanced applications. This playbook provides structured tutorials, practical examples, and real-world projects to help you build powerful LLM applications.

> **⭐ Star this repository** if you find it helpful! It helps others discover this resource.

## 🎯 What You'll Learn

- **Fundamentals**: LLMs, Chat models, Prompts, Chains, and Output Parsers
- **Intermediate**: Memory, Agents, RAG, Callbacks, and Async operations
- **Advanced**: Custom tools, Multi-agent systems, Evaluation, and Deployment
- **Projects**: Build real-world applications like chatbots, document Q&A, and code assistants

## 🏗️ Project Structure

```
Langchain-Playbook/
├── 📚 basics/           # Core LangChain concepts
├── 🔧 intermediate/     # Advanced features and patterns
├── 🚀 advanced/         # Expert-level topics
├── 🎯 projects/         # Real-world applications
├── 📓 notebooks/        # Interactive Jupyter tutorials
├── 🛠️ utils/           # Shared utilities and helpers
├── 🧪 tests/           # Test suites and validation
└── 📖 docs/            # Documentation and guides
```

## 📚 Table of Contents

- [🎆 Quick Start](#-quick-start)
- [📝 Learning Path](#-learning-path)
- [📁 Project Structure](#-project-structure)
- [🔧 Configuration](#-configuration)
- [📚 Documentation](#-documentation)
- [🧪 Testing](#-testing)
- [🤝 Contributing](#-contributing)

## 🎆 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd Langchain-Playbook
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 5. Start Learning!

```bash
# Start with basics
cd basics/01_getting_started
python hello_langchain.py

# Or run Jupyter notebooks
jupyter notebook
```

## 📋 Learning Path

### 🎓 Beginner Track (basics/)
1. **Getting Started** - Hello World and basic setup
2. **Models** - LLMs, Chat models, and Embeddings
3. **Prompts** - Prompt templates and engineering
4. **Chains** - Composing operations and workflows
5. **Output Parsers** - Structured data from LLM responses

### 🔬 Intermediate Track (intermediate/)
1. **Memory** - Conversation context and state management
2. **Agents** - Tool-using AI agents
3. **Retrieval** - RAG systems and vector databases
4. **Callbacks** - Monitoring and logging
5. **Async** - High-performance async operations

### 🎯 Advanced Track (advanced/)
1. **Custom Tools** - Building specialized tools
2. **Multi-Agent** - Coordinated AI systems
3. **Evaluation** - Testing and benchmarking
4. **Deployment** - Production-ready applications
5. **Optimization** - Performance and cost optimization

### 🏗️ Projects Track (projects/)
1. **Chatbot** - Conversational AI with memory
2. **Document Q&A** - RAG-powered document assistant
3. **Code Assistant** - AI-powered code generation
4. **Research Assistant** - Information gathering and synthesis
5. **API Service** - LangChain as a web service

## 🔧 Configuration

### API Keys
Create a `.env` file with your API keys:

```env
# OpenAI
OPENAI_API_KEY=your_openai_key_here

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_key_here

# Google
GOOGLE_API_KEY=your_google_key_here

# Pinecone (optional)
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_ENVIRONMENT=your_pinecone_env_here
```

### Supported Model Providers
- ✅ OpenAI (GPT-3.5, GPT-4)
- ✅ Anthropic (Claude)
- ✅ Google (Gemini)
- ✅ Local models (Ollama, LM Studio)
- ✅ Hugging Face models

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [API Keys Setup](docs/api-keys.md)
- [Troubleshooting](docs/troubleshooting.md)

## 🧪 Testing

Run tests to validate your setup:

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_basics/

# Run with verbose output
pytest -v tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your examples or improvements
4. Include tests and documentation
5. Submit a pull request

## 📝 Requirements

- Python 3.8+
- API keys for your chosen LLM providers
- Optional: Jupyter for interactive notebooks

## 🆘 Getting Help

- 📖 Check the [documentation](docs/)
- 🐛 Report issues on GitHub
- 💬 Join our community discussions

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- LangChain team for the amazing framework
- Community contributors and feedback
- Educational resources and examples

---

**Ready to build amazing LLM applications? Let's get started! 🚀**
