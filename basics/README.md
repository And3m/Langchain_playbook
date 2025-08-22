# LangChain Basics - Foundation Concepts

Welcome to the LangChain Basics section! This is your starting point for mastering LangChain fundamentals. Each module builds upon the previous one, creating a solid foundation for advanced concepts.

## 📚 Learning Path Overview

```
01_getting_started → 02_models → 03_prompts → 04_chains → 05_output_parsers
      ↓                ↓           ↓           ↓             ↓
   Hello World     Model Types   Templates   Workflows   Structured Data
```

## 🎯 Modules in this Section

### [01_getting_started](01_getting_started/) - Your First Steps
**🚀 Start here if you're new to LangChain**

- **hello_langchain.py**: Your first LangChain application
- **basic_chat.py**: Introduction to chat models
- **environment_setup.py**: Validate your setup

**Key Concepts**: LLM instantiation, basic prompting, API integration

### [02_models](02_models/) - Understanding Language Models
**🤖 Learn about different model types and providers**

- **llm_comparison.py**: Compare models and parameters
- **README.md**: Model selection guidelines

**Key Concepts**: Completion vs chat models, temperature, max tokens, providers

### [03_prompts](03_prompts/) - Mastering Prompt Engineering
**📝 Create effective prompts and templates**

- **prompt_templates.py**: Basic templates and variables
- **advanced_prompting.py**: Chain-of-thought, self-consistency
- **README.md**: Prompt engineering best practices

**Key Concepts**: Templates, few-shot learning, prompt optimization

### [04_chains](04_chains/) - Building Workflows
**⛓️ Connect components into powerful workflows**

- **basic_chains.py**: Sequential and conditional chains
- **README.md**: Chain composition patterns

**Key Concepts**: LLM chains, sequential processing, composition patterns

### [05_output_parsers](05_output_parsers/) - Structured Data
**📊 Convert text responses to structured data**

- **output_parsers.py**: Built-in and custom parsers
- **README.md**: Parsing strategies and validation

**Key Concepts**: Pydantic models, validation, error handling

## 🎓 What You'll Learn

By completing this section, you'll understand:

1. **Core Components**: LLMs, prompts, chains, parsers
2. **Integration**: How components work together
3. **Best Practices**: Effective patterns and techniques
4. **Error Handling**: Robust application development
5. **Optimization**: Performance and cost considerations

## 📋 Prerequisites

### Required
- Python 3.8+ installed
- Basic Python programming knowledge
- API key for at least one LLM provider (OpenAI recommended)

### Setup Steps
1. **Environment Setup**:
   ```bash
   python -m venv .venv
   .venv\\Scripts\\activate  # Windows
   pip install -r requirements.txt
   ```

2. **API Keys**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Validation**:
   ```bash
   python basics/01_getting_started/environment_setup.py
   ```

## 🚀 Quick Start Guide

### Option 1: Follow the Learning Path
```bash
# Start from the beginning
cd basics/01_getting_started
python hello_langchain.py

# Continue through each module
cd ../02_models
python llm_comparison.py

# And so on...
```

### Option 2: Jump to Specific Topics
```bash
# Interested in prompts?
cd basics/03_prompts
python prompt_templates.py

# Want to learn about chains?
cd basics/04_chains  
python basic_chains.py
```

### Option 3: Run Everything
```bash
# Run all basic examples (requires API keys)
python -c \"
import subprocess
import os

modules = [
    'basics/01_getting_started/hello_langchain.py',
    'basics/02_models/llm_comparison.py',
    'basics/03_prompts/prompt_templates.py',
    'basics/04_chains/basic_chains.py',
    'basics/05_output_parsers/output_parsers.py'
]

for module in modules:
    print(f'\n🚀 Running {module}...')
    subprocess.run(['python', module])
\"
```

## 💡 Key Concepts Summary

### LangChain Architecture
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Prompts   │───▶│    LLMs     │───▶│   Parsers   │
│ (Templates) │    │  (Models)   │    │ (Structure) │
└─────────────┘    └─────────────┘    └─────────────┘
       ▲                   ▲                   ▲
       │                   │                   │
       └───────────────────┼───────────────────┘
                          │
                   ┌─────────────┐
                   │   Chains    │
                   │(Workflows)  │
                   └─────────────┘
```

### Component Relationships
- **Prompts** → **LLMs**: Templates provide structured input
- **LLMs** → **Parsers**: Raw text becomes structured data
- **Chains**: Orchestrate the entire workflow
- **Memory**: (Next section) Maintains context across interactions

## 🔍 Common Patterns

### 1. Simple Generation
```python
template = PromptTemplate(template=\"Explain {topic}\")
llm = OpenAI()
chain = LLMChain(llm=llm, prompt=template)
result = chain.run(topic=\"machine learning\")
```

### 2. Structured Output
```python
parser = PydanticOutputParser(pydantic_object=MyModel)
template = PromptTemplate(
    template=\"Analyze: {text}\n{format_instructions}\",
    partial_variables={\"format_instructions\": parser.get_format_instructions()}
)
chain = LLMChain(llm=llm, prompt=template, output_parser=parser)
```

### 3. Multi-Step Workflow
```python
step1 = LLMChain(llm=llm, prompt=prompt1, output_key=\"analysis\")
step2 = LLMChain(llm=llm, prompt=prompt2, output_key=\"summary\")
workflow = SequentialChain(chains=[step1, step2], input_variables=[\"text\"])
```

## 🛠️ Troubleshooting

### Common Issues

1. **\"Module not found\" errors**:
   ```bash
   # Make sure you're in the right directory
   cd Langchain-Playbook
   python basics/01_getting_started/hello_langchain.py
   ```

2. **API key errors**:
   ```bash
   # Check your .env file
   cat .env  # Linux/Mac
   type .env  # Windows
   ```

3. **Import errors**:
   ```bash
   # Reinstall requirements
   pip install -r requirements.txt --force-reinstall
   ```

4. **Permission errors**:
   ```bash
   # Ensure virtual environment is activated
   .venv\\Scripts\\activate
   ```

### Getting Help
- Check individual module README files
- Review error messages carefully
- Validate setup with `environment_setup.py`
- Refer to [troubleshooting guide](../docs/troubleshooting.md)

## 📈 Progress Tracking

Track your learning progress:

- [ ] 01_getting_started: Hello world and chat basics
- [ ] 02_models: Model comparison and selection
- [ ] 03_prompts: Template creation and engineering
- [ ] 04_chains: Workflow building and composition
- [ ] 05_output_parsers: Structured data handling

## 🎯 What's Next?

After completing the basics:

### Immediate Next Steps
1. **[Intermediate](../intermediate/)**: Memory, agents, RAG
2. **[Projects](../projects/)**: Real-world applications
3. **[Notebooks](../notebooks/)**: Interactive tutorials

### Recommended Learning Path
```
Basics → Intermediate → Advanced → Projects
   ↓
Foundation → Features → Optimization → Applications
```

### Choose Your Adventure
- **📚 Continue Learning**: Go to `../intermediate/01_memory/`
- **🎯 Build Projects**: Jump to `../projects/chatbot/`
- **📓 Interactive**: Try `../notebooks/tutorials/`
- **🧪 Experiment**: Explore `../notebooks/exploration/`

## 💬 Tips for Success

1. **Start Simple**: Don't skip the basics
2. **Practice**: Run and modify every example
3. **Experiment**: Try different parameters and prompts
4. **Read Logs**: Pay attention to debug output
5. **Ask Questions**: Use comments to understand each step
6. **Build Gradually**: Master one concept before moving on

---

**🎉 Ready to master LangChain basics? Start with [01_getting_started](01_getting_started/) and build your foundation! 🚀**