# Getting Started with LangChain

Welcome to your LangChain journey! This section contains the most basic examples to get you up and running.

## Files in this Section

### 1. `hello_langchain.py`
**Your first LangChain application**

- Basic LLM instantiation
- Simple prompting
- Batch processing
- Error handling

**Run it:**
```bash
python hello_langchain.py
```

### 2. `basic_chat.py` (Coming next)
**Introduction to chat models**

- Chat vs completion models
- Message formatting
- Conversation flow

### 3. `environment_setup.py` (Coming next)
**Validating your setup**

- API key validation
- Model availability testing
- Configuration verification

## What You'll Learn

1. **LLM Basics**: Understanding language models in LangChain
2. **API Integration**: Connecting to different providers
3. **Basic Prompting**: Sending queries and getting responses
4. **Error Handling**: Dealing with common issues
5. **Batch Processing**: Handling multiple requests efficiently

## Prerequisites

- Python 3.8+
- OpenAI API key (see [API Keys Guide](../../docs/api-keys.md))
- Installed dependencies (`pip install -r requirements.txt`)

## Common Issues

1. **\"No module named 'utils'\"**: Make sure you're running from the correct directory
2. **\"API key not found\"**: Check your `.env` file has `OPENAI_API_KEY=your_key`
3. **Connection errors**: Verify internet connection and API key validity

## Next Steps

Once you've run the hello world example:

1. Try modifying the temperature and max_tokens
2. Experiment with different prompts
3. Move on to `../02_models/` for more advanced model usage

## Tips

- Start with small examples and build up
- Read error messages carefully
- Check the logs for detailed information
- Experiment with different parameters

---

**Ready to begin? Run `python hello_langchain.py` and see the magic! ✨**