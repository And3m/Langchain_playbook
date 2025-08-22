# API Keys Setup Guide

This guide explains how to obtain and configure API keys for various LLM providers used in the LangChain Playbook.

## Overview

The playbook supports multiple LLM providers. You don't need all of them - just set up the ones you plan to use.

## Supported Providers

### OpenAI
**Models**: GPT-3.5 Turbo, GPT-4, GPT-4 Turbo, DALL-E, Whisper

1. Visit [OpenAI API](https://platform.openai.com/api-keys)
2. Sign up or log in
3. Click \"Create new secret key\"
4. Copy the key and add to `.env`:
   ```
   OPENAI_API_KEY=sk-...
   ```

**Pricing**: Pay-per-use, starts around $0.0015/1K tokens

### Anthropic
**Models**: Claude 3 (Haiku, Sonnet, Opus)

1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Sign up or log in
3. Go to \"API Keys\" section
4. Create a new API key
5. Add to `.env`:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```

**Pricing**: Pay-per-use, competitive rates

### Google AI
**Models**: Gemini Pro, Gemini Pro Vision

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with Google account
3. Click \"Create API Key\"
4. Add to `.env`:
   ```
   GOOGLE_API_KEY=AIza...
   ```

**Pricing**: Generous free tier, then pay-per-use

### Hugging Face
**Models**: Open source models, Inference API

1. Visit [Hugging Face](https://huggingface.co/settings/tokens)
2. Sign up or log in
3. Click \"New token\"
4. Choose \"Read\" access
5. Add to `.env`:
   ```
   HUGGINGFACE_API_TOKEN=hf_...
   ```

**Pricing**: Free tier available, paid for faster inference

## Vector Store Providers

### Pinecone
**Use case**: Vector database for embeddings

1. Visit [Pinecone](https://app.pinecone.io/)
2. Sign up for free account
3. Create a new index
4. Get API key and environment from dashboard
5. Add to `.env`:
   ```
   PINECONE_API_KEY=your-key
   PINECONE_ENVIRONMENT=your-env
   ```

### Weaviate
**Use case**: Vector database alternative

1. Visit [Weaviate Cloud](https://console.weaviate.cloud/)
2. Create a free cluster
3. Get cluster URL and API key
4. Add to `.env`:
   ```
   WEAVIATE_URL=https://your-cluster.weaviate.network
   WEAVIATE_API_KEY=your-key
   ```

## Configuration

### Environment File

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your keys:
   ```bash
   # Required for basic examples
   OPENAI_API_KEY=your_openai_key
   
   # Optional providers
   ANTHROPIC_API_KEY=your_anthropic_key
   GOOGLE_API_KEY=your_google_key
   
   # Vector stores (for RAG examples)
   PINECONE_API_KEY=your_pinecone_key
   PINECONE_ENVIRONMENT=your_pinecone_env
   ```

### Validation

Test your configuration:

```python
from utils import validate_api_keys

# Check which keys are available
result = validate_api_keys(['openai', 'anthropic', 'google'])
print(result)
```

## Cost Management

### Setting Limits

1. **OpenAI**: Set usage limits in your account dashboard
2. **Anthropic**: Monitor usage in the console
3. **Google**: Use quotas and billing alerts

### Cost Estimation

```python
from utils import estimate_cost

# Estimate cost for 1000 tokens with GPT-3.5
cost = estimate_cost(1000, \"gpt-3.5-turbo\")
print(f\"Estimated cost: ${cost:.4f}\")
```

## Security Best Practices

1. **Never commit API keys** to version control
2. **Use environment variables** or secure vaults
3. **Rotate keys regularly**
4. **Set usage limits** to prevent unexpected charges
5. **Monitor usage** regularly

## Free Tier Options

### Completely Free
- **Hugging Face**: Open source models
- **Ollama**: Run models locally
- **LM Studio**: Local model GUI

### Free Tiers
- **Google AI**: Generous free quota
- **OpenAI**: $5 free credit for new users
- **Anthropic**: Limited free usage

## Local Alternatives

### Ollama
Run models locally without API keys:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download a model
ollama pull llama2

# Use with LangChain
from langchain.llms import Ollama
llm = Ollama(model=\"llama2\")
```

### LM Studio
GUI for running local models:

1. Download [LM Studio](https://lmstudio.ai/)
2. Download a model
3. Start local server
4. Use OpenAI-compatible endpoint

## Troubleshooting

### Common Issues

1. **Invalid API key**: Double-check the key format
2. **Rate limits**: Add delays between requests
3. **Quota exceeded**: Check your usage dashboard
4. **Network errors**: Check internet connection

### Testing Connection

```python
from langchain.llms import OpenAI

try:
    llm = OpenAI()
    response = llm(\"Hello, world!\")
    print(\"Connection successful!\")
except Exception as e:
    print(f\"Connection failed: {e}\")
```

## Getting Help

- Check provider documentation
- Review error messages carefully
- Test with minimal examples first
- Check account status and limits