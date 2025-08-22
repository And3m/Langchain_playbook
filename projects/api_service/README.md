# LangChain API Service 🚀

A production-ready RESTful API service built with FastAPI that exposes LangChain functionality through HTTP endpoints. Perfect for integrating LangChain capabilities into web applications, mobile apps, or other services.

## Features

### 🔧 Core API Endpoints
- **Chat Completion**: Conversational AI with memory
- **Text Completion**: Single-turn text generation
- **Document Processing**: Text analysis, summarization, and extraction
- **Streaming Responses**: Real-time token streaming
- **Model Management**: Multiple LLM provider support

### 🔐 Production Features
- **Authentication**: API key-based security
- **Rate Limiting**: Request throttling and abuse prevention
- **CORS Support**: Cross-origin resource sharing
- **Error Handling**: Comprehensive error responses
- **Logging**: Request/response monitoring
- **Health Checks**: Service status monitoring

### 💬 Conversation Management
- **Memory Persistence**: Conversation context storage
- **Session Management**: Multi-user conversation handling
- **History Retrieval**: Access to conversation logs
- **Context Clearing**: Memory management utilities

## Quick Start

### Installation

```bash
# Install dependencies
pip install fastapi uvicorn langchain openai

# Or install from requirements.txt
pip install -r requirements.txt
```

### Environment Setup

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your_openai_api_key_here

# Or create .env file
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Start the Server

```bash
# Run the development server
python api_service.py run

# Or use uvicorn directly
uvicorn api_service:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API Base**: http://127.0.0.1:8000
- **Documentation**: http://127.0.0.1:8000/docs
- **Health Check**: http://127.0.0.1:8000/health

## API Documentation

### Authentication

All endpoints require an API key in the Authorization header:

```bash
curl -H "Authorization: Bearer demo-api-key" \
     http://127.0.0.1:8000/health
```

### Chat Completion

Start a conversation or continue an existing one:

```bash
curl -X POST "http://127.0.0.1:8000/chat" \
     -H "Authorization: Bearer demo-api-key" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "Hello, how are you?",
       "model": "gpt-3.5-turbo",
       "temperature": 0.7,
       "max_tokens": 500
     }'
```

**Response:**
```json
{
  "response": "Hello! I'm doing well, thank you for asking...",
  "conversation_id": "uuid-string",
  "model_used": "gpt-3.5-turbo",
  "timestamp": "2024-01-15T10:30:00",
  "tokens_used": 45
}
```

### Text Completion

Generate text completion for a prompt:

```bash
curl -X POST "http://127.0.0.1:8000/completion" \
     -H "Authorization: Bearer demo-api-key" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "The future of artificial intelligence is",
       "model": "gpt-3.5-turbo",
       "temperature": 0.8,
       "max_tokens": 200
     }'
```

### Document Processing

Process documents with various operations:

```bash
curl -X POST "http://127.0.0.1:8000/document" \
     -H "Authorization: Bearer demo-api-key" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Your document text here...",
       "operation": "summarize",
       "chunk_size": 1000,
       "model": "gpt-3.5-turbo"
     }'
```

**Supported Operations:**
- `summarize`: Create document summary
- `analyze`: Extract key insights
- `extract_keywords`: Identify key terms

### Streaming Responses

Get real-time streaming responses:

```bash
curl -X POST "http://127.0.0.1:8000/stream" \
     -H "Authorization: Bearer demo-api-key" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "Tell me a story",
       "model": "gpt-3.5-turbo"
     }'
```

### Conversation Management

```bash
# Get conversation history
curl -H "Authorization: Bearer demo-api-key" \
     "http://127.0.0.1:8000/conversations/{conversation_id}"

# Clear conversation
curl -X DELETE -H "Authorization: Bearer demo-api-key" \
     "http://127.0.0.1:8000/conversations/{conversation_id}"

# List available models
curl -H "Authorization: Bearer demo-api-key" \
     "http://127.0.0.1:8000/models"
```

## API Reference

### Request Models

#### ChatRequest
```python
{
  "message": "string",           # Required: User message
  "conversation_id": "string",   # Optional: Conversation ID
  "model": "string",            # Optional: Model name
  "temperature": 0.7,           # Optional: Creativity (0-1)
  "max_tokens": 500,            # Optional: Max response length
  "stream": false               # Optional: Enable streaming
}
```

#### CompletionRequest
```python
{
  "prompt": "string",           # Required: Completion prompt
  "model": "string",            # Optional: Model name
  "temperature": 0.7,           # Optional: Creativity
  "max_tokens": 500             # Optional: Max tokens
}
```

#### DocumentRequest
```python
{
  "text": "string",             # Required: Document text
  "operation": "string",        # Required: Operation type
  "chunk_size": 1000,          # Optional: Text chunk size
  "model": "string"            # Optional: Model name
}
```

### Response Models

#### ChatResponse
```python
{
  "response": "string",         # AI response
  "conversation_id": "string",  # Conversation ID
  "model_used": "string",       # Model used
  "timestamp": "string",        # Response timestamp
  "tokens_used": 123            # Tokens consumed
}
```

### Error Responses

```python
{
  "error": "string",            # Error description
  "status_code": 400,           # HTTP status code
  "timestamp": "string"         # Error timestamp
}
```

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your_openai_api_key

# Optional
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
LOG_LEVEL=INFO
```

### Rate Limiting

Default configuration:
- **Requests**: 100 per hour per IP
- **Window**: 3600 seconds (1 hour)
- **Method**: Token bucket algorithm

Customize in the `RateLimiter` class:

```python
rate_limiter = RateLimiter()
rate_limiter.max_requests = 200  # Increase limit
rate_limiter.window = 1800       # 30 minutes
```

### Model Configuration

Supported models:
- `gpt-3.5-turbo` (default)
- `gpt-4`
- `gpt-4-turbo`

Add new models in `LangChainService._initialize_models()`:

```python
self.models['custom-model'] = ChatOpenAI(
    openai_api_key=api_key,
    model_name='custom-model',
    temperature=0.7
)
```

## Architecture

### Components

#### FastAPI Application
- Route definition and handling
- Request/response validation
- Middleware configuration
- Error handling

#### LangChainService
- Core LangChain operations
- Model management
- Request processing
- Response formatting

#### Authentication & Security
- API key validation
- Rate limiting
- CORS configuration
- Request sanitization

#### Memory Management
- Conversation storage
- Session handling
- Memory cleanup
- Context persistence

### Request Flow

1. **Authentication**: Verify API key
2. **Rate Limiting**: Check request limits
3. **Validation**: Validate request data
4. **Processing**: Execute LangChain operation
5. **Response**: Format and return result
6. **Logging**: Log request/response

## Deployment

### Development

```bash
# Start with auto-reload
python api_service.py run

# Or with uvicorn
uvicorn api_service:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
# Production server with multiple workers
gunicorn api_service:app -w 4 -k uvicorn.workers.UvicornWorker \
         --bind 0.0.0.0:8000 --timeout 120

# Or with uvicorn
uvicorn api_service:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "api_service:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t langchain-api .
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key langchain-api
```

### Cloud Deployment

#### Heroku
```bash
# Install Heroku CLI and login
heroku create your-app-name
heroku config:set OPENAI_API_KEY=your_key
git push heroku main
```

#### AWS Lambda
Use **Mangum** for serverless deployment:

```python
from mangum import Mangum
handler = Mangum(app)
```

## Performance Optimization

### Caching
- Implement Redis for conversation storage
- Cache model responses for common queries
- Use connection pooling for databases

### Scaling
- Use multiple workers with load balancing
- Implement horizontal scaling with orchestration
- Monitor memory usage and optimize accordingly

### Monitoring
- Add metrics collection (Prometheus)
- Implement health checks
- Set up alerting for failures

## Security Best Practices

### API Security
- Use strong API keys (JWT tokens recommended)
- Implement request signing
- Add IP whitelisting for sensitive environments
- Use HTTPS in production

### Data Protection
- Sanitize all input data
- Log requests without sensitive information
- Implement data retention policies
- Use encryption for stored conversations

### Infrastructure Security
- Use firewalls and security groups
- Implement DDoS protection
- Regular security updates
- Vulnerability scanning

## Integration Examples

### Python Client

```python
import requests

class LangChainClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def chat(self, message, conversation_id=None):
        response = requests.post(
            f"{self.base_url}/chat",
            headers=self.headers,
            json={"message": message, "conversation_id": conversation_id}
        )
        return response.json()

# Usage
client = LangChainClient("http://127.0.0.1:8000", "demo-api-key")
result = client.chat("Hello!")
print(result["response"])
```

### JavaScript Client

```javascript
class LangChainAPI {
  constructor(baseUrl, apiKey) {
    this.baseUrl = baseUrl;
    this.apiKey = apiKey;
  }

  async chat(message, conversationId = null) {
    const response = await fetch(`${this.baseUrl}/chat`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        message,
        conversation_id: conversationId
      })
    });
    
    return await response.json();
  }
}

// Usage
const api = new LangChainAPI('http://127.0.0.1:8000', 'demo-api-key');
api.chat('Hello!').then(result => console.log(result.response));
```

## Troubleshooting

### Common Issues

1. **"OpenAI API key not found"**
   - Check environment variables
   - Verify .env file configuration
   - Ensure API key is valid

2. **Rate limit exceeded**
   - Implement exponential backoff
   - Check rate limiting configuration
   - Use multiple API keys for scaling

3. **Memory issues with long conversations**
   - Implement conversation cleanup
   - Use summary memory for long chats
   - Set conversation TTL

4. **Slow response times**
   - Optimize model parameters
   - Implement caching
   - Use faster models for simple tasks

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Monitoring

Check service health:

```bash
curl http://127.0.0.1:8000/health
```

## Contributing

Contributions welcome! Areas for improvement:
- Additional LLM provider support
- Enhanced authentication methods
- Advanced caching strategies
- Performance optimizations
- Documentation improvements

---

**Build Powerful LangChain APIs! 🌟**