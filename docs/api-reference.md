# API Documentation 📚

Comprehensive reference documentation for the LangChain Playbook codebase, utilities, and project APIs.

## 📋 Table of Contents

- [Configuration API](#configuration-api)
- [Utility Functions](#utility-functions)
- [Project APIs](#project-apis)
- [Common Patterns](#common-patterns)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

## Configuration API

### `utils.config`

#### `get_api_key(provider: str) -> Optional[str]`
Retrieves API keys for different LLM providers.

**Parameters:**
- `provider` (str): Provider name ('openai', 'anthropic', 'google')

**Returns:**
- `Optional[str]`: API key if found, None otherwise

**Example:**
```python
from utils.config import get_api_key

# Get OpenAI API key
api_key = get_api_key('openai')
if api_key:
    print("API key loaded successfully")
```

**Supported Providers:**
- `openai`: OpenAI GPT models
- `anthropic`: Claude models  
- `google`: Gemini models
- `huggingface`: Hugging Face models

#### `validate_api_key(provider: str, api_key: str) -> bool`
Validates API key format and accessibility.

**Parameters:**
- `provider` (str): Provider name
- `api_key` (str): API key to validate

**Returns:**
- `bool`: True if valid, False otherwise

---

## Utility Functions

### `utils.logging`

#### `setup_logging(level: str = "INFO") -> None`
Configures application-wide logging.

**Parameters:**
- `level` (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')

#### `get_logger(name: str) -> logging.Logger`
Creates a configured logger instance.

**Parameters:**
- `name` (str): Logger name (usually `__name__`)

**Returns:**
- `logging.Logger`: Configured logger

**Example:**
```python
from utils.logging import setup_logging, get_logger

setup_logging("DEBUG")
logger = get_logger(__name__)
logger.info("Application started")
```

### `utils.decorators`

#### `@timing_decorator`
Measures and logs function execution time.

**Example:**
```python
from utils.decorators import timing_decorator

@timing_decorator
def slow_function():
    # Function implementation
    pass
```

#### `@retry_decorator(max_attempts: int = 3, delay: float = 1.0)`
Retries function on failure with exponential backoff.

**Parameters:**
- `max_attempts` (int): Maximum retry attempts
- `delay` (float): Initial delay between retries

---

## Project APIs

### Chatbot API

#### `class PersonalityChatbot`

**Constructor:**
```python
PersonalityChatbot(
    api_key: str,
    personality: str = "helpful",
    model: str = "gpt-3.5-turbo"
)
```

**Methods:**

##### `chat(message: str, user_id: str = "default") -> Dict[str, Any]`
Process a chat message and return response.

**Parameters:**
- `message` (str): User message
- `user_id` (str): Unique user identifier

**Returns:**
```python
{
    "response": "AI response text",
    "personality": "current_personality",
    "tools_used": ["tool1", "tool2"],
    "conversation_id": "uuid",
    "timestamp": "ISO timestamp"
}
```

##### `change_personality(personality: str) -> bool`
Change the chatbot's personality.

**Available Personalities:**
- `helpful`: Supportive and informative
- `creative`: Imaginative and artistic
- `analytical`: Logical and detail-oriented
- `casual`: Friendly and relaxed

### Document Q&A API

#### `class DocumentQASystem`

**Constructor:**
```python
DocumentQASystem(
    api_key: str,
    vector_store_type: str = "faiss",
    chunk_size: int = 1000
)
```

**Methods:**

##### `add_document(file_path: str, metadata: Dict = None) -> str`
Add a document to the knowledge base.

**Parameters:**
- `file_path` (str): Path to document file
- `metadata` (Dict): Optional document metadata

**Returns:**
- `str`: Document ID

##### `ask_question(question: str) -> Dict[str, Any]`
Ask a question about the documents.

**Returns:**
```python
{
    "answer": "Generated answer",
    "sources": [
        {
            "content": "Source text",
            "metadata": {"page": 1, "source": "file.pdf"},
            "relevance_score": 0.95
        }
    ],
    "confidence": 0.87
}
```

### Code Assistant API

#### `class CodeAssistant`

**Methods:**

##### `generate_code(task: str, language: str = "python") -> Dict[str, Any]`
Generate code from natural language description.

**Returns:**
```python
{
    "status": "success",
    "code": "Generated code",
    "language": "python",
    "explanation": "Code explanation"
}
```

##### `explain_code(code: str, language: str = "python") -> Dict[str, Any]`
Explain what code does.

##### `review_code(code: str, language: str = "python") -> Dict[str, Any]`
Review code for quality and best practices.

---

## Common Patterns

### Error Handling Pattern
```python
from utils import get_logger

logger = get_logger(__name__)

def safe_operation():
    try:
        # Operation that might fail
        result = risky_function()
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        return {"status": "error", "error": str(e)}
```

### API Response Pattern
```python
def api_response(data=None, error=None):
    """Standard API response format"""
    return {
        "status": "success" if error is None else "error",
        "data": data,
        "error": error,
        "timestamp": datetime.now().isoformat()
    }
```

### LLM Chain Pattern
```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def create_chain(llm, template_text, input_vars):
    """Create a reusable LLM chain"""
    prompt = PromptTemplate(
        input_variables=input_vars,
        template=template_text
    )
    return LLMChain(llm=llm, prompt=prompt)
```

---

## Error Handling

### Common Exceptions

#### `APIKeyError`
Raised when API key is missing or invalid.

```python
try:
    api_key = get_api_key('openai')
    if not api_key:
        raise APIKeyError("OpenAI API key not found")
except APIKeyError as e:
    logger.error(f"Configuration error: {e}")
```

#### `ModelNotAvailableError`
Raised when requested model is not available.

#### `RateLimitError`
Raised when API rate limits are exceeded.

### Error Response Format
```python
{
    "status": "error",
    "error_code": "API_KEY_MISSING",
    "error_message": "OpenAI API key not found",
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "uuid"
}
```

---

## Best Practices

### 1. Configuration Management
```python
# Good: Use configuration utilities
from utils.config import get_api_key
api_key = get_api_key('openai')

# Avoid: Hard-coding credentials
api_key = "sk-..."  # Never do this
```

### 2. Logging
```python
# Good: Structured logging
logger.info("Processing request", extra={
    "user_id": user_id,
    "request_type": "chat",
    "model": "gpt-3.5-turbo"
})

# Avoid: Print statements
print("User sent message")  # Use logger instead
```

### 3. Error Handling
```python
# Good: Specific exception handling
try:
    result = llm.predict(prompt)
except RateLimitError:
    return {"error": "Rate limit exceeded, please try again"}
except APIKeyError:
    return {"error": "Invalid API configuration"}
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    return {"error": "Internal server error"}
```

### 4. Resource Management
```python
# Good: Context managers for resources
with open(file_path, 'r') as f:
    content = f.read()

# Good: Cleanup after operations
try:
    vectorstore = create_vectorstore(documents)
    # Use vectorstore
finally:
    vectorstore.cleanup()  # If applicable
```

### 5. API Design
```python
# Good: Consistent response format
def process_request(data):
    try:
        result = perform_operation(data)
        return api_response(data=result)
    except Exception as e:
        return api_response(error=str(e))

# Good: Input validation
def validate_input(data):
    if not isinstance(data, dict):
        raise ValueError("Input must be a dictionary")
    if 'message' not in data:
        raise ValueError("Message field is required")
```

---

## Integration Examples

### Basic LangChain Integration
```python
from utils import get_api_key, get_logger
from langchain.chat_models import ChatOpenAI

logger = get_logger(__name__)

def create_llm():
    api_key = get_api_key('openai')
    if not api_key:
        raise APIKeyError("OpenAI API key required")
    
    return ChatOpenAI(
        openai_api_key=api_key,
        model_name="gpt-3.5-turbo",
        temperature=0.7
    )
```

### Chain with Error Handling
```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def safe_chain_run(chain, **kwargs):
    try:
        result = chain.run(**kwargs)
        logger.info("Chain executed successfully")
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Chain execution failed: {e}")
        return {"status": "error", "error": str(e)}
```

### Memory Management
```python
from langchain.memory import ConversationBufferMemory

class ConversationManager:
    def __init__(self):
        self.conversations = {}
    
    def get_memory(self, user_id: str) -> ConversationBufferMemory:
        if user_id not in self.conversations:
            self.conversations[user_id] = ConversationBufferMemory()
        return self.conversations[user_id]
    
    def clear_memory(self, user_id: str) -> None:
        if user_id in self.conversations:
            del self.conversations[user_id]
```

---

**For More Information:**
- Check individual project READMEs
- Explore example implementations
- Review test files for usage patterns
- Join community discussions for Q&A