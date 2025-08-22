#!/usr/bin/env python3
"""
LangChain API Service - RESTful API for LangChain Applications

This project demonstrates:
1. FastAPI integration with LangChain
2. RESTful endpoints for LLM operations
3. Authentication and rate limiting
4. Async request handling
5. Model management and caching
6. Error handling and logging

Key features:
- Multiple LLM provider support
- Conversation management
- Document processing endpoints
- Real-time streaming responses
- API key management
- Request/response logging
"""

import sys
import asyncio
import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

# Add utils to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import setup_logging, get_logger, get_api_key

# FastAPI and related imports
try:
    from fastapi import FastAPI, HTTPException, Depends, status, Request, BackgroundTasks
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    print("⚠️ FastAPI not installed. Run: pip install fastapi uvicorn")
    sys.exit(1)

# LangChain imports
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# Pydantic models for API requests/responses
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(default=None, description="Message timestamp")


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID for context")
    model: Optional[str] = Field(default="gpt-3.5-turbo", description="Model to use")
    temperature: Optional[float] = Field(default=0.7, description="Response creativity (0-1)")
    max_tokens: Optional[int] = Field(default=500, description="Maximum response length")
    stream: Optional[bool] = Field(default=False, description="Enable streaming response")


class ChatResponse(BaseModel):
    response: str = Field(..., description="AI response")
    conversation_id: str = Field(..., description="Conversation ID")
    model_used: str = Field(..., description="Model that generated the response")
    timestamp: str = Field(..., description="Response timestamp")
    tokens_used: Optional[int] = Field(default=None, description="Tokens consumed")


class CompletionRequest(BaseModel):
    prompt: str = Field(..., description="Prompt for completion")
    model: Optional[str] = Field(default="gpt-3.5-turbo", description="Model to use")
    temperature: Optional[float] = Field(default=0.7, description="Response creativity")
    max_tokens: Optional[int] = Field(default=500, description="Maximum response length")


class CompletionResponse(BaseModel):
    completion: str = Field(..., description="Generated completion")
    model_used: str = Field(..., description="Model used")
    timestamp: str = Field(..., description="Response timestamp")
    tokens_used: Optional[int] = Field(default=None, description="Tokens consumed")


class DocumentRequest(BaseModel):
    text: str = Field(..., description="Document text to process")
    operation: str = Field(..., description="Operation: 'summarize', 'analyze', 'extract_keywords'")
    chunk_size: Optional[int] = Field(default=1000, description="Text chunk size")
    model: Optional[str] = Field(default="gpt-3.5-turbo", description="Model to use")


class DocumentResponse(BaseModel):
    result: str = Field(..., description="Processing result")
    operation: str = Field(..., description="Operation performed")
    chunks_processed: int = Field(..., description="Number of text chunks processed")
    model_used: str = Field(..., description="Model used")
    timestamp: str = Field(..., description="Response timestamp")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    uptime: str = Field(..., description="Service uptime")
    models_available: List[str] = Field(..., description="Available models")


# Rate limiting and authentication
class RateLimiter:
    def __init__(self):
        self.requests = {}
        self.max_requests = 100  # per hour
        self.window = 3600  # 1 hour in seconds
    
    def is_allowed(self, client_id: str) -> bool:
        now = datetime.now()
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Remove old requests outside the window
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < timedelta(seconds=self.window)
        ]
        
        # Check if under limit
        if len(self.requests[client_id]) < self.max_requests:
            self.requests[client_id].append(now)
            return True
        
        return False


# Global instances
rate_limiter = RateLimiter()
conversations = {}  # In-memory conversation storage
security = HTTPBearer()


class LangChainService:
    """Service class for LangChain operations."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.models = {}
        self.start_time = datetime.now()
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available models."""
        api_key = get_api_key('openai')
        if api_key:
            self.models['gpt-3.5-turbo'] = ChatOpenAI(
                openai_api_key=api_key,
                model_name='gpt-3.5-turbo',
                temperature=0.7
            )
            self.models['gpt-4'] = ChatOpenAI(
                openai_api_key=api_key,
                model_name='gpt-4',
                temperature=0.7
            )
            self.logger.info("Models initialized successfully")
        else:
            self.logger.warning("No OpenAI API key found")
    
    def get_model(self, model_name: str, **kwargs):
        """Get model instance with custom parameters."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        # Create new instance with custom parameters
        api_key = get_api_key('openai')
        return ChatOpenAI(
            openai_api_key=api_key,
            model_name=model_name,
            **kwargs
        )
    
    async def chat_completion(self, request: ChatRequest) -> ChatResponse:
        """Handle chat completion request."""
        try:
            model = self.get_model(
                request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            # Get or create conversation
            conv_id = request.conversation_id or str(uuid.uuid4())
            
            if conv_id not in conversations:
                conversations[conv_id] = ConversationBufferMemory()
            
            memory = conversations[conv_id]
            
            # Create conversation chain
            conversation = ConversationChain(
                llm=model,
                memory=memory
            )
            
            # Get response
            response = await asyncio.get_event_loop().run_in_executor(
                None, conversation.predict, request.message
            )
            
            return ChatResponse(
                response=response,
                conversation_id=conv_id,
                model_used=request.model,
                timestamp=datetime.now().isoformat(),
                tokens_used=None  # Would need to extract from response
            )
            
        except Exception as e:
            self.logger.error(f"Chat completion failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def text_completion(self, request: CompletionRequest) -> CompletionResponse:
        """Handle text completion request."""
        try:
            model = self.get_model(
                request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            # Simple completion
            response = await asyncio.get_event_loop().run_in_executor(
                None, model.predict, request.prompt
            )
            
            return CompletionResponse(
                completion=response,
                model_used=request.model,
                timestamp=datetime.now().isoformat(),
                tokens_used=None
            )
            
        except Exception as e:
            self.logger.error(f"Text completion failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def process_document(self, request: DocumentRequest) -> DocumentResponse:
        """Process document with specified operation."""
        try:
            model = self.get_model(request.model)
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=request.chunk_size,
                chunk_overlap=100
            )
            chunks = text_splitter.split_text(request.text)
            
            # Create operation-specific prompt
            if request.operation == "summarize":
                prompt_template = "Summarize the following text:\n\n{text}"
            elif request.operation == "analyze":
                prompt_template = "Analyze the following text and provide key insights:\n\n{text}"
            elif request.operation == "extract_keywords":
                prompt_template = "Extract key terms and concepts from the following text:\n\n{text}"
            else:
                raise ValueError(f"Unknown operation: {request.operation}")
            
            # Process chunks
            results = []
            for chunk in chunks:
                prompt = prompt_template.format(text=chunk)
                result = await asyncio.get_event_loop().run_in_executor(
                    None, model.predict, prompt
                )
                results.append(result)
            
            # Combine results
            if request.operation == "summarize":
                final_result = "\n\n".join(results)
            else:
                final_result = "\n".join(results)
            
            return DocumentResponse(
                result=final_result,
                operation=request.operation,
                chunks_processed=len(chunks),
                model_used=request.model,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"Document processing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def get_health_status(self) -> HealthResponse:
        """Get service health status."""
        uptime = datetime.now() - self.start_time
        uptime_str = str(uptime).split('.')[0]  # Remove microseconds
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            uptime=uptime_str,
            models_available=list(self.models.keys())
        )


# Initialize service
langchain_service = LangChainService()


# Dependency functions
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key (simplified)."""
    # In production, implement proper API key validation
    if credentials.credentials != "demo-api-key":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return credentials.credentials


async def check_rate_limit(request: Request):
    """Check rate limiting."""
    client_ip = request.client.host
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    setup_logging()
    logger = get_logger("API-Service")
    logger.info("🚀 LangChain API Service starting up")
    yield
    logger.info("🛑 LangChain API Service shutting down")


# Create FastAPI app
app = FastAPI(
    title="LangChain API Service",
    description="RESTful API for LangChain operations",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return langchain_service.get_health_status()


@app.post("/chat", response_model=ChatResponse)
async def chat_completion(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
    rate_check: None = Depends(check_rate_limit)
):
    """Chat completion endpoint with conversation memory."""
    
    # Log request
    background_tasks.add_task(
        log_request, "chat", request.dict(), api_key
    )
    
    return await langchain_service.chat_completion(request)


@app.post("/completion", response_model=CompletionResponse)
async def text_completion(
    request: CompletionRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
    rate_check: None = Depends(check_rate_limit)
):
    """Text completion endpoint."""
    
    # Log request
    background_tasks.add_task(
        log_request, "completion", request.dict(), api_key
    )
    
    return await langchain_service.text_completion(request)


@app.post("/document", response_model=DocumentResponse)
async def process_document(
    request: DocumentRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
    rate_check: None = Depends(check_rate_limit)
):
    """Document processing endpoint."""
    
    # Log request
    background_tasks.add_task(
        log_request, "document", request.dict(), api_key
    )
    
    return await langchain_service.process_document(request)


@app.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get conversation history."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    memory = conversations[conversation_id]
    history = memory.load_memory_variables({})
    
    return {
        "conversation_id": conversation_id,
        "history": history,
        "message_count": len(memory.chat_memory.messages)
    }


@app.delete("/conversations/{conversation_id}")
async def clear_conversation(
    conversation_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Clear conversation history."""
    if conversation_id in conversations:
        conversations[conversation_id].clear()
        return {"message": "Conversation cleared"}
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")


@app.get("/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    """List available models."""
    return {
        "models": list(langchain_service.models.keys()),
        "default": "gpt-3.5-turbo"
    }


# Background task functions
def log_request(endpoint: str, request_data: dict, api_key: str):
    """Log API request for monitoring."""
    logger = get_logger("API-Logger")
    
    # Sanitize sensitive data
    sanitized_data = request_data.copy()
    if 'message' in sanitized_data:
        sanitized_data['message'] = sanitized_data['message'][:100] + "..." if len(sanitized_data['message']) > 100 else sanitized_data['message']
    
    logger.info(f"API Request - Endpoint: {endpoint}, Data: {sanitized_data}, API Key: {api_key[:8]}...")


# Streaming endpoint (advanced feature)
@app.post("/stream")
async def stream_completion(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key),
    rate_check: None = Depends(check_rate_limit)
):
    """Streaming completion endpoint."""
    
    async def generate_stream():
        try:
            model = langchain_service.get_model(
                request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                streaming=True
            )
            
            # Simple streaming simulation (in practice, use proper streaming callbacks)
            response = await asyncio.get_event_loop().run_in_executor(
                None, model.predict, request.message
            )
            
            # Simulate token-by-token streaming
            for i, char in enumerate(response):
                yield f"data: {json.dumps({'token': char, 'index': i})}\n\n"
                await asyncio.sleep(0.01)  # Simulate streaming delay
            
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache"}
    )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger = get_logger("API-Error")
    logger.error(f"HTTP {exc.status_code}: {exc.detail} - Path: {request.url.path}")
    
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger = get_logger("API-Error")
    logger.error(f"Unhandled exception: {exc} - Path: {request.url.path}")
    
    return {
        "error": "Internal server error",
        "status_code": 500,
        "timestamp": datetime.now().isoformat()
    }


# CLI runner
def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = True):
    """Run the API server."""
    print(f"🚀 Starting LangChain API Service on {host}:{port}")
    print(f"📖 API Documentation: http://{host}:{port}/docs")
    print(f"🔑 Use API key: demo-api-key")
    
    uvicorn.run(
        "api_service:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        run_server()
    else:
        print("LangChain API Service")
        print("Usage: python api_service.py run")
        print("This will start the API server on http://127.0.0.1:8000")
        print("API documentation will be available at http://127.0.0.1:8000/docs")