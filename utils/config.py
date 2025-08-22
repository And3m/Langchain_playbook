"""
Configuration management utilities for LangChain Playbook.
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    return {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),
        'google_api_key': os.getenv('GOOGLE_API_KEY'),
        'huggingface_api_token': os.getenv('HUGGINGFACE_API_TOKEN'),
        'pinecone_api_key': os.getenv('PINECONE_API_KEY'),
        'pinecone_environment': os.getenv('PINECONE_ENVIRONMENT'),
        'cohere_api_key': os.getenv('COHERE_API_KEY'),
        'weaviate_url': os.getenv('WEAVIATE_URL'),
        'weaviate_api_key': os.getenv('WEAVIATE_API_KEY'),
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'debug': os.getenv('DEBUG', 'False').lower() == 'true',
        'default_model': os.getenv('DEFAULT_MODEL', 'gpt-3.5-turbo'),
        'default_temperature': float(os.getenv('DEFAULT_TEMPERATURE', '0.7')),
        'default_max_tokens': int(os.getenv('DEFAULT_MAX_TOKENS', '1000'))
    }


def get_api_key(provider: str) -> Optional[str]:
    """Get API key for a specific provider.
    
    Args:
        provider: The provider name (openai, anthropic, google, etc.)
        
    Returns:
        The API key if found, None otherwise
    """
    provider = provider.lower()
    key_mapping = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'google': 'GOOGLE_API_KEY',
        'huggingface': 'HUGGINGFACE_API_TOKEN',
        'pinecone': 'PINECONE_API_KEY',
        'cohere': 'COHERE_API_KEY',
        'weaviate': 'WEAVIATE_API_KEY'
    }
    
    if provider not in key_mapping:
        available = ', '.join(key_mapping.keys())
        raise ValueError(f"Unknown provider '{provider}'. Available: {available}")
    
    return os.getenv(key_mapping[provider])


def validate_api_keys(required_providers: list) -> Dict[str, bool]:
    """Validate that required API keys are present.
    
    Args:
        required_providers: List of provider names to check
        
    Returns:
        Dictionary mapping provider names to whether their API key is present
    """
    results = {}
    for provider in required_providers:
        try:
            api_key = get_api_key(provider)
            results[provider] = api_key is not None and api_key.strip() != ''
        except ValueError:
            results[provider] = False
    
    return results


def get_model_config(model_name: Optional[str] = None) -> Dict[str, Any]:
    """Get model configuration.
    
    Args:
        model_name: Specific model name, or None to use default
        
    Returns:
        Model configuration dictionary
    """
    config = load_config()
    
    return {
        'model_name': model_name or config['default_model'],
        'temperature': config['default_temperature'],
        'max_tokens': config['default_max_tokens']
    }