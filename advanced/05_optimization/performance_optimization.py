#!/usr/bin/env python3
"""
LangChain Performance & Cost Optimization

This module demonstrates:
1. Performance optimization techniques
2. Cost reduction strategies
3. Caching mechanisms
4. Model selection optimization
5. Prompt engineering for efficiency
6. Batch processing patterns

Key concepts:
- Response time optimization
- Token usage reduction
- Caching strategies
- Model comparison and selection
"""

import sys
import time
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps, lru_cache
import statistics

# Add utils to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import setup_logging, get_logger, get_api_key

# LangChain imports
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from langchain.schema import BaseOutputParser

# Optional dependencies for advanced optimization
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import aioredis
    ASYNC_REDIS_AVAILABLE = True
except ImportError:
    ASYNC_REDIS_AVAILABLE = False


@dataclass
class OptimizationMetrics:
    """Container for optimization metrics."""
    response_time: float
    token_usage: int
    cost_estimate: float
    cache_hit_rate: float
    throughput: float
    timestamp: str


class TokenOptimizer:
    """Optimize token usage for cost reduction."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        
        # Token pricing (approximate, per 1K tokens)
        self.pricing = {
            'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002},
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'gpt-4-turbo': {'input': 0.01, 'output': 0.03}
        }
    
    def optimize_prompt(self, prompt: str) -> str:
        """Optimize prompt to reduce token usage while maintaining effectiveness."""
        # Remove unnecessary whitespace
        optimized = ' '.join(prompt.split())
        
        # Replace verbose phrases with concise alternatives
        replacements = {
            'Please provide me with': 'Provide',
            'I would like to know': 'What is',
            'Can you tell me': 'What is',
            'I need information about': 'Explain',
            'Could you explain': 'Explain',
            'It would be helpful if': 'Please',
            'I am interested in learning about': 'Explain',
        }
        
        for verbose, concise in replacements.items():
            optimized = optimized.replace(verbose, concise)
        
        return optimized
    
    def create_efficient_prompt_template(self, task_description: str) -> PromptTemplate:
        """Create token-efficient prompt templates."""
        # Use concise, structured prompts
        if 'summarize' in task_description.lower():
            template = "Summarize in {word_limit} words:\n{text}"
            return PromptTemplate(input_variables=["word_limit", "text"], template=template)
        
        elif 'analyze' in task_description.lower():
            template = "Analyze {topic}:\n{content}\n\nKey points:"
            return PromptTemplate(input_variables=["topic", "content"], template=template)
        
        elif 'extract' in task_description.lower():
            template = "Extract {target} from:\n{text}\n\nResults:"
            return PromptTemplate(input_variables=["target", "text"], template=template)
        
        else:
            template = "{input}"
            return PromptTemplate(input_variables=["input"], template=template)
    
    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for token usage."""
        if model not in self.pricing:
            return 0.0
        
        input_cost = (input_tokens / 1000) * self.pricing[model]['input']
        output_cost = (output_tokens / 1000) * self.pricing[model]['output']
        
        return input_cost + output_cost
    
    def compare_model_costs(self, prompt: str, models: List[str]) -> Dict[str, Dict[str, float]]:
        """Compare costs across different models."""
        results = {}
        
        # Rough token estimation (4 chars = 1 token)
        estimated_input_tokens = len(prompt) // 4
        estimated_output_tokens = 100  # Assume 100 token response
        
        for model in models:
            cost = self.estimate_cost(model, estimated_input_tokens, estimated_output_tokens)
            results[model] = {
                'input_tokens': estimated_input_tokens,
                'output_tokens': estimated_output_tokens,
                'estimated_cost': cost
            }
        
        return results


class CacheManager:
    """Intelligent caching for LangChain responses."""
    
    def __init__(self, cache_type: str = "memory", redis_url: str = None):
        self.logger = get_logger(self.__class__.__name__)
        self.cache_type = cache_type
        self.memory_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize Redis if available and requested
        if cache_type == "redis" and REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                self.logger.info("Redis cache initialized")
            except Exception as e:
                self.logger.warning(f"Redis connection failed: {e}, falling back to memory cache")
                self.cache_type = "memory"
        else:
            self.redis_client = None
    
    def _generate_cache_key(self, prompt: str, model: str, temperature: float) -> str:
        """Generate cache key for prompt/model combination."""
        cache_data = f"{prompt}:{model}:{temperature}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    def get(self, prompt: str, model: str, temperature: float = 0.7) -> Optional[str]:
        """Get cached response if available."""
        cache_key = self._generate_cache_key(prompt, model, temperature)
        
        try:
            if self.cache_type == "redis" and self.redis_client:
                cached = self.redis_client.get(cache_key)
                if cached:
                    self.cache_hits += 1
                    return cached.decode('utf-8')
            else:
                if cache_key in self.memory_cache:
                    self.cache_hits += 1
                    return self.memory_cache[cache_key]
        except Exception as e:
            self.logger.error(f"Cache get error: {e}")
        
        self.cache_misses += 1
        return None
    
    def set(self, prompt: str, model: str, response: str, temperature: float = 0.7, ttl: int = 3600):
        """Cache response with TTL."""
        cache_key = self._generate_cache_key(prompt, model, temperature)
        
        try:
            if self.cache_type == "redis" and self.redis_client:
                self.redis_client.setex(cache_key, ttl, response)
            else:
                # Simple memory cache (no TTL implementation for demo)
                self.memory_cache[cache_key] = response
                
                # Limit memory cache size
                if len(self.memory_cache) > 1000:
                    # Remove oldest entries
                    keys_to_remove = list(self.memory_cache.keys())[:100]
                    for key in keys_to_remove:
                        del self.memory_cache[key]
        
        except Exception as e:
            self.logger.error(f"Cache set error: {e}")
    
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.cache_hits + self.cache_misses
        return self.cache_hits / total_requests if total_requests > 0 else 0.0
    
    def clear(self):
        """Clear all cached data."""
        if self.cache_type == "redis" and self.redis_client:
            self.redis_client.flushdb()
        else:
            self.memory_cache.clear()
        
        self.cache_hits = 0
        self.cache_misses = 0


class BatchProcessor:
    """Efficient batch processing for multiple requests."""
    
    def __init__(self, batch_size: int = 10, delay: float = 1.0):
        self.logger = get_logger(self.__class__.__name__)
        self.batch_size = batch_size
        self.delay = delay
    
    def process_batch_sync(self, prompts: List[str], llm_function: Callable[[str], str]) -> List[str]:
        """Process prompts in batches with rate limiting."""
        results = []
        
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            batch_results = []
            
            self.logger.info(f"Processing batch {i//self.batch_size + 1}/{len(prompts)//self.batch_size + 1}")
            
            for prompt in batch:
                try:
                    result = llm_function(prompt)
                    batch_results.append(result)
                except Exception as e:
                    self.logger.error(f"Batch processing error: {e}")
                    batch_results.append(f"Error: {str(e)}")
            
            results.extend(batch_results)
            
            # Rate limiting delay
            if i + self.batch_size < len(prompts):
                time.sleep(self.delay)
        
        return results
    
    async def process_batch_async(self, prompts: List[str], async_llm_function: Callable[[str], str]) -> List[str]:
        """Process prompts asynchronously with concurrency control."""
        semaphore = asyncio.Semaphore(self.batch_size)
        
        async def process_single(prompt: str) -> str:
            async with semaphore:
                try:
                    return await async_llm_function(prompt)
                except Exception as e:
                    self.logger.error(f"Async processing error: {e}")
                    return f"Error: {str(e)}"
        
        tasks = [process_single(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        
        return results


def performance_monitor(func):
    """Decorator to monitor function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Log performance metrics
        logger = get_logger("Performance-Monitor")
        logger.info(f"{func.__name__}: {execution_time:.3f}s, Success: {success}")
        
        if not success:
            logger.error(f"{func.__name__} error: {error}")
            raise Exception(error)
        
        return result
    
    return wrapper


class OptimizedLLMChain:
    """LLM Chain with built-in optimizations."""
    
    def __init__(self, model: str = "gpt-3.5-turbo", cache_manager: CacheManager = None):
        self.logger = get_logger(self.__class__.__name__)
        self.model = model
        self.cache_manager = cache_manager or CacheManager()
        self.token_optimizer = TokenOptimizer()
        
        # Initialize LLM
        api_key = get_api_key('openai')
        if api_key:
            self.llm = ChatOpenAI(openai_api_key=api_key, model_name=model, temperature=0.7)
        else:
            self.llm = None
            self.logger.warning("No API key available, using mock responses")
    
    @performance_monitor
    def invoke(self, prompt: str, use_cache: bool = True, optimize_prompt: bool = True) -> Dict[str, Any]:
        """Optimized LLM invocation with caching and monitoring."""
        # Optimize prompt if requested
        if optimize_prompt:
            original_prompt = prompt
            prompt = self.token_optimizer.optimize_prompt(prompt)
            token_savings = len(original_prompt) - len(prompt)
        else:
            token_savings = 0
        
        # Check cache first
        if use_cache:
            cached_response = self.cache_manager.get(prompt, self.model)
            if cached_response:
                return {
                    'response': cached_response,
                    'cached': True,
                    'token_savings': token_savings,
                    'cost_estimate': 0.0,
                    'response_time': 0.001  # Minimal cache retrieval time
                }
        
        # Generate response
        start_time = time.time()
        
        if self.llm:
            with get_openai_callback() as cb:
                response = self.llm.predict(prompt)
                
                # Extract token usage and cost
                token_usage = cb.total_tokens
                cost_estimate = cb.total_cost
        else:
            # Mock response for demo
            response = f"Mock response for: {prompt[:50]}..."
            token_usage = len(prompt) // 4  # Rough estimation
            cost_estimate = self.token_optimizer.estimate_cost(self.model, token_usage, 50)
        
        response_time = time.time() - start_time
        
        # Cache the response
        if use_cache:
            self.cache_manager.set(prompt, self.model, response)
        
        return {
            'response': response,
            'cached': False,
            'token_usage': token_usage,
            'token_savings': token_savings,
            'cost_estimate': cost_estimate,
            'response_time': response_time
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'cache_hit_rate': self.cache_manager.get_hit_rate(),
            'cache_hits': self.cache_manager.cache_hits,
            'cache_misses': self.cache_manager.cache_misses,
            'model': self.model
        }


class ModelSelector:
    """Intelligent model selection based on task requirements."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        
        # Model capabilities and costs
        self.model_profiles = {
            'gpt-3.5-turbo': {
                'cost_per_1k_tokens': 0.002,
                'speed': 'fast',
                'quality': 'good',
                'best_for': ['simple_qa', 'summarization', 'translation']
            },
            'gpt-4': {
                'cost_per_1k_tokens': 0.06,
                'speed': 'slow',
                'quality': 'excellent',
                'best_for': ['complex_reasoning', 'code_generation', 'analysis']
            },
            'gpt-4-turbo': {
                'cost_per_1k_tokens': 0.03,
                'speed': 'medium',
                'quality': 'excellent',
                'best_for': ['long_context', 'multimodal', 'complex_tasks']
            }
        }
    
    def select_optimal_model(self, task_type: str, budget_constraint: float = None, 
                           speed_requirement: str = None) -> str:
        """Select the most appropriate model for the task."""
        candidates = []
        
        for model, profile in self.model_profiles.items():
            score = 0
            
            # Task type matching
            if task_type in profile['best_for']:
                score += 3
            
            # Budget constraint
            if budget_constraint and profile['cost_per_1k_tokens'] <= budget_constraint:
                score += 2
            
            # Speed requirement
            if speed_requirement == profile['speed']:
                score += 2
            elif speed_requirement == 'fast' and profile['speed'] in ['fast', 'medium']:
                score += 1
            
            candidates.append((model, score))
        
        # Select model with highest score
        best_model = max(candidates, key=lambda x: x[1])[0]
        
        self.logger.info(f"Selected model {best_model} for task type {task_type}")
        return best_model
    
    def compare_models_for_task(self, task_type: str) -> Dict[str, Dict[str, Any]]:
        """Compare all models for a specific task."""
        comparison = {}
        
        for model, profile in self.model_profiles.items():
            suitability = 'high' if task_type in profile['best_for'] else 'medium'
            
            comparison[model] = {
                'suitability': suitability,
                'cost': profile['cost_per_1k_tokens'],
                'speed': profile['speed'],
                'quality': profile['quality']
            }
        
        return comparison


def main():
    """Demonstrate optimization techniques."""
    setup_logging()
    logger = get_logger("Optimization-Demo")
    
    print("⚡ LangChain Performance & Cost Optimization Demo")
    print("=" * 60)
    
    # Initialize components
    cache_manager = CacheManager(cache_type="memory")
    optimized_chain = OptimizedLLMChain(model="gpt-3.5-turbo", cache_manager=cache_manager)
    token_optimizer = TokenOptimizer()
    model_selector = ModelSelector()
    
    # Demo 1: Prompt optimization
    print("\n🎯 Prompt Optimization:")
    original_prompt = "I would like to know if you could please provide me with information about machine learning"
    optimized_prompt = token_optimizer.optimize_prompt(original_prompt)
    print(f"Original:  {original_prompt}")
    print(f"Optimized: {optimized_prompt}")
    print(f"Token savings: {len(original_prompt) - len(optimized_prompt)} characters")
    
    # Demo 2: Model selection
    print(f"\n🤖 Model Selection:")
    task_types = ['simple_qa', 'complex_reasoning', 'summarization']
    for task in task_types:
        best_model = model_selector.select_optimal_model(task, budget_constraint=0.01)
        print(f"  {task}: {best_model}")
    
    # Demo 3: Caching performance
    print(f"\n💾 Caching Performance:")
    test_prompts = [
        "What is machine learning?",
        "Explain artificial intelligence",
        "What is machine learning?",  # Duplicate for cache test
        "How does deep learning work?"
    ]
    
    for i, prompt in enumerate(test_prompts):
        result = optimized_chain.invoke(prompt, use_cache=True)
        cache_status = "HIT" if result['cached'] else "MISS"
        print(f"  Request {i+1}: {cache_status} - {result['response_time']:.3f}s")
    
    # Performance statistics
    stats = optimized_chain.get_performance_stats()
    print(f"\n📊 Performance Statistics:")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache misses: {stats['cache_misses']}")
    
    # Demo 4: Cost comparison
    print(f"\n💰 Cost Comparison:")
    models = ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo']
    sample_prompt = "Analyze the market trends for renewable energy"
    
    cost_comparison = token_optimizer.compare_model_costs(sample_prompt, models)
    for model, costs in cost_comparison.items():
        print(f"  {model}: ${costs['estimated_cost']:.4f}")
    
    # Demo 5: Batch processing
    print(f"\n🔄 Batch Processing:")
    batch_processor = BatchProcessor(batch_size=2, delay=0.1)
    
    def mock_llm_function(prompt: str) -> str:
        time.sleep(0.1)  # Simulate API call
        return f"Response to: {prompt[:30]}..."
    
    batch_prompts = [
        "What is AI?",
        "Explain ML",
        "Define NLP",
        "What is computer vision?"
    ]
    
    start_time = time.time()
    batch_results = batch_processor.process_batch_sync(batch_prompts, mock_llm_function)
    batch_time = time.time() - start_time
    
    print(f"  Processed {len(batch_prompts)} prompts in {batch_time:.2f}s")
    print(f"  Average time per prompt: {batch_time/len(batch_prompts):.2f}s")
    
    print("\n✅ Optimization demo completed!")
    print("\nKey optimization strategies:")
    print("1. Prompt optimization reduces token usage")
    print("2. Intelligent caching improves response times")
    print("3. Model selection balances cost and quality")
    print("4. Batch processing improves throughput")
    print("5. Performance monitoring identifies bottlenecks")


if __name__ == "__main__":
    main()