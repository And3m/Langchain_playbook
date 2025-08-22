# Performance & Cost Optimization ⚡

Advanced techniques for optimizing LangChain applications for performance, cost-efficiency, and scalability.

## 📋 Overview

This module covers:
- **Performance Optimization**: Response time and throughput improvements
- **Cost Reduction**: Token usage and model selection strategies
- **Caching**: Intelligent response caching mechanisms
- **Batch Processing**: Efficient handling of multiple requests
- **Monitoring**: Performance tracking and bottleneck identification

## 🎯 Token Optimization

### Prompt Engineering for Efficiency

```python
from performance_optimization import TokenOptimizer

optimizer = TokenOptimizer()

# Optimize verbose prompts
original = "I would like to know if you could please provide me with information about machine learning"
optimized = optimizer.optimize_prompt(original)
# Result: "Provide information about machine learning"

# Create efficient templates
template = optimizer.create_efficient_prompt_template("summarize")
# Returns: "Summarize in {word_limit} words:\n{text}"
```

### Cost Estimation and Comparison

```python
# Compare costs across models
models = ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo']
prompt = "Analyze market trends"

cost_comparison = optimizer.compare_model_costs(prompt, models)
for model, costs in cost_comparison.items():
    print(f"{model}: ${costs['estimated_cost']:.4f}")
```

## 💾 Intelligent Caching

### Memory Cache
```python
from performance_optimization import CacheManager

# Initialize memory cache
cache = CacheManager(cache_type="memory")

# Check for cached response
cached_response = cache.get(prompt, model="gpt-3.5-turbo")

# Cache new response
cache.set(prompt, model="gpt-3.5-turbo", response="...", ttl=3600)

# Get cache statistics
hit_rate = cache.get_hit_rate()  # Returns percentage
```

### Redis Cache (Optional)
```python
# Initialize Redis cache (requires redis-py)
cache = CacheManager(
    cache_type="redis", 
    redis_url="redis://localhost:6379"
)

# Same interface as memory cache
cached_response = cache.get(prompt, model="gpt-4")
```

## 🚀 Optimized LLM Chain

### All-in-One Optimization

```python
from performance_optimization import OptimizedLLMChain

# Initialize optimized chain
chain = OptimizedLLMChain(
    model="gpt-3.5-turbo",
    cache_manager=CacheManager()
)

# Invoke with optimizations
result = chain.invoke(
    prompt="Explain quantum computing",
    use_cache=True,
    optimize_prompt=True
)

# Access optimization metrics
print(f"Response time: {result['response_time']:.3f}s")
print(f"Token savings: {result['token_savings']}")
print(f"Cost estimate: ${result['cost_estimate']:.4f}")
print(f"From cache: {result['cached']}")
```

## 🤖 Intelligent Model Selection

### Task-Based Selection

```python
from performance_optimization import ModelSelector

selector = ModelSelector()

# Select optimal model for task
best_model = selector.select_optimal_model(
    task_type='simple_qa',
    budget_constraint=0.01,  # $0.01 per 1K tokens
    speed_requirement='fast'
)

# Compare models for specific task
comparison = selector.compare_models_for_task('complex_reasoning')
```

### Model Profiles
- **GPT-3.5-turbo**: Fast, cost-effective, good for simple tasks
- **GPT-4**: Highest quality, slower, expensive, best for complex reasoning
- **GPT-4-turbo**: Balanced option, good for long context

## 🔄 Batch Processing

### Synchronous Batch Processing

```python
from performance_optimization import BatchProcessor

processor = BatchProcessor(batch_size=5, delay=1.0)

def your_llm_function(prompt):
    # Your LLM implementation
    return llm.predict(prompt)

prompts = ["Question 1", "Question 2", "Question 3"]
results = processor.process_batch_sync(prompts, your_llm_function)
```

### Asynchronous Batch Processing

```python
import asyncio

async def your_async_llm_function(prompt):
    # Your async LLM implementation
    return await async_llm.apredict(prompt)

# Process asynchronously
results = await processor.process_batch_async(prompts, your_async_llm_function)
```

## 📊 Performance Monitoring

### Function Monitoring Decorator

```python
from performance_optimization import performance_monitor

@performance_monitor
def your_function():
    # Function automatically monitored
    return llm.predict("What is AI?")

# Execution time and success rate automatically logged
```

### Performance Metrics

```python
# Get comprehensive performance stats
stats = chain.get_performance_stats()

print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
print(f"Cache hits: {stats['cache_hits']}")
print(f"Cache misses: {stats['cache_misses']}")
```

## 💡 Optimization Strategies

### 1. Prompt Optimization
- **Remove redundant words**: "Please provide" → "Provide"
- **Use structured formats**: Bullet points, numbered lists
- **Set clear constraints**: Word limits, format requirements
- **Eliminate ambiguity**: Specific, direct questions

### 2. Model Selection
- **Simple tasks**: GPT-3.5-turbo for basic Q&A
- **Complex reasoning**: GPT-4 for analysis and creativity
- **Long context**: GPT-4-turbo for document processing
- **Cost-sensitive**: Always start with cheaper models

### 3. Caching Strategy
- **Cache duration**: Set appropriate TTL based on content freshness
- **Cache keys**: Include model, temperature, and other parameters
- **Cache size**: Monitor memory usage and implement LRU eviction
- **Cache warming**: Pre-populate with common queries

### 4. Batch Processing
- **Optimal batch size**: Balance latency and throughput
- **Rate limiting**: Respect API rate limits
- **Error handling**: Implement retry logic for failed requests
- **Async processing**: Use async for I/O-bound operations

## 📈 Performance Patterns

### Response Time Optimization

```python
# Pattern 1: Cache-first approach
def optimized_query(prompt):
    # Check cache first
    cached = cache.get(prompt, model)
    if cached:
        return cached
    
    # Generate and cache
    response = llm.predict(prompt)
    cache.set(prompt, model, response)
    return response

# Pattern 2: Parallel processing
async def parallel_queries(prompts):
    tasks = [llm.apredict(prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)
```

### Cost Optimization

```python
# Pattern 1: Progressive model selection
def cost_aware_query(prompt, max_cost=0.01):
    # Try cheaper model first
    if estimated_cost(prompt, 'gpt-3.5-turbo') <= max_cost:
        return llm_35.predict(prompt)
    
    # Use more expensive model if needed
    return llm_4.predict(prompt)

# Pattern 2: Prompt compression
def compressed_query(long_prompt):
    # Summarize or compress long prompts
    if len(long_prompt) > 1000:
        summary = summarize(long_prompt)
        return llm.predict(summary)
    return llm.predict(long_prompt)
```

## 🔧 Configuration Examples

### Production Configuration

```python
# High-performance production setup
cache_manager = CacheManager(
    cache_type="redis",
    redis_url="redis://redis-cluster:6379"
)

optimized_chain = OptimizedLLMChain(
    model="gpt-3.5-turbo",  # Cost-effective choice
    cache_manager=cache_manager
)

batch_processor = BatchProcessor(
    batch_size=10,  # Higher throughput
    delay=0.5       # Rate limiting
)
```

### Development Configuration

```python
# Development setup with verbose logging
cache_manager = CacheManager(cache_type="memory")

optimized_chain = OptimizedLLMChain(
    model="gpt-3.5-turbo",
    cache_manager=cache_manager
)

# Enable detailed monitoring
import logging
logging.getLogger("Performance-Monitor").setLevel(logging.DEBUG)
```

## 📊 Benchmarking

### Performance Benchmarking

```python
import time

def benchmark_optimization():
    prompts = ["Question " + str(i) for i in range(100)]
    
    # Without optimization
    start = time.time()
    results_basic = [llm.predict(p) for p in prompts]
    time_basic = time.time() - start
    
    # With optimization
    start = time.time()
    results_optimized = [optimized_chain.invoke(p) for p in prompts]
    time_optimized = time.time() - start
    
    improvement = (time_basic - time_optimized) / time_basic * 100
    print(f"Performance improvement: {improvement:.1f}%")
```

### Cost Analysis

```python
def analyze_cost_savings():
    test_prompts = ["Sample prompt " + str(i) for i in range(50)]
    
    total_cost_basic = sum(estimate_cost(p, 'gpt-4') for p in test_prompts)
    total_cost_optimized = sum(estimate_cost(optimize_prompt(p), 'gpt-3.5-turbo') for p in test_prompts)
    
    savings = (total_cost_basic - total_cost_optimized) / total_cost_basic * 100
    print(f"Cost savings: {savings:.1f}%")
```

## 🚀 Quick Start

1. **Run the optimization demo**:
   ```bash
   cd advanced/05_optimization
   python performance_optimization.py
   ```

2. **Integrate optimizations**:
   ```python
   from performance_optimization import OptimizedLLMChain, CacheManager
   
   # Set up optimized chain
   cache = CacheManager(cache_type="memory")
   chain = OptimizedLLMChain(model="gpt-3.5-turbo", cache_manager=cache)
   
   # Use optimized invocation
   result = chain.invoke("Your prompt here", use_cache=True, optimize_prompt=True)
   ```

3. **Monitor performance**:
   ```python
   stats = chain.get_performance_stats()
   print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
   ```

## 📚 Advanced Topics

- **Custom caching strategies** for domain-specific needs
- **Model fine-tuning** for specialized tasks
- **Response streaming** for real-time applications
- **Load balancing** across multiple model instances
- **Cost budgeting** and usage alerts

## 📖 Additional Resources

- [OpenAI Pricing](https://openai.com/pricing)
- [LangChain Performance Guide](https://docs.langchain.com/docs/guides/performance)
- [Redis Documentation](https://redis.io/documentation)

---

**Optimize for Speed, Cost, and Scale! 🚀**