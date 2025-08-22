#!/usr/bin/env python3
\"\"\"
Asynchronous LangChain Operations - High-Performance Applications

This example demonstrates:
1. Async LLM calls and chains
2. Concurrent processing with asyncio
3. Batch processing optimization
4. Error handling in async contexts
5. Performance comparison: sync vs async
6. Best practices for async LangChain applications

Key concepts:
- Asynchronous programming with LangChain
- Concurrent execution for improved performance
- Proper error handling and resource management
- Optimization strategies for high-throughput applications
\"\"\"

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
import time
from datetime import datetime
import aiohttp

# Add utils to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import setup_logging, get_logger, get_api_key, async_timing_decorator
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager


class AsyncPerformanceMonitor(AsyncCallbackHandler):
    \"\"\"Async callback for monitoring performance.\"\"\"
    
    def __init__(self):
        self.start_times = {}
        self.metrics = []
    
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        \"\"\"Called when LLM starts.\"\"\"
        request_id = kwargs.get('invocation_params', {}).get('request_id', 'unknown')
        self.start_times[request_id] = time.time()
        print(f\"⚡ Async LLM started: {request_id}\")
    
    async def on_llm_end(self, response, **kwargs) -> None:
        \"\"\"Called when LLM ends.\"\"\"
        request_id = kwargs.get('invocation_params', {}).get('request_id', 'unknown')
        if request_id in self.start_times:
            duration = time.time() - self.start_times[request_id]
            self.metrics.append({
                'request_id': request_id,
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            })
            print(f\"✅ Async LLM completed: {request_id} ({duration:.2f}s)\")
            del self.start_times[request_id]
    
    async def on_llm_error(self, error, **kwargs) -> None:
        \"\"\"Called when LLM encounters an error.\"\"\"
        request_id = kwargs.get('invocation_params', {}).get('request_id', 'unknown')
        print(f\"❌ Async LLM error: {request_id} - {error}\")
        if request_id in self.start_times:
            del self.start_times[request_id]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        \"\"\"Get performance metrics summary.\"\"\"
        if not self.metrics:
            return {\"message\": \"No metrics collected\"}
        
        durations = [m['duration'] for m in self.metrics]
        return {
            \"total_requests\": len(self.metrics),
            \"average_duration\": sum(durations) / len(durations),
            \"min_duration\": min(durations),
            \"max_duration\": max(durations),
            \"total_time\": sum(durations)
        }


@async_timing_decorator
async def demonstrate_basic_async():
    \"\"\"Demonstrate basic async LLM operations.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping demonstration\")
        return
    
    logger.info(\"⚡ Basic Async Operations\")
    
    # Create async LLM
    llm = ChatOpenAI(
        openai_api_key=api_key,
        temperature=0.7,
        max_tokens=100
    )
    
    print(\"\n\" + \"=\"*60)
    print(\"BASIC ASYNC LLM OPERATIONS\")
    print(\"=\"*60)
    
    # Single async call
    print(\"\n1. Single Async Call:\")
    message = HumanMessage(content=\"What is artificial intelligence?\")
    
    start_time = time.time()
    response = await llm.agenerate([[message]])
    duration = time.time() - start_time
    
    print(f\"Response: {response.generations[0][0].text[:100]}...\")
    print(f\"Duration: {duration:.2f}s\")
    
    # Multiple sequential async calls
    print(\"\n2. Sequential Async Calls:\")
    questions = [
        \"What is machine learning?\",
        \"What is deep learning?\",
        \"What is natural language processing?\"
    ]
    
    start_time = time.time()
    responses = []
    
    for question in questions:
        message = HumanMessage(content=question)
        response = await llm.agenerate([[message]])
        responses.append(response.generations[0][0].text)
        print(f\"✅ Completed: {question}\")
    
    sequential_duration = time.time() - start_time
    print(f\"Sequential duration: {sequential_duration:.2f}s\")
    
    print(\"\n💡 Basic async operations still run sequentially.\")
    print(\"For true concurrency, use asyncio.gather() or similar.\")
    print(\"=\"*60)


@async_timing_decorator
async def demonstrate_concurrent_processing():
    \"\"\"Demonstrate concurrent processing with asyncio.gather().\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping demonstration\")
        return
    
    logger.info(\"🚀 Concurrent Processing\")
    
    # Create async LLM
    llm = ChatOpenAI(
        openai_api_key=api_key,
        temperature=0.5,
        max_tokens=100
    )
    
    print(\"\n\" + \"=\"*60)
    print(\"CONCURRENT PROCESSING WITH ASYNCIO\")
    print(\"=\"*60)
    
    # Define multiple tasks
    questions = [
        \"Explain quantum computing in one sentence.\",
        \"What is blockchain technology?\",
        \"How does machine learning work?\",
        \"What is cloud computing?\",
        \"Explain artificial intelligence briefly.\"
    ]
    
    # Function to process a single question
    async def process_question(question: str, index: int) -> Dict[str, Any]:
        print(f\"🚀 Starting task {index + 1}: {question[:30]}...\")
        start_time = time.time()
        
        try:
            message = HumanMessage(content=question)
            response = await llm.agenerate([[message]])
            duration = time.time() - start_time
            
            result = {
                \"index\": index,
                \"question\": question,
                \"answer\": response.generations[0][0].text.strip(),
                \"duration\": duration,
                \"status\": \"success\"
            }
            
            print(f\"✅ Completed task {index + 1} in {duration:.2f}s\")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            print(f\"❌ Failed task {index + 1}: {e}\")
            return {
                \"index\": index,
                \"question\": question,
                \"error\": str(e),
                \"duration\": duration,
                \"status\": \"failed\"
            }
    
    # Execute all tasks concurrently
    print(f\"\n🔄 Processing {len(questions)} questions concurrently...\")
    start_time = time.time()
    
    tasks = [process_question(q, i) for i, q in enumerate(questions)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    total_duration = time.time() - start_time
    
    # Display results
    print(f\"\n📊 Concurrent Results:\")
    successful_results = [r for r in results if isinstance(r, dict) and r.get('status') == 'success']
    
    for result in successful_results:
        print(f\"\n{result['index'] + 1}. {result['question']}\")
        print(f\"   Answer: {result['answer'][:80]}...\")
        print(f\"   Duration: {result['duration']:.2f}s\")
    
    # Performance summary
    if successful_results:
        avg_individual_time = sum(r['duration'] for r in successful_results) / len(successful_results)
        print(f\"\n⚡ Performance Summary:\")
        print(f\"Total concurrent time: {total_duration:.2f}s\")
        print(f\"Average individual time: {avg_individual_time:.2f}s\")
        print(f\"Speedup factor: {(avg_individual_time * len(successful_results)) / total_duration:.2f}x\")
    
    print(\"\n💡 Concurrent processing significantly improves performance.\")
    print(\"=\"*60)


@async_timing_decorator
async def demonstrate_async_chains():
    \"\"\"Demonstrate async chains and pipelines.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping demonstration\")
        return
    
    logger.info(\"⛓️ Async Chains\")
    
    # Create async LLM
    llm = ChatOpenAI(
        openai_api_key=api_key,
        temperature=0.3,
        max_tokens=150
    )
    
    # Create async chain
    prompt = PromptTemplate(
        input_variables=[\"topic\"],
        template=\"\"\"Analyze the topic '{topic}' and provide:
        1. A brief definition
        2. One key benefit
        3. One potential challenge
        
        Keep the response concise and structured.\"\"\"
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    print(\"\n\" + \"=\"*60)
    print(\"ASYNC CHAINS AND PIPELINES\")
    print(\"=\"*60)
    
    topics = [
        \"renewable energy\",
        \"artificial intelligence\",
        \"space exploration\",
        \"biotechnology\"
    ]
    
    # Function to process with async chain
    async def analyze_topic(topic: str) -> Dict[str, Any]:
        print(f\"🔍 Analyzing: {topic}\")
        start_time = time.time()
        
        try:
            result = await chain.arun(topic=topic)
            duration = time.time() - start_time
            
            print(f\"✅ Completed analysis of '{topic}' in {duration:.2f}s\")
            return {
                \"topic\": topic,
                \"analysis\": result,
                \"duration\": duration,
                \"status\": \"success\"
            }
        except Exception as e:
            duration = time.time() - start_time
            print(f\"❌ Failed to analyze '{topic}': {e}\")
            return {
                \"topic\": topic,
                \"error\": str(e),
                \"duration\": duration,
                \"status\": \"failed\"
            }
    
    # Run async chain analysis
    print(f\"\n🔗 Running async chain analysis on {len(topics)} topics...\")
    start_time = time.time()
    
    tasks = [analyze_topic(topic) for topic in topics]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    total_time = time.time() - start_time
    
    # Display results
    print(f\"\n📊 Chain Analysis Results:\")
    for result in results:
        if isinstance(result, dict) and result.get('status') == 'success':
            print(f\"\n📋 {result['topic'].title()}:\")
            print(f\"   {result['analysis'][:150]}...\")
            print(f\"   Processing time: {result['duration']:.2f}s\")
    
    print(f\"\n⏱️ Total async chain time: {total_time:.2f}s\")
    print(\"\n💡 Async chains enable efficient pipeline processing.\")
    print(\"=\"*60)


async def demonstrate_batch_optimization():
    \"\"\"Demonstrate batch processing optimization strategies.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping demonstration\")
        return
    
    logger.info(\"📦 Batch Processing Optimization\")
    
    # Create async LLM
    llm = ChatOpenAI(
        openai_api_key=api_key,
        temperature=0.5,
        max_tokens=80
    )
    
    print(\"\n\" + \"=\"*60)
    print(\"BATCH PROCESSING OPTIMIZATION\")
    print(\"=\"*60)
    
    # Large dataset simulation
    items = [
        \"smartphone\", \"laptop\", \"tablet\", \"smartwatch\", \"headphones\",
        \"camera\", \"speaker\", \"keyboard\", \"mouse\", \"monitor\",
        \"printer\", \"router\", \"microphone\", \"webcam\", \"charger\"
    ]
    
    async def process_item(item: str) -> str:
        \"\"\"Process a single item.\"\"\"
        try:
            message = HumanMessage(content=f\"Write a one-sentence marketing description for: {item}\")
            response = await llm.agenerate([[message]])
            return response.generations[0][0].text.strip()
        except Exception as e:
            return f\"Error processing {item}: {e}\"
    
    async def process_batch(batch: List[str], batch_num: int) -> List[Dict[str, Any]]:
        \"\"\"Process a batch of items.\"\"\"
        print(f\"📦 Processing batch {batch_num} ({len(batch)} items)...\")
        start_time = time.time()
        
        tasks = [process_item(item) for item in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        duration = time.time() - start_time
        print(f\"✅ Batch {batch_num} completed in {duration:.2f}s\")
        
        return [{
            \"item\": item,
            \"description\": result if isinstance(result, str) else f\"Error: {result}\",
            \"batch\": batch_num
        } for item, result in zip(batch, results)]
    
    # Different batch sizes for comparison
    batch_sizes = [3, 5, 8]
    
    for batch_size in batch_sizes:
        print(f\"\n🔧 Testing batch size: {batch_size}\")
        
        # Create batches
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        start_time = time.time()
        
        # Process all batches
        batch_tasks = [process_batch(batch, i + 1) for i, batch in enumerate(batches)]
        all_results = await asyncio.gather(*batch_tasks)
        
        total_time = time.time() - start_time
        
        # Flatten results
        flat_results = [item for batch_result in all_results for item in batch_result]
        
        print(f\"   📊 Batch size {batch_size}: {len(flat_results)} items in {total_time:.2f}s\")
        print(f\"   ⚡ Rate: {len(flat_results) / total_time:.2f} items/second\")
    
    print(\"\n💡 Batch Processing Tips:\")
    tips = [
        \"Optimize batch size based on API rate limits\",
        \"Consider memory usage with large batches\",
        \"Implement retry logic for failed batches\",
        \"Monitor performance across different batch sizes\",
        \"Use semaphores to control concurrency\"
    ]
    
    for tip in tips:
        print(f\"   • {tip}\")
    
    print(\"=\"*60)


async def demonstrate_error_handling():
    \"\"\"Demonstrate error handling in async operations.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping demonstration\")
        return
    
    logger.info(\"🛡️ Async Error Handling\")
    
    # Create async LLM
    llm = ChatOpenAI(
        openai_api_key=api_key,
        temperature=0.5,
        max_tokens=50
    )
    
    print(\"\n\" + \"=\"*60)
    print(\"ASYNC ERROR HANDLING STRATEGIES\")
    print(\"=\"*60)
    
    async def resilient_llm_call(prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        \"\"\"Make a resilient LLM call with retry logic.\"\"\"
        for attempt in range(max_retries):
            try:
                print(f\"🔄 Attempt {attempt + 1} for: {prompt[:30]}...\")
                
                message = HumanMessage(content=prompt)
                response = await llm.agenerate([[message]])
                
                result = {
                    \"prompt\": prompt,
                    \"response\": response.generations[0][0].text.strip(),
                    \"attempt\": attempt + 1,
                    \"status\": \"success\"
                }
                
                print(f\"✅ Success on attempt {attempt + 1}\")
                return result
                
            except Exception as e:
                print(f\"❌ Attempt {attempt + 1} failed: {e}\")
                
                if attempt == max_retries - 1:
                    return {
                        \"prompt\": prompt,
                        \"error\": str(e),
                        \"attempts\": max_retries,
                        \"status\": \"failed\"
                    }
                
                # Wait before retry (exponential backoff)
                wait_time = 2 ** attempt
                print(f\"⏳ Waiting {wait_time}s before retry...\")
                await asyncio.sleep(wait_time)
        
        return {\"status\": \"failed\", \"error\": \"Max retries exceeded\"}
    
    # Test with various prompts (some may fail)
    test_prompts = [
        \"What is the capital of France?\",
        \"Explain quantum physics in simple terms.\",
        \"\" * 1000,  # This might cause an error
        \"What is artificial intelligence?\"
    ]
    
    print(\"\n🧪 Testing resilient async calls...\")
    
    # Process with error handling
    tasks = [resilient_llm_call(prompt) for prompt in test_prompts if prompt.strip()]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Display results
    print(\"\n📊 Error Handling Results:\")
    for i, result in enumerate(results):
        if isinstance(result, dict):
            if result.get('status') == 'success':
                print(f\"\n✅ Success {i + 1}:\")
                print(f\"   Prompt: {result['prompt'][:50]}...\")
                print(f\"   Response: {result['response'][:80]}...\")
                print(f\"   Attempts: {result['attempt']}\")
            else:
                print(f\"\n❌ Failed {i + 1}:\")
                print(f\"   Prompt: {result.get('prompt', 'Unknown')[:50]}...\")
                print(f\"   Error: {result.get('error', 'Unknown error')}\")
        else:
            print(f\"\n💥 Exception {i + 1}: {result}\")
    
    print(\"\n💡 Async Error Handling Best Practices:\")
    practices = [
        \"Use try-catch blocks for individual operations\",
        \"Implement retry logic with exponential backoff\",
        \"Set reasonable timeouts for async operations\",
        \"Use circuit breakers for external service calls\",
        \"Log errors with sufficient context\",
        \"Provide graceful degradation when possible\",
        \"Monitor error rates and patterns\"
    ]
    
    for practice in practices:
        print(f\"   • {practice}\")
    
    print(\"=\"*60)


def demonstrate_performance_comparison():
    \"\"\"Demonstrate performance comparison between sync and async.\"\"\"
    logger = get_logger(__name__)
    logger.info(\"⚖️ Performance Comparison\")
    
    print(\"\n\" + \"=\"*70)
    print(\"SYNC VS ASYNC PERFORMANCE COMPARISON\")
    print(\"=\"*70)
    
    scenarios = {
        \"Single Request\": {
            \"description\": \"Processing one request at a time\",
            \"sync_advantage\": \"Lower overhead, simpler code\",
            \"async_advantage\": \"Minimal difference for single requests\",
            \"recommendation\": \"Use sync for simple, single-request scenarios\"
        },
        \"Multiple Independent Requests\": {
            \"description\": \"Processing multiple unrelated requests\",
            \"sync_advantage\": \"Predictable execution order\",
            \"async_advantage\": \"Significant speedup through concurrency\",
            \"recommendation\": \"Use async for 5+ concurrent requests\"
        },
        \"I/O Intensive Operations\": {
            \"description\": \"Operations with network calls, file I/O\",
            \"sync_advantage\": \"Easier debugging and error tracking\",
            \"async_advantage\": \"Much better resource utilization\",
            \"recommendation\": \"Async is almost always better for I/O\"
        },
        \"High-Throughput Applications\": {
            \"description\": \"Processing hundreds or thousands of requests\",
            \"sync_advantage\": \"Simpler deployment and monitoring\",
            \"async_advantage\": \"Dramatically better performance and resource usage\",
            \"recommendation\": \"Async is essential for high throughput\"
        }
    }
    
    for scenario_name, details in scenarios.items():
        print(f\"\n📊 {scenario_name}:\")
        print(f\"   Description: {details['description']}\")
        print(f\"   Sync Advantage: {details['sync_advantage']}\")
        print(f\"   Async Advantage: {details['async_advantage']}\")
        print(f\"   Recommendation: {details['recommendation']}\")
        print(\"-\" * 50)
    
    print(\"\n⚡ Performance Guidelines:\")
    guidelines = [
        \"Use async for I/O-bound operations (API calls, file operations)\",
        \"Sync is fine for CPU-bound tasks and simple workflows\",
        \"Async shines with 5+ concurrent operations\",
        \"Consider complexity vs. performance trade-offs\",
        \"Test both approaches with realistic workloads\",
        \"Monitor resource usage (CPU, memory, network)\",
        \"Async requires more careful error handling\"
    ]
    
    for guideline in guidelines:
        print(f\"   • {guideline}\")
    
    print(\"\n🎯 When to Choose Async:\")
    async_scenarios = [
        \"Multiple API calls that can run concurrently\",
        \"Real-time applications with streaming responses\",
        \"High-throughput batch processing\",
        \"Applications with significant I/O wait times\",
        \"Microservices with multiple external dependencies\"
    ]
    
    for scenario in async_scenarios:
        print(f\"   ✅ {scenario}\")
    
    print(\"\n🎯 When Sync Is Sufficient:\")
    sync_scenarios = [
        \"Simple scripts and one-off tasks\",
        \"Linear workflows with dependencies\",
        \"CPU-intensive processing\",
        \"Applications with low concurrency requirements\",
        \"Prototyping and development\"
    ]
    
    for scenario in sync_scenarios:
        print(f\"   ✅ {scenario}\")
    
    print(\"=\"*70)


async def main():
    \"\"\"Main async function demonstrating async concepts.\"\"\"
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info(\"🚀 Starting Async LangChain Operations Demonstration\")
    
    try:
        # Run all async demonstrations
        await demonstrate_basic_async()
        await demonstrate_concurrent_processing()
        await demonstrate_async_chains()
        await demonstrate_batch_optimization()
        await demonstrate_error_handling()
        demonstrate_performance_comparison()
        
        print(\"\n🎯 Async Key Takeaways:\")
        print(\"1. Async operations enable true concurrency and better performance\")
        print(\"2. Use asyncio.gather() for concurrent execution\")
        print(\"3. Async chains and callbacks provide non-blocking operations\")
        print(\"4. Proper error handling is crucial in async environments\")
        print(\"5. Batch processing strategies optimize throughput\")
        print(\"6. Choose async for I/O-bound and high-concurrency scenarios\")
        
        logger.info(\"✅ Async LangChain Operations demonstration completed successfully!\")
        
    except Exception as e:
        logger.error(f\"❌ Error occurred: {e}\")
        logger.info(\"💡 Check your API keys and internet connection\")


if __name__ == \"__main__\":
    # Run the async main function
    asyncio.run(main())