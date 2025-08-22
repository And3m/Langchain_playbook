#!/usr/bin/env python3
\"\"\"
LangChain Callbacks - Monitoring and Observability

This example demonstrates:
1. Built-in callback handlers
2. Custom callback implementations
3. Streaming callbacks for real-time responses
4. Token usage tracking and cost monitoring
5. Performance monitoring and debugging
6. Integration with logging and monitoring systems

Key concepts:
- Observability in LLM applications
- Real-time monitoring and feedback
- Cost tracking and optimization
- Debugging and troubleshooting
\"\"\"

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import time
import json
from datetime import datetime

# Add utils to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import setup_logging, get_logger, get_api_key, timing_decorator
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import LLMChain, ConversationChain
from langchain.prompts import PromptTemplate
from langchain.schema import LLMResult, HumanMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.memory import ConversationBufferMemory


class TokenUsageTracker(BaseCallbackHandler):
    \"\"\"Custom callback to track token usage and costs.\"\"\"
    
    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0.0
        self.requests = []
        
        # Pricing per 1K tokens (approximate as of 2024)
        self.pricing = {
            \"gpt-3.5-turbo\": 0.0015,
            \"gpt-4\": 0.03,
            \"gpt-4-turbo\": 0.01,
            \"text-davinci-003\": 0.02
        }
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        \"\"\"Called when LLM starts running.\"\"\"
        self.start_time = time.time()
        print(f\"🚀 LLM started with {len(prompts)} prompt(s)\")
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        \"\"\"Called when LLM ends running.\"\"\"
        end_time = time.time()
        duration = end_time - self.start_time
        
        # Extract token usage if available
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            if token_usage:
                prompt_tokens = token_usage.get('prompt_tokens', 0)
                completion_tokens = token_usage.get('completion_tokens', 0)
                total_tokens = token_usage.get('total_tokens', prompt_tokens + completion_tokens)
                
                self.total_tokens += total_tokens
                
                # Estimate cost (using gpt-3.5-turbo pricing as default)
                model_name = response.llm_output.get('model_name', 'gpt-3.5-turbo')
                price_per_1k = self.pricing.get(model_name, 0.0015)
                request_cost = (total_tokens / 1000) * price_per_1k
                self.total_cost += request_cost
                
                # Record request details
                request_info = {
                    \"timestamp\": datetime.now().isoformat(),
                    \"model\": model_name,
                    \"duration\": round(duration, 2),
                    \"prompt_tokens\": prompt_tokens,
                    \"completion_tokens\": completion_tokens,
                    \"total_tokens\": total_tokens,
                    \"cost\": round(request_cost, 6)
                }
                self.requests.append(request_info)
                
                print(f\"📊 Request completed: {total_tokens} tokens, ${request_cost:.6f}, {duration:.2f}s\")
            else:
                print(f\"⏱️ Request completed in {duration:.2f}s (no token info available)\")
    
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        \"\"\"Called when LLM encounters an error.\"\"\"
        print(f\"❌ LLM Error: {error}\")
    
    def get_summary(self) -> Dict[str, Any]:
        \"\"\"Get usage summary.\"\"\"
        return {
            \"total_requests\": len(self.requests),
            \"total_tokens\": self.total_tokens,
            \"total_cost\": round(self.total_cost, 4),
            \"average_tokens_per_request\": round(self.total_tokens / max(len(self.requests), 1), 2),
            \"requests\": self.requests
        }


class PerformanceMonitor(BaseCallbackHandler):
    \"\"\"Monitor chain performance and execution flow.\"\"\"
    
    def __init__(self):
        self.chain_stack = []
        self.step_times = []
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        \"\"\"Called when a chain starts.\"\"\"
        chain_name = serialized.get('name', 'Unknown Chain')
        start_time = time.time()
        
        self.chain_stack.append({
            \"name\": chain_name,
            \"start_time\": start_time,
            \"inputs\": inputs
        })
        
        indent = \"  \" * (len(self.chain_stack) - 1)
        print(f\"{indent}🔗 {chain_name} started\")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        \"\"\"Called when a chain ends.\"\"\"
        if self.chain_stack:
            chain_info = self.chain_stack.pop()
            duration = time.time() - chain_info[\"start_time\"]
            
            indent = \"  \" * len(self.chain_stack)
            print(f\"{indent}✅ {chain_info['name']} completed in {duration:.2f}s\")
            
            self.step_times.append({
                \"chain\": chain_info[\"name\"],
                \"duration\": duration,
                \"timestamp\": datetime.now().isoformat()
            })
    
    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        \"\"\"Called when a chain encounters an error.\"\"\"
        if self.chain_stack:
            chain_info = self.chain_stack.pop()
            indent = \"  \" * len(self.chain_stack)
            print(f\"{indent}❌ {chain_info['name']} failed: {error}\")
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        \"\"\"Called when a tool starts.\"\"\"
        tool_name = serialized.get('name', 'Unknown Tool')
        indent = \"  \" * len(self.chain_stack)
        print(f\"{indent}🔧 Tool {tool_name} called with: {input_str[:50]}...\")
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        \"\"\"Called when a tool ends.\"\"\"
        indent = \"  \" * len(self.chain_stack)
        print(f\"{indent}🔧 Tool completed with output: {output[:50]}...\")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        \"\"\"Get performance summary.\"\"\"
        if not self.step_times:
            return {\"message\": \"No performance data collected\"}
        
        total_time = sum(step[\"duration\"] for step in self.step_times)
        avg_time = total_time / len(self.step_times)
        
        return {
            \"total_steps\": len(self.step_times),
            \"total_time\": round(total_time, 2),
            \"average_step_time\": round(avg_time, 2),
            \"steps\": self.step_times
        }


class DetailedLogger(BaseCallbackHandler):
    \"\"\"Detailed logging callback for debugging.\"\"\"
    
    def __init__(self, log_level: str = \"INFO\"):
        self.logger = get_logger(self.__class__.__name__)
        self.log_level = log_level
        self.call_count = 0
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        \"\"\"Log LLM start.\"\"\"
        self.call_count += 1
        self.logger.info(f\"LLM Call #{self.call_count} - Model: {serialized.get('name', 'Unknown')}\")
        if self.log_level == \"DEBUG\":
            for i, prompt in enumerate(prompts):
                self.logger.debug(f\"Prompt {i+1}: {prompt[:200]}...\")
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        \"\"\"Log new tokens (for streaming).\"\"\"
        if self.log_level == \"DEBUG\":
            self.logger.debug(f\"New token: '{token}'\")
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        \"\"\"Log LLM completion.\"\"\"
        self.logger.info(f\"LLM Call #{self.call_count} completed\")
        if self.log_level == \"DEBUG\":
            for i, generation in enumerate(response.generations):
                for j, gen in enumerate(generation):
                    self.logger.debug(f\"Generation {i}-{j}: {gen.text[:100]}...\")
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        \"\"\"Log chain start.\"\"\"
        chain_name = serialized.get('name', 'Unknown')
        self.logger.info(f\"Chain '{chain_name}' started with inputs: {list(inputs.keys())}\")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        \"\"\"Log chain completion.\"\"\"
        self.logger.info(f\"Chain completed with outputs: {list(outputs.keys())}\")
    
    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        \"\"\"Log chain errors.\"\"\"
        self.logger.error(f\"Chain error: {error}\")


class StreamingCallback(BaseCallbackHandler):
    \"\"\"Custom streaming callback with formatting.\"\"\"
    
    def __init__(self, prefix: str = \"🤖 \"):
        self.prefix = prefix
        self.tokens = []
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        \"\"\"Handle new token during streaming.\"\"\"
        if not self.tokens:  # First token
            print(f\"\n{self.prefix}\", end=\"\", flush=True)
        
        print(token, end=\"\", flush=True)
        self.tokens.append(token)
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        \"\"\"Handle end of streaming.\"\"\"
        print()  # New line after streaming
        self.tokens = []


def demonstrate_token_tracking():
    \"\"\"Demonstrate token usage tracking and cost monitoring.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping demonstration\")
        return
    
    logger.info(\"📊 Token Usage Tracking\")
    
    # Create token tracker
    token_tracker = TokenUsageTracker()
    
    # Create LLM with callback
    llm = ChatOpenAI(
        openai_api_key=api_key,
        temperature=0.7,
        max_tokens=100,
        callbacks=[token_tracker]
    )
    
    print(\"\n\" + \"=\"*60)
    print(\"TOKEN USAGE TRACKING AND COST MONITORING\")
    print(\"=\"*60)
    
    # Test different types of prompts
    test_prompts = [
        \"What is machine learning?\",
        \"Explain quantum computing in simple terms.\",
        \"Write a short poem about programming.\",
        \"List 5 benefits of renewable energy.\"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        try:
            print(f\"\n📝 Request {i}: {prompt}\")
            response = llm([HumanMessage(content=prompt)])
            print(f\"Response: {response.content[:100]}...\")
        except Exception as e:
            logger.error(f\"Error in request {i}: {e}\")
    
    # Show summary
    summary = token_tracker.get_summary()
    print(f\"\n📈 Usage Summary:\")
    print(f\"Total Requests: {summary['total_requests']}\")
    print(f\"Total Tokens: {summary['total_tokens']}\")
    print(f\"Total Cost: ${summary['total_cost']}\")
    print(f\"Average Tokens/Request: {summary['average_tokens_per_request']}\")
    
    print(\"\n💡 Token tracking helps monitor costs and optimize usage.\")
    print(\"=\"*60)


def demonstrate_performance_monitoring():
    \"\"\"Demonstrate performance monitoring with chains.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping demonstration\")
        return
    
    logger.info(\"⚡ Performance Monitoring\")
    
    # Create performance monitor
    perf_monitor = PerformanceMonitor()
    
    # Create LLM and chain with callback
    llm = ChatOpenAI(
        openai_api_key=api_key,
        temperature=0.5,
        max_tokens=150,
        callbacks=[perf_monitor]
    )
    
    # Create a chain for testing
    prompt = PromptTemplate(
        input_variables=[\"topic\"],
        template=\"Write a brief explanation of {topic} in exactly 3 sentences.\"
    )
    
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        callbacks=[perf_monitor]
    )
    
    print(\"\n\" + \"=\"*60)
    print(\"PERFORMANCE MONITORING\")
    print(\"=\"*60)
    
    topics = [\"artificial intelligence\", \"blockchain\", \"renewable energy\"]
    
    for topic in topics:
        try:
            print(f\"\n🎯 Processing topic: {topic}\")
            result = chain.run(topic=topic)
            print(f\"Result: {result[:100]}...\")
        except Exception as e:
            logger.error(f\"Error processing {topic}: {e}\")
    
    # Show performance summary
    perf_summary = perf_monitor.get_performance_summary()
    print(f\"\n📊 Performance Summary:\")
    print(f\"Total Steps: {perf_summary.get('total_steps', 0)}\")
    print(f\"Total Time: {perf_summary.get('total_time', 0)}s\")
    print(f\"Average Step Time: {perf_summary.get('average_step_time', 0)}s\")
    
    print(\"\n💡 Performance monitoring helps identify bottlenecks.\")
    print(\"=\"*60)


def demonstrate_streaming_callbacks():
    \"\"\"Demonstrate streaming callbacks for real-time responses.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping demonstration\")
        return
    
    logger.info(\"🌊 Streaming Callbacks\")
    
    # Create streaming callback
    streaming_callback = StreamingCallback(prefix=\"🤖 AI: \")
    
    # Create LLM with streaming
    llm = ChatOpenAI(
        openai_api_key=api_key,
        temperature=0.7,
        max_tokens=200,
        streaming=True,
        callbacks=[streaming_callback]
    )
    
    print(\"\n\" + \"=\"*60)
    print(\"STREAMING CALLBACKS\")
    print(\"=\"*60)
    print(\"Watch the response appear in real-time:\n\")
    
    prompts = [
        \"Tell me a short story about a robot learning to paint.\",
        \"Explain the process of photosynthesis step by step.\"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        try:
            print(f\"👤 User: {prompt}\")
            response = llm([HumanMessage(content=prompt)])
            print(f\"\n📝 Complete response received for prompt {i}\n\")
            print(\"-\" * 40)
        except Exception as e:
            logger.error(f\"Error in streaming {i}: {e}\")
    
    print(\"\n💡 Streaming provides better user experience for long responses.\")
    print(\"=\"*60)


def demonstrate_detailed_logging():
    \"\"\"Demonstrate detailed logging for debugging.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping demonstration\")
        return
    
    logger.info(\"🔍 Detailed Logging\")
    
    # Create detailed logger
    detailed_logger = DetailedLogger(log_level=\"INFO\")
    
    # Create chain with detailed logging
    llm = ChatOpenAI(
        openai_api_key=api_key,
        temperature=0.3,
        max_tokens=100,
        callbacks=[detailed_logger]
    )
    
    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        callbacks=[detailed_logger]
    )
    
    print(\"\n\" + \"=\"*60)
    print(\"DETAILED LOGGING FOR DEBUGGING\")
    print(\"=\"*60)
    print(\"Check the logs for detailed execution information\n\")
    
    conversation_turns = [
        \"Hi, I'm learning about AI. What should I start with?\",
        \"That's helpful! Can you tell me more about machine learning?\"
    ]
    
    for turn in conversation_turns:
        try:
            print(f\"👤 User: {turn}\")
            response = conversation.predict(input=turn)
            print(f\"🤖 Assistant: {response[:100]}...\")
            print(\"-\" * 30)
        except Exception as e:
            logger.error(f\"Error in conversation: {e}\")
    
    print(\"\n💡 Detailed logging helps with debugging and monitoring.\")
    print(\"=\"*60)


def demonstrate_multiple_callbacks():
    \"\"\"Demonstrate using multiple callbacks together.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping demonstration\")
        return
    
    logger.info(\"🔧 Multiple Callbacks\")
    
    # Create multiple callbacks
    token_tracker = TokenUsageTracker()
    perf_monitor = PerformanceMonitor()
    detailed_logger = DetailedLogger(log_level=\"INFO\")
    
    # Combine callbacks
    callbacks = [token_tracker, perf_monitor, detailed_logger]
    
    # Create LLM with all callbacks
    llm = ChatOpenAI(
        openai_api_key=api_key,
        temperature=0.5,
        max_tokens=150,
        callbacks=callbacks
    )
    
    prompt = PromptTemplate(
        input_variables=[\"task\", \"context\"],
        template=\"\"\"Given the context: {context}
        
        Please complete this task: {task}
        
        Provide a clear and concise response.\"\"\"
    )
    
    chain = LLMChain(llm=llm, prompt=prompt, callbacks=callbacks)
    
    print(\"\n\" + \"=\"*60)
    print(\"MULTIPLE CALLBACKS WORKING TOGETHER\")
    print(\"=\"*60)
    
    test_case = {
        \"context\": \"You are helping a student learn programming concepts.\",
        \"task\": \"Explain what a function is in programming with a simple example.\"
    }
    
    try:
        print(f\"🎯 Task: {test_case['task']}\")
        result = chain.run(**test_case)
        print(f\"\n📝 Result: {result}\")
        
        # Show combined insights
        print(\"\n📊 Combined Monitoring Results:\")
        
        # Token usage
        token_summary = token_tracker.get_summary()
        print(f\"💰 Cost: ${token_summary['total_cost']}\")
        print(f\"🎯 Tokens: {token_summary['total_tokens']}\")
        
        # Performance
        perf_summary = perf_monitor.get_performance_summary()
        print(f\"⏱️ Duration: {perf_summary.get('total_time', 0)}s\")
        print(f\"🔗 Steps: {perf_summary.get('total_steps', 0)}\")
        
    except Exception as e:
        logger.error(f\"Error in multiple callbacks demo: {e}\")
    
    print(\"\n💡 Multiple callbacks provide comprehensive monitoring.\")
    print(\"=\"*60)


def demonstrate_callback_best_practices():
    \"\"\"Demonstrate callback best practices and patterns.\"\"\"
    logger = get_logger(__name__)
    logger.info(\"🎯 Callback Best Practices\")
    
    print(\"\n\" + \"=\"*70)
    print(\"CALLBACK BEST PRACTICES AND PATTERNS\")
    print(\"=\"*70)
    
    best_practices = {
        \"Performance\": [
            \"Keep callback logic lightweight and fast\",
            \"Avoid blocking operations in callbacks\",
            \"Use async callbacks for I/O operations\",
            \"Cache expensive computations\"
        ],
        \"Error Handling\": [
            \"Always handle exceptions in callback methods\",
            \"Use try-catch blocks to prevent callback failures\",
            \"Log errors but don't propagate them\",
            \"Provide fallback behavior for critical callbacks\"
        ],
        \"Data Management\": [
            \"Be careful with memory usage in long-running callbacks\",
            \"Clean up resources properly\",
            \"Use appropriate data structures for storing metrics\",
            \"Consider data persistence for important metrics\"
        ],
        \"Security\": [
            \"Don't log sensitive information\",
            \"Validate inputs before processing\",
            \"Be careful with external API calls\",
            \"Follow data privacy regulations\"
        ],
        \"Production Readiness\": [
            \"Implement proper logging levels\",
            \"Make callbacks configurable\",
            \"Include monitoring and alerting\",
            \"Test callback behavior under load\"
        ]
    }
    
    for category, practices in best_practices.items():
        print(f\"\n🏷️ {category}:\")
        for practice in practices:
            print(f\"   • {practice}\")
    
    print(\"\n🔧 Common Callback Patterns:\")
    patterns = {
        \"Observer Pattern\": \"Multiple callbacks observing the same events\",
        \"Chain of Responsibility\": \"Callbacks processing events in sequence\",
        \"Event Aggregation\": \"Combining multiple callback outputs\",
        \"Conditional Callbacks\": \"Activating callbacks based on conditions\",
        \"Callback Composition\": \"Building complex monitoring from simple callbacks\"
    }
    
    for pattern, description in patterns.items():
        print(f\"   📋 {pattern}: {description}\")
    
    print(\"\n⚠️ Common Pitfalls to Avoid:\")
    pitfalls = [
        \"Making callbacks too heavy or slow\",
        \"Not handling callback exceptions\",
        \"Logging too much sensitive information\",
        \"Creating memory leaks in long-running callbacks\",
        \"Not testing callback behavior properly\",
        \"Ignoring callback performance impact\"
    ]
    
    for pitfall in pitfalls:
        print(f\"   ❌ {pitfall}\")
    
    print(\"\n✅ Callback Implementation Checklist:\")
    checklist = [
        \"Implement proper error handling\",
        \"Keep callback logic lightweight\",
        \"Add appropriate logging\",
        \"Test with various scenarios\",
        \"Monitor callback performance\",
        \"Document callback behavior\",
        \"Consider thread safety if needed\",
        \"Plan for callback maintenance\"
    ]
    
    for item in checklist:
        print(f\"   ☑️ {item}\")
    
    print(\"=\"*70)


def main():
    \"\"\"Main function demonstrating callback concepts.\"\"\"
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info(\"🚀 Starting LangChain Callbacks Demonstration\")
    
    try:
        # Run all demonstrations
        demonstrate_token_tracking()
        demonstrate_performance_monitoring()
        demonstrate_streaming_callbacks()
        demonstrate_detailed_logging()
        demonstrate_multiple_callbacks()
        demonstrate_callback_best_practices()
        
        print(\"\n🎯 Callback Key Takeaways:\")
        print(\"1. Callbacks provide observability into LLM applications\")
        print(\"2. Token tracking helps monitor costs and usage\")
        print(\"3. Performance monitoring identifies bottlenecks\")
        print(\"4. Streaming callbacks improve user experience\")
        print(\"5. Multiple callbacks can work together for comprehensive monitoring\")
        print(\"6. Proper callback design is crucial for production applications\")
        
        logger.info(\"✅ LangChain Callbacks demonstration completed successfully!\")
        
    except Exception as e:
        logger.error(f\"❌ Error occurred: {e}\")
        logger.info(\"💡 Check your API keys and internet connection\")


if __name__ == \"__main__\":
    main()