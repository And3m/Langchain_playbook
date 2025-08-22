#!/usr/bin/env python3
\"\"\"
LLM Models Comparison

This example demonstrates different types of language models in LangChain:
1. Completion models (OpenAI GPT-3.5, GPT-4)
2. Chat models (ChatOpenAI)
3. Different providers (OpenAI, Anthropic, Google)
4. Model parameters and their effects

Key concepts:
- Model selection and configuration
- Temperature and creativity control
- Max tokens and response length
- Comparing outputs across models
\"\"\"

import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Any

# Add utils to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import setup_logging, get_logger, get_api_key, timing_decorator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage


@timing_decorator
def compare_completion_models():
    \"\"\"Compare different OpenAI completion models.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping comparison\")
        return
    
    logger.info(\"🔄 Comparing completion models...\")
    
    # Different model configurations
    models_config = [
        {\"name\": \"GPT-3.5 (Creative)\", \"model\": \"gpt-3.5-turbo-instruct\", \"temperature\": 0.9},
        {\"name\": \"GPT-3.5 (Balanced)\", \"model\": \"gpt-3.5-turbo-instruct\", \"temperature\": 0.7},
        {\"name\": \"GPT-3.5 (Precise)\", \"model\": \"gpt-3.5-turbo-instruct\", \"temperature\": 0.1},
    ]
    
    prompt = \"Write a creative story about a robot learning to paint in exactly 50 words.\"
    
    print(\"\n\" + \"=\"*70)
    print(\"COMPLETION MODELS COMPARISON\")
    print(\"=\"*70)
    print(f\"Prompt: {prompt}\n\")
    
    for config in models_config:
        try:
            llm = OpenAI(
                model_name=config[\"model\"],
                temperature=config[\"temperature\"],
                max_tokens=100,
                openai_api_key=api_key
            )
            
            response = llm(prompt)
            
            print(f\"📋 {config['name']} (temp={config['temperature']})\")
            print(f\"Response: {response.strip()}\")
            print(\"-\" * 50)
            
        except Exception as e:
            logger.error(f\"Error with {config['name']}: {e}\")


@timing_decorator
def compare_chat_models():
    \"\"\"Compare different chat model configurations.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping comparison\")
        return
    
    logger.info(\"💬 Comparing chat models...\")
    
    # Different chat model configurations
    chat_configs = [
        {\"name\": \"GPT-3.5 Turbo\", \"model\": \"gpt-3.5-turbo\", \"temperature\": 0.7},
        {\"name\": \"GPT-4 (if available)\", \"model\": \"gpt-4\", \"temperature\": 0.7},
    ]
    
    message = HumanMessage(content=\"Explain quantum computing in simple terms, using an analogy.\")
    
    print(\"\n\" + \"=\"*70)
    print(\"CHAT MODELS COMPARISON\")
    print(\"=\"*70)
    print(f\"Message: {message.content}\n\")
    
    for config in chat_configs:
        try:
            chat = ChatOpenAI(
                model_name=config[\"model\"],
                temperature=config[\"temperature\"],
                max_tokens=150,
                openai_api_key=api_key
            )
            
            response = chat([message])
            
            print(f\"🤖 {config['name']} (temp={config['temperature']})\")
            print(f\"Response: {response.content}\")
            print(\"-\" * 50)
            
        except Exception as e:
            logger.warning(f\"Could not test {config['name']}: {e}\")
            # GPT-4 might not be available for all users
            if \"gpt-4\" in config[\"model\"]:
                logger.info(\"💡 GPT-4 requires special access. Using GPT-3.5 for now.\")


def demonstrate_temperature_effects():
    \"\"\"Demonstrate how temperature affects model creativity.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        return
    
    logger.info(\"🌡️ Demonstrating temperature effects...\")
    
    temperatures = [0.0, 0.5, 1.0]
    prompt = \"Complete this sentence: The future of artificial intelligence is\"
    
    print(\"\n\" + \"=\"*70)
    print(\"TEMPERATURE EFFECTS DEMONSTRATION\")
    print(\"=\"*70)
    print(f\"Prompt: {prompt}\n\")
    
    for temp in temperatures:
        try:
            llm = OpenAI(
                temperature=temp,
                max_tokens=50,
                openai_api_key=api_key
            )
            
            response = llm(prompt)
            
            temp_desc = {
                0.0: \"Deterministic/Precise\",
                0.5: \"Balanced\", 
                1.0: \"Creative/Random\"
            }
            
            print(f\"🌡️ Temperature {temp} ({temp_desc[temp]})\")
            print(f\"Response: {prompt}{response.strip()}\")
            print(\"-\" * 50)
            
        except Exception as e:
            logger.error(f\"Error with temperature {temp}: {e}\")


def demonstrate_max_tokens():
    \"\"\"Demonstrate how max_tokens affects response length.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        return
    
    logger.info(\"📏 Demonstrating max_tokens effects...\")
    
    token_limits = [20, 50, 100]
    prompt = \"Explain the concept of machine learning and its applications in detail.\"
    
    print(\"\n\" + \"=\"*70)
    print(\"MAX TOKENS EFFECTS DEMONSTRATION\")
    print(\"=\"*70)
    print(f\"Prompt: {prompt}\n\")
    
    for max_tokens in token_limits:
        try:
            llm = OpenAI(
                temperature=0.5,
                max_tokens=max_tokens,
                openai_api_key=api_key
            )
            
            response = llm(prompt)
            word_count = len(response.split())
            
            print(f\"📏 Max tokens: {max_tokens} (≈{word_count} words)\")
            print(f\"Response: {response.strip()}\")
            print(\"-\" * 50)
            
        except Exception as e:
            logger.error(f\"Error with max_tokens {max_tokens}: {e}\")


def demonstrate_multiple_providers():
    \"\"\"Demonstrate using multiple model providers if available.\"\"\"
    logger = get_logger(__name__)
    
    logger.info(\"🌐 Checking multiple providers...\")
    
    prompt = \"What is the meaning of life?\"
    providers_tested = []
    
    print(\"\n\" + \"=\"*70)
    print(\"MULTIPLE PROVIDERS DEMONSTRATION\")
    print(\"=\"*70)
    print(f\"Prompt: {prompt}\n\")
    
    # OpenAI
    openai_key = get_api_key('openai')
    if openai_key:
        try:
            from langchain.chat_models import ChatOpenAI
            chat = ChatOpenAI(openai_api_key=openai_key, temperature=0.7, max_tokens=100)
            response = chat([HumanMessage(content=prompt)])
            
            print(\"🔵 OpenAI (GPT-3.5 Turbo)\")
            print(f\"Response: {response.content}\")
            print(\"-\" * 50)
            providers_tested.append(\"OpenAI\")
        except Exception as e:
            logger.error(f\"OpenAI error: {e}\")
    
    # Anthropic (if available)
    anthropic_key = get_api_key('anthropic')
    if anthropic_key:
        try:
            from langchain.chat_models import ChatAnthropic
            chat = ChatAnthropic(anthropic_api_key=anthropic_key, temperature=0.7, max_tokens=100)
            response = chat([HumanMessage(content=prompt)])
            
            print(\"🟠 Anthropic (Claude)\")
            print(f\"Response: {response.content}\")
            print(\"-\" * 50)
            providers_tested.append(\"Anthropic\")
        except Exception as e:
            logger.warning(f\"Anthropic not available: {e}\")
    
    # Google (if available)
    google_key = get_api_key('google')
    if google_key:
        try:
            from langchain.chat_models import ChatGoogleGenerativeAI
            chat = ChatGoogleGenerativeAI(google_api_key=google_key, temperature=0.7, max_output_tokens=100)
            response = chat([HumanMessage(content=prompt)])
            
            print(\"🔴 Google (Gemini)\")
            print(f\"Response: {response.content}\")
            print(\"-\" * 50)
            providers_tested.append(\"Google\")
        except Exception as e:
            logger.warning(f\"Google not available: {e}\")
    
    if not providers_tested:
        logger.warning(\"⚠️ No providers available. Set up API keys to test multiple providers.\")
    else:
        logger.info(f\"✅ Tested providers: {', '.join(providers_tested)}\")


def main():
    \"\"\"Main function demonstrating model concepts.\"\"\"
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info(\"🚀 Starting LLM Models Comparison\")
    
    try:
        # Run all demonstrations
        compare_completion_models()
        compare_chat_models()
        demonstrate_temperature_effects()
        demonstrate_max_tokens()
        demonstrate_multiple_providers()
        
        print(\"\n💡 Key Takeaways:\")
        print(\"1. Temperature controls creativity (0.0 = deterministic, 1.0 = creative)\")
        print(\"2. Max tokens limits response length\")
        print(\"3. Different models have different strengths\")
        print(\"4. Chat models are better for conversations\")
        print(\"5. Completion models are good for text generation\")
        
        logger.info(\"✅ LLM Models Comparison completed successfully!\")
        
    except Exception as e:
        logger.error(f\"❌ Error occurred: {e}\")
        logger.info(\"💡 Make sure you have valid API keys\")


if __name__ == \"__main__\":
    main()