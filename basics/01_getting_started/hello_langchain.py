#!/usr/bin/env python3
\"\"\"
Hello LangChain - Your First LangChain Application

This is the simplest possible LangChain example to get you started.
It demonstrates:
1. Loading environment variables
2. Creating an LLM instance
3. Making a basic query
4. Handling responses

Prerequisites:
- OpenAI API key in .env file
- pip install -r requirements.txt
\"\"\"

import os
import sys
from pathlib import Path

# Add utils to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import setup_logging, get_logger, get_api_key
from langchain.llms import OpenAI
from langchain.schema import LLMResult


def main():
    \"\"\"Main function demonstrating basic LangChain usage.\"\"\"
    # Set up logging
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info(\"🚀 Starting Hello LangChain example\")
    
    # Check if API key is available
    api_key = get_api_key('openai')
    if not api_key:
        logger.error(\"❌ OpenAI API key not found. Please check your .env file.\")
        logger.info(\"💡 Copy .env.example to .env and add your OpenAI API key\")
        return
    
    logger.info(\"✅ API key found\")
    
    try:
        # Create an LLM instance
        logger.info(\"🔧 Creating OpenAI LLM instance...\")
        llm = OpenAI(
            temperature=0.7,  # Controls randomness (0.0 = deterministic, 1.0 = very creative)
            max_tokens=100,   # Limit response length
            openai_api_key=api_key
        )
        
        # Make a simple query
        prompt = \"What is LangChain and why is it useful?\"
        logger.info(f\"📝 Sending prompt: {prompt}\")
        
        response = llm(prompt)
        
        # Display results
        logger.info(\"🎉 Response received!\")
        print(\"\n\" + \"=\"*50)
        print(\"PROMPT:\")
        print(prompt)
        print(\"\nRESPONSE:\")
        print(response)
        print(\"=\"*50 + \"\n\")
        
        # Demonstrate batch processing
        logger.info(\"🔄 Demonstrating batch processing...\")
        prompts = [
            \"What is machine learning?\",
            \"Explain artificial intelligence briefly.\",
            \"What are large language models?\"
        ]
        
        responses = llm.generate(prompts)
        
        print(\"BATCH RESPONSES:\")
        for i, generation in enumerate(responses.generations):
            print(f\"\n{i+1}. {prompts[i]}\")
            print(f\"   → {generation[0].text.strip()}\")
        
        logger.info(\"✅ Hello LangChain example completed successfully!\")
        
    except Exception as e:
        logger.error(f\"❌ Error occurred: {e}\")
        logger.info(\"💡 Make sure your API key is valid and you have internet connection\")
        return


if __name__ == \"__main__\":
    main()