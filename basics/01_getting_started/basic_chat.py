#!/usr/bin/env python3
\"\"\"
Basic Chat Models - Introduction to Chat-based LLMs

This example demonstrates the difference between completion LLMs and chat models.
Chat models are designed for conversational interactions and use message-based inputs.

Key concepts:
1. Chat models vs completion models
2. Message types (System, Human, AI)
3. Conversation structure
4. Response handling
\"\"\"

import sys
from pathlib import Path

# Add utils to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import setup_logging, get_logger, get_api_key
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


def demonstrate_basic_chat():
    \"\"\"Demonstrate basic chat model usage.\"\"\"
    logger = get_logger(__name__)
    
    # Create a chat model
    chat = ChatOpenAI(
        temperature=0.7,
        model_name=\"gpt-3.5-turbo\",
        openai_api_key=get_api_key('openai')
    )
    
    logger.info(\"💬 Basic chat interaction\")
    
    # Single message
    message = HumanMessage(content=\"Hello! What can you help me with?\")
    response = chat([message])
    
    print(\"\n\" + \"=\"*50)
    print(\"BASIC CHAT INTERACTION\")
    print(\"Human:\", message.content)
    print(\"AI:\", response.content)
    print(\"=\"*50)
    
    return response


def demonstrate_system_message():
    \"\"\"Demonstrate using system messages to set behavior.\"\"\"
    logger = get_logger(__name__)
    
    chat = ChatOpenAI(
        temperature=0.3,
        model_name=\"gpt-3.5-turbo\",
        openai_api_key=get_api_key('openai')
    )
    
    logger.info(\"🎭 Using system message to set personality\")
    
    messages = [
        SystemMessage(content=\"You are a helpful assistant that explains things like you're talking to a 5-year-old. Use simple words and fun examples.\"),
        HumanMessage(content=\"What is machine learning?\")
    ]
    
    response = chat(messages)
    
    print(\"\n\" + \"=\"*50)
    print(\"WITH SYSTEM MESSAGE\")
    print(\"System:\", messages[0].content)
    print(\"Human:\", messages[1].content)
    print(\"AI:\", response.content)
    print(\"=\"*50)
    
    return response


def demonstrate_conversation():
    \"\"\"Demonstrate a multi-turn conversation.\"\"\"
    logger = get_logger(__name__)
    
    chat = ChatOpenAI(
        temperature=0.7,
        model_name=\"gpt-3.5-turbo\",
        openai_api_key=get_api_key('openai')
    )
    
    logger.info(\"🔄 Multi-turn conversation\")
    
    # Build a conversation history
    conversation = [
        SystemMessage(content=\"You are a helpful coding assistant.\"),
        HumanMessage(content=\"I'm learning Python. Can you help me understand functions?\"),
    ]
    
    # First response
    response1 = chat(conversation)
    conversation.append(AIMessage(content=response1.content))
    
    # Follow-up question
    conversation.append(HumanMessage(content=\"Can you show me a simple example?\"))
    response2 = chat(conversation)
    
    print(\"\n\" + \"=\"*50)
    print(\"MULTI-TURN CONVERSATION\")
    for i, msg in enumerate(conversation):
        if isinstance(msg, SystemMessage):
            print(f\"System: {msg.content}\")
        elif isinstance(msg, HumanMessage):
            print(f\"Human: {msg.content}\")
        elif isinstance(msg, AIMessage):
            print(f\"AI: {msg.content}\")
        print(\"-\" * 30)
    
    print(f\"AI: {response2.content}\")
    print(\"=\"*50)


def demonstrate_batch_chat():
    \"\"\"Demonstrate batch processing with chat models.\"\"\"
    logger = get_logger(__name__)
    
    chat = ChatOpenAI(
        temperature=0.5,
        model_name=\"gpt-3.5-turbo\",
        openai_api_key=get_api_key('openai')
    )
    
    logger.info(\"📦 Batch chat processing\")
    
    # Multiple conversations
    conversations = [
        [HumanMessage(content=\"What's the capital of France?\")],
        [HumanMessage(content=\"Explain photosynthesis in one sentence.\")],
        [HumanMessage(content=\"What's 15 * 23?\")]
    ]
    
    responses = chat.generate(conversations)
    
    print(\"\n\" + \"=\"*50)
    print(\"BATCH PROCESSING RESULTS\")
    for i, generation in enumerate(responses.generations):
        human_msg = conversations[i][0].content
        ai_response = generation[0].message.content
        print(f\"\n{i+1}. Human: {human_msg}\")
        print(f\"   AI: {ai_response}\")
    print(\"=\"*50)


def main():
    \"\"\"Main function demonstrating chat model concepts.\"\"\"
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info(\"🚀 Starting Basic Chat Models example\")
    
    # Check API key
    if not get_api_key('openai'):
        logger.error(\"❌ OpenAI API key not found. Please check your .env file.\")
        return
    
    try:
        # Run demonstrations
        demonstrate_basic_chat()
        demonstrate_system_message()
        demonstrate_conversation()
        demonstrate_batch_chat()
        
        logger.info(\"✅ Basic Chat Models example completed successfully!\")
        
        print(\"\n💡 Key Takeaways:\")
        print(\"1. Chat models use message-based inputs (System, Human, AI)\")
        print(\"2. System messages set the assistant's behavior\")
        print(\"3. Conversations maintain context through message history\")
        print(\"4. Batch processing can handle multiple conversations efficiently\")
        
    except Exception as e:
        logger.error(f\"❌ Error occurred: {e}\")
        logger.info(\"💡 Check your API key and internet connection\")


if __name__ == \"__main__\":
    main()