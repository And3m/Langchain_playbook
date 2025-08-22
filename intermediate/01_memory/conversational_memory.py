#!/usr/bin/env python3
\"\"\"
Conversational Memory - Maintaining Context in LangChain

This example demonstrates:
1. Different types of memory in LangChain
2. ConversationBufferMemory for basic conversation history
3. ConversationSummaryMemory for long conversations
4. ConversationBufferWindowMemory for recent context
5. Memory integration with chains
6. Custom memory implementations

Key concepts:
- Storing and retrieving conversation context
- Managing memory size and performance
- Different memory strategies for different use cases
- Integration with chat models and chains
\"\"\"

import sys
from pathlib import Path
from typing import Dict, Any, List

# Add utils to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import setup_logging, get_logger, get_api_key, timing_decorator
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory,
    ConversationEntityMemory
)
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMemory, HumanMessage, AIMessage


class CustomMemory(BaseMemory):
    \"\"\"Custom memory implementation that stores key facts.\"\"\"
    
    def __init__(self):
        self.facts: Dict[str, str] = {}
        self.conversation_history: List[str] = []
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        \"\"\"Save context from this conversation turn.\"\"\"
        human_input = inputs.get('input', '')
        ai_output = outputs.get('response', '')
        
        # Store the conversation
        self.conversation_history.append(f\"Human: {human_input}\")
        self.conversation_history.append(f\"AI: {ai_output}\")
        
        # Extract and store facts (simple keyword-based)
        self._extract_facts(human_input, ai_output)
    
    def _extract_facts(self, human_input: str, ai_output: str) -> None:
        \"\"\"Extract key facts from the conversation.\"\"\"
        # Simple fact extraction (in practice, use NLP or LLM)
        text = f\"{human_input} {ai_output}\".lower()
        
        # Look for name patterns
        if \"my name is\" in text:
            words = text.split()
            try:
                name_idx = words.index(\"is\") + 1
                if name_idx < len(words):
                    self.facts[\"user_name\"] = words[name_idx].capitalize()
            except ValueError:
                pass
        
        # Look for occupation patterns
        if \"i work as\" in text or \"i am a\" in text:
            for phrase in [\"i work as\", \"i am a\"]:
                if phrase in text:
                    words = text.split()
                    try:
                        start_idx = text.index(phrase) + len(phrase)
                        rest = text[start_idx:].strip().split()[0]
                        self.facts[\"occupation\"] = rest
                    except (ValueError, IndexError):
                        pass
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        \"\"\"Load memory variables for the prompt.\"\"\"
        # Recent conversation (last 4 exchanges)
        recent_history = self.conversation_history[-4:] if self.conversation_history else []
        
        # Format facts
        facts_text = \"\"
        if self.facts:
            facts_list = [f\"{k}: {v}\" for k, v in self.facts.items()]
            facts_text = f\"Known facts about user: {', '.join(facts_list)}\"
        
        return {
            \"history\": \"\n\".join(recent_history),
            \"facts\": facts_text
        }
    
    def clear(self) -> None:
        \"\"\"Clear the memory.\"\"\"
        self.facts.clear()
        self.conversation_history.clear()
    
    @property
    def memory_variables(self) -> List[str]:
        \"\"\"Variables this memory class provides.\"\"\"
        return [\"history\", \"facts\"]


def demonstrate_buffer_memory():
    \"\"\"Demonstrate basic conversation buffer memory.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping demonstration\")
        return
    
    logger.info(\"💭 Conversation Buffer Memory\")
    
    # Create chat model and memory
    llm = ChatOpenAI(openai_api_key=api_key, temperature=0.7, max_tokens=150)
    memory = ConversationBufferMemory()
    
    # Create conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    print(\"\n\" + \"=\"*60)
    print(\"CONVERSATION BUFFER MEMORY\")
    print(\"=\"*60)
    print(\"This memory stores the entire conversation history.\n\")
    
    # Simulate a conversation
    exchanges = [
        \"Hi! My name is Alice and I'm a software engineer.\",
        \"What programming languages do you recommend for beginners?\",
        \"Can you remember what I told you about myself?\",
        \"What was my name again?\"
    ]
    
    for i, user_input in enumerate(exchanges, 1):
        try:
            print(f\"👤 Turn {i}: {user_input}\")
            response = conversation.predict(input=user_input)
            print(f\"🤖 Response: {response}\")
            
            # Show memory content after each turn
            memory_vars = memory.load_memory_variables({})
            print(f\"📝 Memory: {memory_vars['history'][-100:]}...\")  # Show last 100 chars
            print(\"-\" * 40)
            
        except Exception as e:
            logger.error(f\"Error in turn {i}: {e}\")
    
    print(\"\n💡 Buffer Memory stores everything but can become very long.\")
    print(\"=\"*60)


def demonstrate_summary_memory():
    \"\"\"Demonstrate conversation summary memory.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping demonstration\")
        return
    
    logger.info(\"📄 Conversation Summary Memory\")
    
    # Create chat model and summary memory
    llm = ChatOpenAI(openai_api_key=api_key, temperature=0.3, max_tokens=150)
    memory = ConversationSummaryMemory(llm=llm)
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    print(\"\n\" + \"=\"*60)
    print(\"CONVERSATION SUMMARY MEMORY\")
    print(\"=\"*60)
    print(\"This memory creates summaries to manage long conversations.\n\")
    
    # Longer conversation to show summarization
    exchanges = [
        \"I'm planning a trip to Japan next month. I love Japanese culture and food.\",
        \"What are the must-visit places in Tokyo? I'm especially interested in traditional temples.\",
        \"How about food recommendations? I want to try authentic ramen and sushi.\",
        \"What should I know about Japanese customs and etiquette?\",
        \"Can you remind me what we discussed about my Japan trip?\"
    ]
    
    for i, user_input in enumerate(exchanges, 1):
        try:
            print(f\"👤 Turn {i}: {user_input}\")
            response = conversation.predict(input=user_input)
            print(f\"🤖 Response: {response[:100]}...\")  # Truncate for readability
            
            # Show current summary
            memory_vars = memory.load_memory_variables({})
            print(f\"📝 Summary: {memory_vars.get('history', 'No summary yet')}\")
            print(\"-\" * 40)
            
        except Exception as e:
            logger.error(f\"Error in turn {i}: {e}\")
    
    print(\"\n💡 Summary Memory keeps conversations manageable for long chats.\")
    print(\"=\"*60)


def demonstrate_window_memory():
    \"\"\"Demonstrate conversation buffer window memory.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping demonstration\")
        return
    
    logger.info(\"🪟 Conversation Buffer Window Memory\")
    
    # Create chat model and window memory (keep last 2 exchanges)
    llm = ChatOpenAI(openai_api_key=api_key, temperature=0.7, max_tokens=100)
    memory = ConversationBufferWindowMemory(k=2)  # Keep last 2 exchanges
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    print(\"\n\" + \"=\"*60)
    print(\"CONVERSATION BUFFER WINDOW MEMORY (k=2)\")
    print(\"=\"*60)
    print(\"This memory keeps only the last k conversation turns.\n\")
    
    # Multiple exchanges to show windowing effect
    exchanges = [
        \"I like pizza with pepperoni.\",
        \"I also enjoy chocolate ice cream.\",
        \"My favorite color is blue.\",
        \"I have a cat named Whiskers.\",
        \"What do you remember about my preferences?\"
    ]
    
    for i, user_input in enumerate(exchanges, 1):
        try:
            print(f\"👤 Turn {i}: {user_input}\")
            response = conversation.predict(input=user_input)
            print(f\"🤖 Response: {response}\")
            
            # Show what's in the window
            memory_vars = memory.load_memory_variables({})
            print(f\"📝 Window: {memory_vars['history']}\")
            print(\"-\" * 40)
            
        except Exception as e:
            logger.error(f\"Error in turn {i}: {e}\")
    
    print(\"\n💡 Window Memory balances context and efficiency.\")
    print(\"Only the most recent exchanges are remembered.\")
    print(\"=\"*60)


def demonstrate_custom_memory():
    \"\"\"Demonstrate custom memory implementation.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, showing custom memory structure only\")
        
        # Show the custom memory structure without LLM
        memory = CustomMemory()
        
        print(\"\n\" + \"=\"*60)
        print(\"CUSTOM MEMORY IMPLEMENTATION (Structure Demo)\")
        print(\"=\"*60)
        
        # Simulate saving context
        test_inputs = [{\"input\": \"Hi, my name is Bob and I work as a teacher\"}]
        test_outputs = [{\"response\": \"Nice to meet you, Bob! Teaching is a wonderful profession.\"}]
        
        for inp, out in zip(test_inputs, test_outputs):
            memory.save_context(inp, out)
        
        # Show memory variables
        variables = memory.load_memory_variables({})
        print(\"Custom Memory Variables:\")
        for key, value in variables.items():
            print(f\"{key}: {value}\")
        
        print(\"\nExtracted Facts:\")
        for fact_key, fact_value in memory.facts.items():
            print(f\"{fact_key}: {fact_value}\")
        
        return
    
    logger.info(\"🛠️ Custom Memory Implementation\")
    
    # Create custom memory and simple chain
    llm = ChatOpenAI(openai_api_key=api_key, temperature=0.7, max_tokens=150)
    memory = CustomMemory()
    
    # Custom prompt that uses our memory variables
    prompt = PromptTemplate(
        input_variables=[\"history\", \"facts\", \"input\"],
        template=\"\"\"You are a helpful assistant. Use the conversation history and known facts to provide personalized responses.
        
{facts}

Recent conversation:
{history}

Human: {input}
Assistant:\"\"\"
    )
    
    print(\"\n\" + \"=\"*60)
    print(\"CUSTOM MEMORY IMPLEMENTATION\")
    print(\"=\"*60)
    print(\"This memory extracts and stores key facts about the user.\n\")
    
    exchanges = [
        \"Hello! My name is Charlie and I work as a data scientist.\",
        \"I'm working on a machine learning project.\",
        \"What do you know about me so far?\"
    ]
    
    for i, user_input in enumerate(exchanges, 1):
        try:
            # Load memory variables
            memory_vars = memory.load_memory_variables({\"input\": user_input})
            
            # Format prompt with memory
            formatted_prompt = prompt.format(
                input=user_input,
                **memory_vars
            )
            
            print(f\"👤 Turn {i}: {user_input}\")
            
            # Get LLM response
            response = llm.predict(formatted_prompt)
            print(f\"🤖 Response: {response}\")
            
            # Save context to memory
            memory.save_context(
                {\"input\": user_input},
                {\"response\": response}
            )
            
            # Show extracted facts
            print(f\"📝 Extracted Facts: {memory.facts}\")
            print(\"-\" * 40)
            
        except Exception as e:
            logger.error(f\"Error in turn {i}: {e}\")
    
    print(\"\n💡 Custom Memory allows specialized information extraction.\")
    print(\"=\"*60)


def demonstrate_memory_comparison():
    \"\"\"Compare different memory types side by side.\"\"\"
    logger = get_logger(__name__)
    logger.info(\"⚖️ Memory Types Comparison\")
    
    print(\"\n\" + \"=\"*70)
    print(\"MEMORY TYPES COMPARISON\")
    print(\"=\"*70)
    
    memory_types = {
        \"Buffer Memory\": {
            \"description\": \"Stores entire conversation history\",
            \"pros\": [\"Complete context\", \"Simple to implement\", \"Perfect recall\"],
            \"cons\": [\"Can become very long\", \"Expensive for long conversations\", \"Token limit issues\"],
            \"best_for\": \"Short to medium conversations, detailed context needed\"
        },
        \"Summary Memory\": {
            \"description\": \"Creates summaries of conversation history\",
            \"pros\": [\"Handles long conversations\", \"Efficient token usage\", \"Maintains key info\"],
            \"cons\": [\"May lose details\", \"Requires LLM calls for summaries\", \"Summary quality varies\"],
            \"best_for\": \"Long conversations, efficiency important\"
        },
        \"Window Memory\": {
            \"description\": \"Keeps only recent conversation turns\",
            \"pros\": [\"Fixed size\", \"Fast performance\", \"Predictable costs\"],
            \"cons\": [\"Loses older context\", \"May miss important info\", \"Fixed window size\"],
            \"best_for\": \"Real-time chat, cost control needed\"
        },
        \"Custom Memory\": {
            \"description\": \"Application-specific memory logic\",
            \"pros\": [\"Tailored to needs\", \"Efficient storage\", \"Domain-specific\"],
            \"cons\": [\"Development time\", \"Maintenance needed\", \"Complexity\"],
            \"best_for\": \"Specialized applications, specific requirements\"
        }
    }
    
    for memory_name, details in memory_types.items():
        print(f\"\n📋 {memory_name}\")
        print(f\"Description: {details['description']}\")
        print(f\"✅ Pros: {', '.join(details['pros'])}\")
        print(f\"❌ Cons: {', '.join(details['cons'])}\")
        print(f\"🎯 Best for: {details['best_for']}\")
        print(\"-\" * 50)
    
    print(\"\n💡 Choosing the Right Memory Type:\")
    guidelines = [
        \"Consider conversation length and context needs\",
        \"Balance performance with context preservation\",
        \"Think about cost implications (token usage)\",
        \"Evaluate domain-specific requirements\",
        \"Test with realistic conversation flows\",
        \"Monitor memory performance in production\"
    ]
    
    for guideline in guidelines:
        print(f\"• {guideline}\")
    
    print(\"=\"*70)


def main():
    \"\"\"Main function demonstrating memory concepts.\"\"\"
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info(\"🚀 Starting Conversational Memory Demonstration\")
    
    try:
        # Run all demonstrations
        demonstrate_buffer_memory()
        demonstrate_summary_memory()
        demonstrate_window_memory()
        demonstrate_custom_memory()
        demonstrate_memory_comparison()
        
        print(\"\n🎯 Memory Key Takeaways:\")
        print(\"1. Memory enables context-aware conversations\")
        print(\"2. Different memory types serve different needs\")
        print(\"3. Balance context preservation with performance\")
        print(\"4. Custom memory enables specialized applications\")
        print(\"5. Consider token costs and conversation length\")
        print(\"6. Memory choice impacts user experience\")
        
        logger.info(\"✅ Conversational Memory demonstration completed successfully!\")
        
    except Exception as e:
        logger.error(f\"❌ Error occurred: {e}\")
        logger.info(\"💡 Check your API keys and internet connection\")


if __name__ == \"__main__\":
    main()