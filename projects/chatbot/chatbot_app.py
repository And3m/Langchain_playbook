#!/usr/bin/env python3
\"\"\"
Chatbot Project - Conversational AI with Memory and Tools

This project demonstrates building a complete chatbot application with:
1. Conversational memory for context awareness
2. Multiple tool integrations
3. Personality and role customization
4. Error handling and fallback responses
5. Streaming responses for better UX
6. Usage monitoring and analytics

Features:
- Multi-turn conversations with memory
- Weather information tool
- Calculator for math operations
- Web search capabilities
- Customizable personality
- Real-time streaming responses
- Cost and usage tracking
\"\"\"

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio
import json
from datetime import datetime
import os

# Add utils to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import setup_logging, get_logger, get_api_key
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool, DuckDuckGoSearchRun
from langchain.memory import ConversationSummaryBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import HumanMessage, AIMessage
from pydantic import BaseModel, Field


class WeatherTool(BaseTool):
    \"\"\"Tool to get weather information.\"\"\"
    
    name = \"weather\"
    description = \"Get current weather information for a city. Input should be a city name.\"
    
    def _run(self, city: str) -> str:
        \"\"\"Get weather for a city (mock implementation for demo).\"\"\"
        # In a real implementation, you'd use a weather API
        mock_weather = {
            \"new york\": \"Sunny, 72°F (22°C), light breeze\",
            \"london\": \"Cloudy, 59°F (15°C), chance of rain\",
            \"tokyo\": \"Partly cloudy, 68°F (20°C), humid\",
            \"paris\": \"Clear, 65°F (18°C), pleasant\",
            \"sydney\": \"Sunny, 75°F (24°C), perfect beach weather\",
            \"moscow\": \"Cold, 32°F (0°C), snowing\",
            \"mumbai\": \"Hot and humid, 86°F (30°C), monsoon season\",
            \"cairo\": \"Very hot, 95°F (35°C), dry and sunny\"
        }
        
        city_lower = city.lower().strip()
        weather = mock_weather.get(city_lower, f\"Weather data not available for {city}. Try major cities like New York, London, Tokyo, etc.\")
        
        return f\"🌤️ Weather in {city.title()}: {weather}\"
    
    async def _arun(self, city: str) -> str:
        return self._run(city)


class CalculatorTool(BaseTool):
    \"\"\"Enhanced calculator tool with more functions.\"\"\"
    
    name = \"calculator\"
    description = \"\"\"Perform mathematical calculations. Supports:
    - Basic operations: +, -, *, /
    - Powers: ** or pow(x,y)
    - Square root: sqrt(x)
    - Trigonometry: sin(x), cos(x), tan(x)
    - Logarithms: log(x), log10(x)
    Input should be a valid mathematical expression.
    \"\"\"
    
    def _run(self, expression: str) -> str:
        \"\"\"Evaluate mathematical expression safely.\"\"\"
        try:
            import math
            
            # Define safe functions
            safe_dict = {
                \"__builtins__\": {},
                \"abs\": abs, \"round\": round, \"min\": min, \"max\": max,
                \"sum\": sum, \"pow\": pow,
                \"sqrt\": math.sqrt, \"sin\": math.sin, \"cos\": math.cos,
                \"tan\": math.tan, \"log\": math.log, \"log10\": math.log10,
                \"pi\": math.pi, \"e\": math.e
            }
            
            # Clean and validate expression
            expression = expression.strip()
            
            # Check for dangerous operations
            forbidden = ['import', 'exec', 'eval', 'open', 'file', '__']
            if any(word in expression.lower() for word in forbidden):
                return \"❌ Error: Invalid characters or operations in expression\"
            
            # Evaluate safely
            result = eval(expression, safe_dict)
            
            # Format result nicely
            if isinstance(result, float):
                if result.is_integer():
                    result = int(result)
                else:
                    result = round(result, 6)
            
            return f\"🔢 {expression} = {result}\"
            
        except ZeroDivisionError:
            return \"❌ Error: Division by zero\"
        except ValueError as e:
            return f\"❌ Error: Invalid value - {e}\"
        except Exception as e:
            return f\"❌ Error: {e}\"
    
    async def _arun(self, expression: str) -> str:
        return self._run(expression)


class TimeInfoTool(BaseTool):
    \"\"\"Tool to get current time and date information.\"\"\"
    
    name = \"time_info\"
    description = \"Get current date, time, or both. Can also get time for different timezones.\"
    
    def _run(self, query: str = \"\") -> str:
        \"\"\"Get time information.\"\"\"
        from datetime import datetime
        import pytz
        
        try:
            now = datetime.now()
            
            # Check for timezone requests
            query_lower = query.lower()
            
            if \"utc\" in query_lower or \"gmt\" in query_lower:
                utc_time = datetime.now(pytz.UTC)
                return f\"🕐 UTC Time: {utc_time.strftime('%Y-%m-%d %H:%M:%S UTC')}\"
            
            elif \"timezone\" in query_lower or \"time zone\" in query_lower:
                return f\"🌍 Available timezones: UTC, US/Eastern, US/Pacific, Europe/London, Asia/Tokyo, etc.\"
            
            elif \"date\" in query_lower:
                return f\"📅 Current date: {now.strftime('%A, %B %d, %Y')}\"
            
            elif \"time\" in query_lower:
                return f\"🕐 Current time: {now.strftime('%H:%M:%S')}\"
            
            else:
                return f\"📅🕐 Current date and time: {now.strftime('%A, %B %d, %Y at %H:%M:%S')}\"
                
        except Exception as e:
            return f\"❌ Error getting time information: {e}\"
    
    async def _arun(self, query: str = \"\") -> str:
        return self._run(query)


class ChatbotAnalytics(BaseCallbackHandler):
    \"\"\"Callback to track chatbot usage and performance.\"\"\"
    
    def __init__(self):
        self.conversation_count = 0
        self.total_tokens = 0
        self.tool_usage = {}
        self.conversation_lengths = []
        self.start_time = datetime.now()
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        \"\"\"Track tool usage.\"\"\"
        tool_name = serialized.get('name', 'unknown')
        self.tool_usage[tool_name] = self.tool_usage.get(tool_name, 0) + 1
    
    def on_llm_end(self, response, **kwargs) -> None:
        \"\"\"Track token usage.\"\"\"
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            if token_usage:
                self.total_tokens += token_usage.get('total_tokens', 0)
    
    def new_conversation(self, length: int):
        \"\"\"Record a new conversation.\"\"\"
        self.conversation_count += 1
        self.conversation_lengths.append(length)
    
    def get_analytics(self) -> Dict[str, Any]:
        \"\"\"Get usage analytics.\"\"\"
        uptime = datetime.now() - self.start_time
        avg_length = sum(self.conversation_lengths) / max(len(self.conversation_lengths), 1)
        
        return {
            \"uptime_minutes\": round(uptime.total_seconds() / 60, 2),
            \"total_conversations\": self.conversation_count,
            \"total_tokens\": self.total_tokens,
            \"average_conversation_length\": round(avg_length, 2),
            \"tool_usage\": self.tool_usage,
            \"conversations_per_hour\": round(self.conversation_count / max(uptime.total_seconds() / 3600, 1), 2)
        }


class PersonalityChatbot:
    \"\"\"Configurable chatbot with personality and tools.\"\"\"
    
    def __init__(self, personality: str = \"helpful\", api_key: Optional[str] = None):
        self.personality = personality
        self.api_key = api_key or get_api_key('openai')
        self.analytics = ChatbotAnalytics()
        self.logger = get_logger(self.__class__.__name__)
        
        if not self.api_key:
            raise ValueError(\"OpenAI API key is required\")
        
        self._setup_chatbot()
    
    def _setup_chatbot(self):
        \"\"\"Initialize the chatbot components.\"\"\"
        # Create LLM with analytics callback
        self.llm = ChatOpenAI(
            openai_api_key=self.api_key,
            temperature=0.7,
            max_tokens=300,
            streaming=True,
            callbacks=[self.analytics]
        )
        
        # Create tools
        tools = [
            WeatherTool(),
            CalculatorTool(),
            TimeInfoTool(),
            DuckDuckGoSearchRun()
        ]
        
        # Create memory with personality-aware summary
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=1000,
            memory_key=\"chat_history\",
            return_messages=True
        )
        
        # Create agent with personality
        system_message = self._get_personality_prompt()
        
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=False,
            agent_kwargs={
                \"system_message\": system_message
            },
            callbacks=[self.analytics],
            handle_parsing_errors=True
        )
    
    def _get_personality_prompt(self) -> str:
        \"\"\"Get personality-specific system prompt.\"\"\"
        personalities = {
            \"helpful\": \"You are a helpful and friendly AI assistant. You're knowledgeable, patient, and always try to provide useful information. You use tools when needed to give accurate answers.\",
            
            \"casual\": \"You're a casual, laid-back AI buddy. You speak in a relaxed, friendly tone and use emojis occasionally. You're helpful but keep things light and conversational.\",
            
            \"professional\": \"You are a professional AI assistant. You communicate in a formal, business-appropriate manner. You're efficient, accurate, and focus on providing clear, actionable information.\",
            
            \"enthusiastic\": \"You're an enthusiastic and energetic AI assistant! You're excited to help and always positive. You use exclamation points and encouraging language while being genuinely helpful.\",
            
            \"academic\": \"You are a scholarly AI assistant with a focus on accuracy and detailed explanations. You provide well-reasoned responses and cite sources when appropriate. You enjoy discussing complex topics.\"
        }
        
        base_prompt = personalities.get(self.personality, personalities[\"helpful\"])
        
        return f\"\"\"{base_prompt}
        
You have access to several tools:
- Weather information for cities
- Calculator for math operations
- Current time and date information
- Web search for current information

Use these tools when users ask relevant questions. Always be helpful and engaging in your responses.\"\"\"
    
    def chat(self, message: str) -> str:
        \"\"\"Process a chat message and return response.\"\"\"
        try:
            self.logger.info(f\"Processing message: {message[:50]}...\")
            response = self.agent.run(input=message)
            return response
        except Exception as e:
            self.logger.error(f\"Error in chat: {e}\")
            return f\"I apologize, but I encountered an error: {e}. Please try rephrasing your question.\"
    
    async def chat_async(self, message: str) -> str:
        \"\"\"Async version of chat for better performance.\"\"\"
        try:
            self.logger.info(f\"Processing async message: {message[:50]}...\")
            response = await self.agent.arun(input=message)
            return response
        except Exception as e:
            self.logger.error(f\"Error in async chat: {e}\")
            return f\"I apologize, but I encountered an error: {e}. Please try rephrasing your question.\"
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        \"\"\"Get formatted conversation history.\"\"\"
        messages = self.memory.chat_memory.messages
        history = []
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({\"role\": \"user\", \"content\": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({\"role\": \"assistant\", \"content\": msg.content})
        
        return history
    
    def clear_memory(self):
        \"\"\"Clear conversation memory.\"\"\"
        self.memory.clear()
        self.logger.info(\"Conversation memory cleared\")
    
    def save_conversation(self, filename: str = None):
        \"\"\"Save conversation to file.\"\"\"
        if not filename:
            timestamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")
            filename = f\"conversation_{timestamp}.json\"
        
        conversation_data = {
            \"timestamp\": datetime.now().isoformat(),
            \"personality\": self.personality,
            \"conversation\": self.get_conversation_history(),
            \"analytics\": self.analytics.get_analytics()
        }
        
        with open(filename, 'w') as f:
            json.dump(conversation_data, f, indent=2)
        
        self.logger.info(f\"Conversation saved to {filename}\")
        return filename


def demonstrate_basic_chatbot():
    \"\"\"Demonstrate basic chatbot functionality.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping demonstration\")
        return
    
    logger.info(\"🤖 Basic Chatbot Demonstration\")
    
    print(\"\n\" + \"=\"*60)
    print(\"BASIC CHATBOT WITH TOOLS AND MEMORY\")
    print(\"=\"*60)
    
    try:
        # Create chatbot
        chatbot = PersonalityChatbot(personality=\"helpful\")
        
        # Test conversation
        test_messages = [
            \"Hi! What's the weather like in Tokyo?\",
            \"Can you calculate 15 * 23 + 7?\",
            \"What time is it?\",
            \"What did we talk about regarding Tokyo?\",
            \"Search for the latest news about artificial intelligence\"
        ]
        
        print(\"🗨️ Starting conversation...\n\")
        
        for i, message in enumerate(test_messages, 1):
            print(f\"👤 User: {message}\")
            response = chatbot.chat(message)
            print(f\"🤖 Bot: {response}\")
            print(\"-\" * 40)
        
        # Show analytics
        analytics = chatbot.analytics.get_analytics()
        print(f\"\n📊 Analytics:\")
        print(f\"Total tokens used: {analytics['total_tokens']}\")
        print(f\"Tools used: {analytics['tool_usage']}\")
        
        # Save conversation
        filename = chatbot.save_conversation()
        print(f\"\n💾 Conversation saved to: {filename}\")
        
    except Exception as e:
        logger.error(f\"Error in basic chatbot demo: {e}\")
    
    print(\"\n💡 The chatbot remembers context and can use multiple tools.\")
    print(\"=\"*60)


def demonstrate_personality_variations():
    \"\"\"Demonstrate different chatbot personalities.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping demonstration\")
        return
    
    logger.info(\"🎭 Personality Variations\")
    
    print(\"\n\" + \"=\"*60)
    print(\"CHATBOT PERSONALITY VARIATIONS\")
    print(\"=\"*60)
    
    personalities = [\"helpful\", \"casual\", \"professional\", \"enthusiastic\"]
    test_question = \"Can you tell me about machine learning?\"
    
    print(f\"Question: {test_question}\n\")
    
    for personality in personalities:
        try:
            print(f\"🎭 {personality.title()} Personality:\")
            chatbot = PersonalityChatbot(personality=personality)
            response = chatbot.chat(test_question)
            print(f\"Response: {response[:200]}...\")
            print(\"-\" * 40)
        except Exception as e:
            logger.error(f\"Error with {personality} personality: {e}\")
    
    print(\"\n💡 Different personalities provide varied interaction styles.\")
    print(\"=\"*60)


async def demonstrate_async_chatbot():
    \"\"\"Demonstrate async chatbot for better performance.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping demonstration\")
        return
    
    logger.info(\"⚡ Async Chatbot Performance\")
    
    print(\"\n\" + \"=\"*60)
    print(\"ASYNC CHATBOT FOR HIGH PERFORMANCE\")
    print(\"=\"*60)
    
    try:
        chatbot = PersonalityChatbot(personality=\"casual\")
        
        # Multiple concurrent questions
        questions = [
            \"What's 2+2?\",
            \"What's the weather in Paris?\",
            \"What time is it?\",
            \"Calculate sqrt(144)\"
        ]
        
        print(\"🚀 Processing multiple questions concurrently...\n\")
        
        import time
        start_time = time.time()
        
        # Process all questions concurrently
        tasks = [chatbot.chat_async(q) for q in questions]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        duration = time.time() - start_time
        
        # Display results
        for i, (question, response) in enumerate(zip(questions, responses)):
            print(f\"❓ Q{i+1}: {question}\")
            if isinstance(response, str):
                print(f\"✅ A{i+1}: {response[:100]}...\")
            else:
                print(f\"❌ A{i+1}: Error - {response}\")
            print()
        
        print(f\"⚡ All questions processed in {duration:.2f} seconds\")
        
    except Exception as e:
        logger.error(f\"Error in async chatbot demo: {e}\")
    
    print(\"\n💡 Async processing enables handling multiple users concurrently.\")
    print(\"=\"*60)


def demonstrate_chatbot_features():
    \"\"\"Demonstrate advanced chatbot features.\"\"\"
    logger = get_logger(__name__)
    logger.info(\"🔧 Advanced Chatbot Features\")
    
    print(\"\n\" + \"=\"*70)
    print(\"ADVANCED CHATBOT FEATURES AND CAPABILITIES\")
    print(\"=\"*70)
    
    features = {
        \"Conversational Memory\": {
            \"description\": \"Maintains context across multiple turns\",
            \"implementation\": \"ConversationSummaryBufferMemory\",
            \"benefits\": [\"Context awareness\", \"Natural conversation flow\", \"Reference to earlier topics\"]
        },
        \"Tool Integration\": {
            \"description\": \"Access to external tools and APIs\",
            \"implementation\": \"Custom tools with BaseTool\",
            \"benefits\": [\"Real-time information\", \"Calculations\", \"External service access\"]
        },
        \"Personality Customization\": {
            \"description\": \"Different conversation styles and tones\",
            \"implementation\": \"Customizable system prompts\",
            \"benefits\": [\"Brand consistency\", \"User preference matching\", \"Context-appropriate responses\"]
        },
        \"Error Handling\": {
            \"description\": \"Graceful handling of errors and edge cases\",
            \"implementation\": \"Try-catch blocks with fallbacks\",
            \"benefits\": [\"Robust operation\", \"User-friendly error messages\", \"Continued conversation\"]
        },
        \"Analytics & Monitoring\": {
            \"description\": \"Usage tracking and performance monitoring\",
            \"implementation\": \"Custom callback handlers\",
            \"benefits\": [\"Usage insights\", \"Performance optimization\", \"Cost tracking\"]
        },
        \"Async Support\": {
            \"description\": \"Concurrent request handling\",
            \"implementation\": \"Async/await patterns\",
            \"benefits\": [\"Better performance\", \"Scalability\", \"Resource efficiency\"]
        }
    }
    
    for feature_name, details in features.items():
        print(f\"\n🔧 {feature_name}:\")
        print(f\"   Description: {details['description']}\")
        print(f\"   Implementation: {details['implementation']}\")
        print(f\"   Benefits:\")
        for benefit in details['benefits']:
            print(f\"     • {benefit}\")
        print(\"-\" * 50)
    
    print(\"\n🚀 Production Considerations:\")
    considerations = [
        \"Rate limiting to prevent abuse\",
        \"User authentication and session management\",
        \"Conversation persistence across sessions\",
        \"Load balancing for multiple instances\",
        \"Monitoring and alerting for issues\",
        \"Content filtering and safety measures\",
        \"API key rotation and security\",
        \"Cost optimization and budgeting\"
    ]
    
    for consideration in considerations:
        print(f\"   • {consideration}\")
    
    print(\"\n📈 Scaling Strategies:\")
    strategies = [
        \"Horizontal scaling with multiple instances\",
        \"Caching frequently asked questions\",
        \"Async processing for concurrent users\",
        \"Database integration for conversation history\",
        \"CDN for static assets and responses\",
        \"Queue systems for high-volume requests\"
    ]
    
    for strategy in strategies:
        print(f\"   • {strategy}\")
    
    print(\"=\"*70)


async def main():
    \"\"\"Main function to demonstrate chatbot project.\"\"\"
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info(\"🚀 Starting Chatbot Project Demonstration\")
    
    try:
        # Run demonstrations
        demonstrate_basic_chatbot()
        demonstrate_personality_variations()
        await demonstrate_async_chatbot()
        demonstrate_chatbot_features()
        
        print(\"\n🎯 Chatbot Project Key Takeaways:\")
        print(\"1. Memory enables natural, context-aware conversations\")
        print(\"2. Tools extend chatbot capabilities beyond text generation\")
        print(\"3. Personality customization creates engaging user experiences\")
        print(\"4. Async operations enable scalable, high-performance bots\")
        print(\"5. Analytics provide insights for optimization\")
        print(\"6. Proper error handling ensures robust operation\")
        
        logger.info(\"✅ Chatbot project demonstration completed successfully!\")
        
    except Exception as e:
        logger.error(f\"❌ Error occurred: {e}\")
        logger.info(\"💡 Check your API keys and dependencies\")


if __name__ == \"__main__\":
    # Run the async main function
    asyncio.run(main())