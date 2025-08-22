#!/usr/bin/env python3
\"\"\"
LangChain Agents - Tool-Using AI Systems

This example demonstrates:
1. Basic agent concepts and architecture
2. Built-in tools (search, calculator, etc.)
3. Custom tool creation
4. Agent types and selection strategies
5. Agent execution and reasoning
6. Error handling and safety considerations

Key concepts:
- Agents that can use tools to accomplish tasks
- Reasoning and action selection
- Tool integration and execution
- Agent memory and planning
\"\"\"

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import requests
from datetime import datetime

# Add utils to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import setup_logging, get_logger, get_api_key, timing_decorator
from langchain.chat_models import ChatOpenAI
from langchain.agents import (
    initialize_agent,
    AgentType,
    Tool,
    AgentExecutor,
    create_react_agent
)
from langchain.tools import (
    BaseTool,
    DuckDuckGoSearchRun,
    ShellTool
)
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field


class CalculatorTool(BaseTool):
    \"\"\"Custom tool for basic mathematical calculations.\"\"\"
    
    name = \"calculator\"
    description = \"Useful for mathematical calculations. Input should be a valid mathematical expression.\"
    
    def _run(self, query: str) -> str:
        \"\"\"Execute the calculation.\"\"\"
        try:
            # Safe evaluation of mathematical expressions
            # Only allow basic math operations
            allowed_chars = set('0123456789+-*/().% ')
            if not all(c in allowed_chars for c in query):
                return \"Error: Invalid characters in mathematical expression\"
            
            # Evaluate the expression
            result = eval(query)
            return f\"The result is: {result}\"
        except Exception as e:
            return f\"Error in calculation: {str(e)}\"
    
    async def _arun(self, query: str) -> str:
        \"\"\"Async version of the tool.\"\"\"
        return self._run(query)


class WeatherTool(BaseTool):
    \"\"\"Custom tool to get weather information (mock implementation).\"\"\"
    
    name = \"weather\"
    description = \"Get current weather information for a city. Input should be a city name.\"
    
    def _run(self, city: str) -> str:
        \"\"\"Get weather for a city (mock implementation).\"\"\"
        # This is a mock implementation
        # In practice, you'd use a real weather API
        mock_weather = {
            \"new york\": \"Sunny, 72°F (22°C)\",
            \"london\": \"Cloudy, 59°F (15°C)\",
            \"tokyo\": \"Rainy, 68°F (20°C)\",
            \"paris\": \"Partly cloudy, 65°F (18°C)\",
            \"sydney\": \"Sunny, 75°F (24°C)\"
        }
        
        city_lower = city.lower().strip()
        weather = mock_weather.get(city_lower, \"Weather data not available for this city\")
        
        return f\"Weather in {city}: {weather}\"
    
    async def _arun(self, city: str) -> str:
        \"\"\"Async version of the tool.\"\"\"
        return self._run(city)


class DateTimeTool(BaseTool):
    \"\"\"Tool to get current date and time information.\"\"\"
    
    name = \"datetime\"
    description = \"Get current date and time information.\"
    
    def _run(self, query: str = \"\") -> str:
        \"\"\"Get current date and time.\"\"\"
        now = datetime.now()
        return f\"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}\"
    
    async def _arun(self, query: str = \"\") -> str:
        \"\"\"Async version of the tool.\"\"\"
        return self._run(query)


class TextAnalysisTool(BaseTool):
    \"\"\"Tool for basic text analysis.\"\"\"
    
    name = \"text_analysis\"
    description = \"Analyze text and provide statistics like word count, character count, etc.\"
    
    def _run(self, text: str) -> str:
        \"\"\"Analyze the provided text.\"\"\"
        if not text:
            return \"Please provide text to analyze\"
        
        words = text.split()
        sentences = text.split('.')
        paragraphs = text.split('\n\n')
        
        analysis = {
            \"characters\": len(text),
            \"characters_no_spaces\": len(text.replace(' ', '')),
            \"words\": len(words),
            \"sentences\": len([s for s in sentences if s.strip()]),
            \"paragraphs\": len([p for p in paragraphs if p.strip()]),
            \"avg_words_per_sentence\": round(len(words) / max(len([s for s in sentences if s.strip()]), 1), 2)
        }
        
        result = \"Text Analysis Results:\n\"
        for key, value in analysis.items():
            result += f\"- {key.replace('_', ' ').title()}: {value}\n\"
        
        return result
    
    async def _arun(self, text: str) -> str:
        \"\"\"Async version of the tool.\"\"\"
        return self._run(text)


@timing_decorator
def demonstrate_basic_agent():
    \"\"\"Demonstrate basic agent with simple tools.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping demonstration\")
        return
    
    logger.info(\"🤖 Basic Agent with Tools\")
    
    # Create LLM
    llm = ChatOpenAI(openai_api_key=api_key, temperature=0.2, max_tokens=200)
    
    # Create tools
    tools = [
        CalculatorTool(),
        WeatherTool(),
        DateTimeTool(),
        TextAnalysisTool()
    ]
    
    # Create agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    
    print(\"\n\" + \"=\"*70)
    print(\"BASIC AGENT WITH CUSTOM TOOLS\")
    print(\"=\"*70)
    print(\"Agent has access to: calculator, weather, datetime, text_analysis\n\")
    
    # Test queries
    queries = [
        \"What's 25 * 47?\",
        \"What's the weather like in Tokyo?\",
        \"What's the current date and time?\",
        \"Analyze this text: 'LangChain is a powerful framework for building AI applications. It provides tools for working with language models and creating complex workflows.'\"
    ]
    
    for i, query in enumerate(queries, 1):
        try:
            print(f\"\n👤 Query {i}: {query}\")
            response = agent.run(query)
            print(f\"🤖 Final Answer: {response}\")
            print(\"-\" * 50)
        except Exception as e:
            logger.error(f\"Error processing query {i}: {e}\")
    
    print(\"\n💡 The agent uses ReAct (Reasoning + Acting) to solve problems.\")
    print(\"It reasons about what tools to use and how to use them.\")
    print(\"=\"*70)


def demonstrate_agent_with_search():
    \"\"\"Demonstrate agent with search capabilities.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping demonstration\")
        return
    
    logger.info(\"🔍 Agent with Search Tool\")
    
    # Create LLM
    llm = ChatOpenAI(openai_api_key=api_key, temperature=0.3, max_tokens=300)
    
    # Create tools including search
    tools = [
        DuckDuckGoSearchRun(),
        CalculatorTool(),
        DateTimeTool()
    ]
    
    # Create agent with memory
    memory = ConversationBufferMemory(memory_key=\"chat_history\", return_messages=True)
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )
    
    print(\"\n\" + \"=\"*70)
    print(\"AGENT WITH SEARCH CAPABILITIES\")
    print(\"=\"*70)
    print(\"Agent can search the web for current information\n\")
    
    # Queries that require search
    search_queries = [
        \"What's the latest news about artificial intelligence?\",
        \"Who won the most recent Nobel Prize in Physics and why?\",
        \"What's the current price of Bitcoin?\"
    ]
    
    for i, query in enumerate(search_queries, 1):
        try:
            print(f\"\n👤 Query {i}: {query}\")
            response = agent.run(query)
            print(f\"🤖 Final Answer: {response[:200]}...\")  # Truncate for readability
            print(\"-\" * 50)
        except Exception as e:
            logger.error(f\"Error processing search query {i}: {e}\")
            print(f\"⚠️ Search failed: {e}\")
    
    print(\"\n💡 Search agents can access real-time information.\")
    print(\"They combine reasoning with web search capabilities.\")
    print(\"=\"*70)


def demonstrate_agent_planning():
    \"\"\"Demonstrate agent planning and multi-step reasoning.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping demonstration\")
        return
    
    logger.info(\"🧠 Agent Planning and Multi-Step Reasoning\")
    
    # Create LLM
    llm = ChatOpenAI(openai_api_key=api_key, temperature=0.2, max_tokens=400)
    
    # Tools for planning scenario
    tools = [
        CalculatorTool(),
        DateTimeTool(),
        WeatherTool(),
        TextAnalysisTool()
    ]
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    
    print(\"\n\" + \"=\"*70)
    print(\"AGENT PLANNING AND MULTI-STEP REASONING\")
    print(\"=\"*70)
    
    # Complex multi-step query
    complex_query = \"\"\"
    I need to plan a day trip. First, check what time it is now. 
    Then, if I want to visit 3 places and spend 2 hours at each place, 
    plus 30 minutes travel time between each place, how long will the trip take? 
    Also, what's the weather like in New York for my trip?
    \"\"\"
    
    try:
        print(f\"👤 Complex Query: {complex_query.strip()}\")
        print(\"\n🧠 Agent Planning Process:\")
        response = agent.run(complex_query)
        print(f\"\n🎯 Final Plan: {response}\")
    except Exception as e:
        logger.error(f\"Error in planning demonstration: {e}\")
    
    print(\"\n💡 Agents can break down complex problems into steps.\")
    print(\"They use multiple tools to gather information and solve problems.\")
    print(\"=\"*70)


def demonstrate_agent_types():
    \"\"\"Demonstrate different agent types and their characteristics.\"\"\"
    logger = get_logger(__name__)
    logger.info(\"🎭 Agent Types Comparison\")
    
    print(\"\n\" + \"=\"*70)
    print(\"AGENT TYPES AND CHARACTERISTICS\")
    print(\"=\"*70)
    
    agent_types = {
        \"ZERO_SHOT_REACT_DESCRIPTION\": {
            \"description\": \"Uses ReAct framework for reasoning and acting\",
            \"best_for\": \"General problem solving, single-turn queries\",
            \"memory\": \"No built-in memory\",
            \"reasoning\": \"Step-by-step reasoning with tool usage\"
        },
        \"REACT_DOCSTORE\": {
            \"description\": \"ReAct agent optimized for document search\",
            \"best_for\": \"Document Q&A, information retrieval\",
            \"memory\": \"No built-in memory\",
            \"reasoning\": \"Search and lookup focused\"
        },
        \"CHAT_CONVERSATIONAL_REACT_DESCRIPTION\": {
            \"description\": \"Conversational agent with memory\",
            \"best_for\": \"Multi-turn conversations, context awareness\",
            \"memory\": \"Built-in conversation memory\",
            \"reasoning\": \"Context-aware reasoning\"
        },
        \"SELF_ASK_WITH_SEARCH\": {
            \"description\": \"Breaks down questions into sub-questions\",
            \"best_for\": \"Complex factual queries, research tasks\",
            \"memory\": \"No built-in memory\",
            \"reasoning\": \"Decomposition and search\"
        }
    }
    
    for agent_name, details in agent_types.items():
        print(f\"\n🤖 {agent_name}\")
        for key, value in details.items():
            print(f\"   {key.title()}: {value}\")
        print(\"-\" * 50)
    
    print(\"\n💡 Choosing the Right Agent Type:\")
    selection_guide = [
        \"Consider whether you need memory/conversation context\",
        \"Think about the complexity of reasoning required\",
        \"Evaluate the types of tools your agent will use\",
        \"Consider performance vs. capability trade-offs\",
        \"Test different types with your specific use case\"
    ]
    
    for tip in selection_guide:
        print(f\"• {tip}\")
    
    print(\"=\"*70)


def demonstrate_agent_safety():
    \"\"\"Demonstrate agent safety considerations and limitations.\"\"\"
    logger = get_logger(__name__)
    logger.info(\"🛡️ Agent Safety and Best Practices\")
    
    print(\"\n\" + \"=\"*70)
    print(\"AGENT SAFETY AND BEST PRACTICES\")
    print(\"=\"*70)
    
    safety_considerations = {
        \"Tool Security\": [
            \"Validate all tool inputs before execution\",
            \"Limit tool capabilities (e.g., file system access)\",
            \"Use sandboxed environments for code execution\",
            \"Implement timeouts for long-running operations\"
        ],
        \"Input Validation\": [
            \"Sanitize user inputs to prevent injection attacks\",
            \"Set reasonable limits on input length and complexity\",
            \"Validate data types and formats\",
            \"Handle edge cases and error conditions\"
        ],
        \"Error Handling\": [
            \"Gracefully handle tool failures\",
            \"Provide meaningful error messages\",
            \"Implement retry logic with exponential backoff\",
            \"Log errors for debugging and monitoring\"
        ],
        \"Resource Management\": [
            \"Set limits on token usage and API calls\",
            \"Implement rate limiting for expensive operations\",
            \"Monitor and control execution time\",
            \"Manage memory usage for long conversations\"
        ],
        \"Privacy and Data\": [
            \"Don't store sensitive information in memory\",
            \"Be careful with data passed to external tools\",
            \"Implement proper data encryption and access controls\",
            \"Follow data protection regulations\"
        ]
    }
    
    for category, practices in safety_considerations.items():
        print(f\"\n🔒 {category}:\")
        for practice in practices:
            print(f\"   • {practice}\")
    
    print(\"\n⚠️ Common Pitfalls to Avoid:\")
    pitfalls = [
        \"Giving agents unrestricted access to system functions\",
        \"Not validating tool outputs before using them\",
        \"Ignoring error handling in tool implementations\",
        \"Not setting appropriate timeouts and limits\",
        \"Assuming agent reasoning is always correct\",
        \"Not monitoring agent behavior in production\"
    ]
    
    for pitfall in pitfalls:
        print(f\"❌ {pitfall}\")
    
    print(\"\n✅ Best Practices Summary:\")
    best_practices = [
        \"Start with simple, safe tools and gradually add complexity\",
        \"Test agents thoroughly with various inputs and scenarios\",
        \"Implement proper logging and monitoring\",
        \"Use the principle of least privilege for tool access\",
        \"Keep humans in the loop for critical decisions\",
        \"Regularly review and update agent behavior\"
    ]
    
    for practice in best_practices:
        print(f\"✅ {practice}\")
    
    print(\"=\"*70)


def main():
    \"\"\"Main function demonstrating agent concepts.\"\"\"
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info(\"🚀 Starting LangChain Agents Demonstration\")
    
    try:
        # Run all demonstrations
        demonstrate_basic_agent()
        demonstrate_agent_with_search()
        demonstrate_agent_planning()
        demonstrate_agent_types()
        demonstrate_agent_safety()
        
        print(\"\n🎯 Agent Key Takeaways:\")
        print(\"1. Agents combine reasoning with tool usage\")
        print(\"2. Different agent types serve different purposes\")
        print(\"3. Custom tools enable domain-specific capabilities\")
        print(\"4. Planning agents can solve complex multi-step problems\")
        print(\"5. Safety and validation are crucial for production use\")
        print(\"6. Agents bridge the gap between LLMs and external systems\")
        
        logger.info(\"✅ LangChain Agents demonstration completed successfully!\")
        
    except Exception as e:
        logger.error(f\"❌ Error occurred: {e}\")
        logger.info(\"💡 Check your API keys and internet connection\")


if __name__ == \"__main__\":
    main()