#!/usr/bin/env python3
"""
Advanced Custom Tools - Building Specialized LangChain Tools

This example demonstrates how to create sophisticated custom tools for LangChain agents,
including tools with complex logic, external API integration, and advanced features.

Key concepts:
1. Custom tool development patterns
2. Tool validation and error handling
3. External API integration
4. Tool composition and chaining
5. Async tool implementation
6. Tool metadata and documentation
"""

import sys
import asyncio
import json
import re
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional, Type, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from pydantic import BaseModel, Field

# Add utils to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import setup_logging, get_logger, get_api_key, timing_decorator
from langchain.tools import BaseTool, Tool, tool
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.base import ToolException


# Custom tool implementations

class WeatherAPITool(BaseTool):
    """Advanced weather tool with caching and error handling."""
    
    name = "weather_api"
    description = "Get current weather for any city. Input should be a city name."
    
    def __init__(self):
        super().__init__()
        self.cache = {}
        self.cache_duration = timedelta(minutes=10)
        self.logger = get_logger(self.__class__.__name__)
    
    def _is_cache_valid(self, city: str) -> bool:
        """Check if cached data is still valid."""
        if city not in self.cache:
            return False
        
        cached_time = self.cache[city]['timestamp']
        return datetime.now() - cached_time < self.cache_duration
    
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute the weather tool."""
        try:
            city = query.strip()
            
            # Check cache first
            if self._is_cache_valid(city):
                self.logger.info(f"Returning cached weather for {city}")
                return self.cache[city]['data']
            
            # Mock weather API call (replace with real API)
            weather_data = self._fetch_weather(city)
            
            # Cache the result
            self.cache[city] = {
                'data': weather_data,
                'timestamp': datetime.now()
            }
            
            return weather_data
            
        except Exception as e:
            self.logger.error(f"Weather tool error: {e}")
            return f"Sorry, I couldn't get weather information for {query}. Error: {str(e)}"
    
    def _fetch_weather(self, city: str) -> str:
        """Fetch weather data (mock implementation)."""
        # In real implementation, use actual weather API
        import random
        
        temperatures = [15, 18, 22, 25, 28, 30, 32]
        conditions = ["sunny", "cloudy", "partly cloudy", "rainy", "windy"]
        
        temp = random.choice(temperatures)
        condition = random.choice(conditions)
        
        return f"Weather in {city}: {temp}°C, {condition}"


class DatabaseQueryTool(BaseTool):
    """Tool for querying a mock database with safety checks."""
    
    name = "database_query"
    description = "Query database with SQL. Input should be a valid SQL SELECT statement."
    
    # Mock database
    mock_data = {
        'users': [
            {'id': 1, 'name': 'Alice', 'email': 'alice@example.com', 'age': 25},
            {'id': 2, 'name': 'Bob', 'email': 'bob@example.com', 'age': 30},
            {'id': 3, 'name': 'Charlie', 'email': 'charlie@example.com', 'age': 35}
        ],
        'orders': [
            {'id': 1, 'user_id': 1, 'product': 'Laptop', 'amount': 1200},
            {'id': 2, 'user_id': 2, 'product': 'Phone', 'amount': 800},
            {'id': 3, 'user_id': 1, 'product': 'Tablet', 'amount': 400}
        ]
    }
    
    def _validate_query(self, query: str) -> bool:
        """Validate SQL query for safety."""
        query_lower = query.lower().strip()
        
        # Only allow SELECT statements
        if not query_lower.startswith('select'):
            return False
        
        # Block dangerous keywords
        dangerous_keywords = ['drop', 'delete', 'insert', 'update', 'alter', 'create']
        for keyword in dangerous_keywords:
            if keyword in query_lower:
                return False
        
        return True
    
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute database query."""
        try:
            if not self._validate_query(query):
                raise ToolException("Invalid or unsafe SQL query. Only SELECT statements are allowed.")
            
            # Simple query parser for mock data
            result = self._execute_mock_query(query)
            return json.dumps(result, indent=2)
            
        except ToolException:
            raise
        except Exception as e:
            return f"Database query failed: {str(e)}"
    
    def _execute_mock_query(self, query: str) -> List[Dict]:
        """Execute query on mock data."""
        query_lower = query.lower()
        
        # Simple table detection
        if 'from users' in query_lower:
            data = self.mock_data['users']
        elif 'from orders' in query_lower:
            data = self.mock_data['orders']
        else:
            return [{"error": "Table not found. Available tables: users, orders"}]
        
        # Simple filtering (very basic implementation)
        if 'where' in query_lower:
            # Extract simple conditions
            where_part = query_lower.split('where')[1].strip()
            data = self._apply_where_clause(data, where_part)
        
        return data[:5]  # Limit results
    
    def _apply_where_clause(self, data: List[Dict], where_clause: str) -> List[Dict]:
        """Apply simple WHERE clause filtering."""
        # Very basic implementation - in practice, use proper SQL parser
        if 'age >' in where_clause:
            age_threshold = int(re.search(r'age > (\d+)', where_clause).group(1))
            return [item for item in data if item.get('age', 0) > age_threshold]
        
        return data


class FileOperationTool(BaseTool):
    """Safe file operations tool with sandboxing."""
    
    name = "file_operations"
    description = "Read, write, or list files. Input format: 'action:path' (e.g., 'read:data.txt', 'list:.')"
    
    def __init__(self, allowed_paths: List[str] = None):
        super().__init__()
        # Sandbox: only allow operations in specific directories
        self.allowed_paths = allowed_paths or [str(Path.cwd())]
        self.logger = get_logger(self.__class__.__name__)
    
    def _is_path_allowed(self, path: str) -> bool:
        """Check if path is within allowed directories."""
        try:
            abs_path = Path(path).resolve()
            for allowed in self.allowed_paths:
                if abs_path.is_relative_to(Path(allowed).resolve()):
                    return True
            return False
        except Exception:
            return False
    
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute file operation."""
        try:
            if ':' not in query:
                return "Invalid format. Use 'action:path' (e.g., 'read:file.txt')"
            
            action, path = query.split(':', 1)
            action = action.strip().lower()
            path = path.strip()
            
            if not self._is_path_allowed(path):
                return f"Access denied: Path '{path}' is not allowed"
            
            if action == 'read':
                return self._read_file(path)
            elif action == 'list':
                return self._list_directory(path)
            elif action == 'write':
                return "Write operations not implemented for safety"
            else:
                return f"Unknown action: {action}. Supported: read, list"
                
        except Exception as e:
            self.logger.error(f"File operation error: {e}")
            return f"File operation failed: {str(e)}"
    
    def _read_file(self, path: str) -> str:
        """Read file contents safely."""
        try:
            file_path = Path(path)
            if not file_path.exists():
                return f"File not found: {path}"
            
            if file_path.stat().st_size > 10000:  # 10KB limit
                return f"File too large: {path} (limit: 10KB)"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return f"Content of {path}:\n{content}"
                
        except UnicodeDecodeError:
            return f"Cannot read binary file: {path}"
        except Exception as e:
            return f"Error reading {path}: {str(e)}"
    
    def _list_directory(self, path: str) -> str:
        """List directory contents."""
        try:
            dir_path = Path(path)
            if not dir_path.exists():
                return f"Directory not found: {path}"
            
            if not dir_path.is_dir():
                return f"Not a directory: {path}"
            
            items = []
            for item in dir_path.iterdir():
                if item.is_dir():
                    items.append(f"📁 {item.name}/")
                else:
                    size = item.stat().st_size
                    items.append(f"📄 {item.name} ({size} bytes)")
            
            return f"Contents of {path}:\n" + "\n".join(items[:20])  # Limit output
            
        except Exception as e:
            return f"Error listing {path}: {str(e)}"


# Async tool example
class AsyncWebSearchTool(BaseTool):
    """Async web search tool (mock implementation)."""
    
    name = "web_search"
    description = "Search the web for information. Input should be a search query."
    
    async def _arun(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Async execution of web search."""
        try:
            # Simulate async web search
            await asyncio.sleep(1)  # Simulate network delay
            
            # Mock search results
            results = [
                f"Result 1 for '{query}': This is a mock search result",
                f"Result 2 for '{query}': Another mock result with relevant information",
                f"Result 3 for '{query}': Third result for your query"
            ]
            
            return "Search results:\n" + "\n".join(results)
            
        except Exception as e:
            return f"Search failed: {str(e)}"
    
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Sync execution (fallback)."""
        return "Use async version of this tool for better performance"


# Tool composition example
def create_calculator_tool() -> Tool:
    """Create a calculator tool using the @tool decorator."""
    
    @tool
    def calculator(expression: str) -> str:
        """
        Calculate mathematical expressions safely.
        Input should be a mathematical expression like '2 + 2' or 'sqrt(16)'.
        """
        try:
            # Safety: only allow specific characters and functions
            allowed_chars = set('0123456789+-*/().sqrt()pow()abs()min()max() ')
            if not all(c in allowed_chars for c in expression):
                return "Invalid characters in expression"
            
            # Safe evaluation using restricted environment
            import math
            safe_dict = {
                "__builtins__": {},
                "sqrt": math.sqrt,
                "pow": math.pow,
                "abs": abs,
                "min": min,
                "max": max
            }
            
            result = eval(expression, safe_dict)
            return f"Result: {result}"
            
        except Exception as e:
            return f"Calculation error: {str(e)}"
    
    return calculator


# Tool registry for dynamic tool loading
class ToolRegistry:
    """Registry for managing custom tools."""
    
    def __init__(self):
        self.tools = {}
        self.logger = get_logger(self.__class__.__name__)
    
    def register_tool(self, tool: BaseTool) -> None:
        """Register a new tool."""
        self.tools[tool.name] = tool
        self.logger.info(f"Registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tools."""
        return list(self.tools.keys())
    
    def create_tool_list(self) -> List[BaseTool]:
        """Create a list of all tools for agent use."""
        return list(self.tools.values())


def demonstrate_custom_tools():
    """Demonstrate custom tool usage with agents."""
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning("⚠️ OpenAI API key not found, showing tool structure only")
        
        print("\n" + "="*60)
        print("CUSTOM TOOLS DEMONSTRATION (STRUCTURE)")
        print("="*60)
        
        # Show tool registry
        registry = ToolRegistry()
        
        # Register tools
        tools = [
            WeatherAPITool(),
            DatabaseQueryTool(),
            FileOperationTool(allowed_paths=[str(Path.cwd())]),
            AsyncWebSearchTool(),
            create_calculator_tool()
        ]
        
        for tool in tools:
            registry.register_tool(tool)
        
        print(f"🔧 Registered {len(registry.list_tools())} custom tools:")
        for tool_name in registry.list_tools():
            tool = registry.get_tool(tool_name)
            print(f"   • {tool_name}: {tool.description}")
        
        print("\n💡 Tool Features Demonstrated:")
        print("   • Caching and performance optimization")
        print("   • Input validation and safety checks")
        print("   • Error handling and graceful failures")
        print("   • Async tool implementation")
        print("   • Tool composition and registry pattern")
        
        return
    
    logger.info("🔧 Custom Tools Demonstration")
    
    # Create LLM
    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name="gpt-3.5-turbo",
        temperature=0.3
    )
    
    # Create tool registry
    registry = ToolRegistry()
    
    # Register custom tools
    tools = [
        WeatherAPITool(),
        DatabaseQueryTool(),
        FileOperationTool(allowed_paths=[str(Path.cwd())]),
        create_calculator_tool()
    ]
    
    for tool in tools:
        registry.register_tool(tool)
    
    # Create agent with custom tools
    agent = initialize_agent(
        tools=registry.create_tool_list(),
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=3
    )
    
    print("\n" + "="*60)
    print("CUSTOM TOOLS WITH AGENT")
    print("="*60)
    
    # Test scenarios
    test_queries = [
        "What's the weather in New York?",
        "Calculate the square root of 144",
        "List the files in the current directory",
        "Query the database to find all users"
    ]
    
    for i, query in enumerate(test_queries, 1):
        try:
            print(f"\n🤖 Query {i}: {query}")
            response = agent.run(query)
            print(f"📝 Response: {response}")
            
            if i == 1:  # Only run first example to save tokens
                break
                
        except Exception as e:
            logger.error(f"Error in query {i}: {e}")
            print(f"❌ Error: {e}")
    
    print(f"\n✅ Custom tools demonstration completed!")


def demonstrate_tool_best_practices():
    """Demonstrate best practices for custom tool development."""
    logger = get_logger(__name__)
    logger.info("📋 Tool Development Best Practices")
    
    print("\n" + "="*70)
    print("CUSTOM TOOL DEVELOPMENT BEST PRACTICES")
    print("="*70)
    
    practices = {
        "🔒 Security": [
            "Validate all inputs thoroughly",
            "Implement sandboxing for file operations",
            "Use allow-lists for dangerous operations",
            "Sanitize user-provided data",
            "Implement proper access controls"
        ],
        "⚡ Performance": [
            "Implement caching for expensive operations",
            "Use async operations when possible",
            "Set reasonable timeouts",
            "Limit output size to prevent memory issues",
            "Implement rate limiting for external APIs"
        ],
        "🛡️ Error Handling": [
            "Handle all exceptions gracefully",
            "Provide meaningful error messages",
            "Log errors for debugging",
            "Implement fallback mechanisms",
            "Validate tool outputs"
        ],
        "📚 Documentation": [
            "Write clear tool descriptions",
            "Provide input format examples",
            "Document expected outputs",
            "Include usage examples",
            "Specify tool limitations"
        ],
        "🧪 Testing": [
            "Test with various input types",
            "Test error conditions",
            "Validate edge cases",
            "Test tool composition",
            "Monitor tool performance"
        ]
    }
    
    for category, items in practices.items():
        print(f"\n{category}:")
        for item in items:
            print(f"   • {item}")
    
    print("\n🏗️ Tool Architecture Patterns:")
    patterns = {
        "Registry Pattern": "Centralized tool management and discovery",
        "Decorator Pattern": "Adding features like caching and logging",
        "Strategy Pattern": "Different implementations for same tool interface",
        "Factory Pattern": "Dynamic tool creation based on configuration",
        "Proxy Pattern": "Adding security and validation layers"
    }
    
    for pattern, description in patterns.items():
        print(f"   📋 {pattern}: {description}")
    
    print("\n⚠️ Common Pitfalls:")
    pitfalls = [
        "Not validating inputs (security risk)",
        "Blocking operations in async tools",
        "Not handling external API failures",
        "Exposing sensitive information in errors",
        "Not limiting resource usage",
        "Poor error messages for users"
    ]
    
    for pitfall in pitfalls:
        print(f"   ❌ {pitfall}")
    
    print("\n✅ Tool Quality Checklist:")
    checklist = [
        "Input validation implemented",
        "Error handling comprehensive",
        "Security measures in place",
        "Performance optimized",
        "Documentation complete",
        "Tests written and passing",
        "Logging implemented",
        "Resource limits set"
    ]
    
    for item in checklist:
        print(f"   ☑️ {item}")
    
    print("="*70)


def main():
    """Main function demonstrating advanced custom tools."""
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("🚀 Starting Advanced Custom Tools Demonstration")
    
    try:
        demonstrate_custom_tools()
        demonstrate_tool_best_practices()
        
        print("\n🎯 Custom Tools Key Takeaways:")
        print("1. Security is paramount - validate everything")
        print("2. Performance matters - implement caching and async")
        print("3. Error handling should be comprehensive and user-friendly")
        print("4. Documentation and testing are essential")
        print("5. Tool composition enables powerful agent capabilities")
        print("6. Registry patterns help manage complex tool ecosystems")
        
        logger.info("✅ Advanced Custom Tools demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error occurred: {e}")
        logger.info("💡 Check your API keys and internet connection")


if __name__ == "__main__":
    main()