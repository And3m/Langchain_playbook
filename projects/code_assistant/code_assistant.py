#!/usr/bin/env python3
"""
Code Assistant - AI-Powered Code Generation and Analysis

This project demonstrates:
1. Code generation from natural language descriptions
2. Code explanation and documentation
3. Code review and optimization suggestions
4. Bug detection and fixing
5. Code refactoring assistance
6. Multi-language support

Key features:
- Natural language to code conversion
- Code analysis and explanation
- Automated documentation generation
- Code quality assessment
- Refactoring suggestions
"""

import sys
import ast
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Add utils to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import setup_logging, get_logger, get_api_key, timing_decorator
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, SystemMessage


class CodeAssistant:
    """AI-powered code assistant for various programming tasks."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.logger = get_logger(self.__class__.__name__)
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name=model,
            temperature=0.3,
            max_tokens=2000
        )
        
        # Initialize specialized chains
        self._setup_chains()
        
    def _setup_chains(self):
        """Set up specialized chains for different code tasks."""
        
        # Code generation chain
        self.code_gen_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert programmer. Generate clean, well-documented code 
            based on user requirements. Include error handling and follow best practices."""),
            HumanMessage(content="""
            Language: {language}
            Task: {task}
            Requirements: {requirements}
            
            Please provide:
            1. Complete, working code
            2. Clear comments explaining the logic
            3. Error handling where appropriate
            4. Usage examples if applicable
            """)
        ])
        
        self.code_generation_chain = LLMChain(
            llm=self.llm,
            prompt=self.code_gen_prompt
        )
        
        # Code explanation chain
        self.explain_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a code explanation expert. Analyze code and provide 
            clear, comprehensive explanations suitable for developers of different skill levels."""),
            HumanMessage(content="""
            Please explain this code:
            
            ```{language}
            {code}
            ```
            
            Provide:
            1. High-level overview of what the code does
            2. Step-by-step breakdown of key components
            3. Explanation of any complex logic or algorithms
            4. Potential improvements or considerations
            """)
        ])
        
        self.code_explanation_chain = LLMChain(
            llm=self.llm,
            prompt=self.explain_prompt
        )
        
        # Code review chain
        self.review_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a senior code reviewer. Analyze code for quality, 
            performance, security, and maintainability. Provide constructive feedback."""),
            HumanMessage(content="""
            Please review this {language} code:
            
            ```{language}
            {code}
            ```
            
            Focus on:
            1. Code quality and best practices
            2. Performance considerations
            3. Security vulnerabilities
            4. Maintainability and readability
            5. Specific improvement suggestions
            """)
        ])
        
        self.code_review_chain = LLMChain(
            llm=self.llm,
            prompt=self.review_prompt
        )
        
        # Bug fixing chain
        self.debug_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a debugging expert. Analyze code with errors 
            and provide fixed versions with explanations."""),
            HumanMessage(content="""
            This {language} code has issues:
            
            ```{language}
            {code}
            ```
            
            Error message: {error_message}
            
            Please provide:
            1. Analysis of the problem
            2. Fixed code
            3. Explanation of the changes made
            4. Prevention tips for similar issues
            """)
        ])
        
        self.debug_chain = LLMChain(
            llm=self.llm,
            prompt=self.debug_prompt
        )
    
    def generate_code(self, task: str, language: str = "python", 
                     requirements: str = "") -> Dict[str, Any]:
        """Generate code based on natural language description."""
        try:
            self.logger.info(f"Generating {language} code for: {task}")
            
            result = self.code_generation_chain.run(
                language=language,
                task=task,
                requirements=requirements
            )
            
            return {
                "status": "success",
                "code": result,
                "language": language,
                "task": task,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Code generation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "task": task
            }
    
    def explain_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Explain what the given code does."""
        try:
            self.logger.info(f"Explaining {language} code")
            
            result = self.code_explanation_chain.run(
                code=code,
                language=language
            )
            
            return {
                "status": "success",
                "explanation": result,
                "language": language,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Code explanation failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def review_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Review code for quality, performance, and best practices."""
        try:
            self.logger.info(f"Reviewing {language} code")
            
            result = self.code_review_chain.run(
                code=code,
                language=language
            )
            
            return {
                "status": "success",
                "review": result,
                "language": language,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Code review failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def debug_code(self, code: str, error_message: str, 
                   language: str = "python") -> Dict[str, Any]:
        """Help debug code by analyzing errors and providing fixes."""
        try:
            self.logger.info(f"Debugging {language} code")
            
            result = self.debug_chain.run(
                code=code,
                error_message=error_message,
                language=language
            )
            
            return {
                "status": "success",
                "debug_analysis": result,
                "language": language,
                "original_error": error_message,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Code debugging failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def analyze_complexity(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Analyze code complexity (currently supports Python)."""
        try:
            if language.lower() != "python":
                return {
                    "status": "error",
                    "error": "Complexity analysis currently only supports Python"
                }
            
            # Parse Python code
            tree = ast.parse(code)
            
            # Count different elements
            functions = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
            classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
            loops = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While)))
            conditions = sum(1 for node in ast.walk(tree) if isinstance(node, ast.If))
            
            # Calculate lines of code
            lines = len([line for line in code.split('\n') if line.strip() and not line.strip().startswith('#')])
            
            # Estimate complexity
            complexity_score = functions * 2 + classes * 3 + loops * 2 + conditions * 1
            
            if complexity_score < 10:
                complexity_level = "Low"
            elif complexity_score < 25:
                complexity_level = "Medium"
            else:
                complexity_level = "High"
            
            return {
                "status": "success",
                "metrics": {
                    "lines_of_code": lines,
                    "functions": functions,
                    "classes": classes,
                    "loops": loops,
                    "conditions": conditions,
                    "complexity_score": complexity_score,
                    "complexity_level": complexity_level
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except SyntaxError as e:
            return {
                "status": "error",
                "error": f"Syntax error in code: {e}"
            }
        except Exception as e:
            self.logger.error(f"Complexity analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def suggest_improvements(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Suggest specific improvements for the code."""
        try:
            improvement_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are a code optimization expert. Suggest specific, 
                actionable improvements for code quality, performance, and maintainability."""),
                HumanMessage(content=f"""
                Analyze this {language} code and suggest improvements:
                
                ```{language}
                {code}
                ```
                
                Provide specific suggestions for:
                1. Performance optimizations
                2. Code readability improvements
                3. Better error handling
                4. Design pattern applications
                5. Security enhancements (if applicable)
                """)
            ])
            
            improvement_chain = LLMChain(llm=self.llm, prompt=improvement_prompt)
            result = improvement_chain.run(code=code, language=language)
            
            return {
                "status": "success",
                "improvements": result,
                "language": language,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Improvement suggestion failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }


def demo_code_generation():
    """Demonstrate code generation capabilities."""
    print("\n" + "="*60)
    print("CODE GENERATION DEMO")
    print("="*60)
    
    api_key = get_api_key('openai')
    if not api_key:
        print("⚠️ OpenAI API key not found")
        return
    
    assistant = CodeAssistant(api_key)
    
    # Test different code generation tasks
    tasks = [
        {
            "task": "Create a function to calculate the factorial of a number",
            "language": "python",
            "requirements": "Include input validation and handle edge cases"
        },
        {
            "task": "Implement a simple binary search algorithm",
            "language": "python",
            "requirements": "Include docstring and example usage"
        }
    ]
    
    for i, task_info in enumerate(tasks, 1):
        print(f"\n🔨 Task {i}: {task_info['task']}")
        result = assistant.generate_code(**task_info)
        
        if result["status"] == "success":
            print("✅ Generated Code:")
            print(result["code"])
        else:
            print(f"❌ Error: {result['error']}")
        
        print("-" * 40)


def demo_code_explanation():
    """Demonstrate code explanation capabilities."""
    print("\n" + "="*60)
    print("CODE EXPLANATION DEMO")
    print("="*60)
    
    api_key = get_api_key('openai')
    if not api_key:
        print("⚠️ OpenAI API key not found")
        return
    
    assistant = CodeAssistant(api_key)
    
    # Sample code to explain
    sample_code = """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
    """
    
    print("🔍 Explaining this code:")
    print(sample_code)
    
    result = assistant.explain_code(sample_code, "python")
    
    if result["status"] == "success":
        print("\n📝 Explanation:")
        print(result["explanation"])
    else:
        print(f"❌ Error: {result['error']}")


def demo_code_review():
    """Demonstrate code review capabilities."""
    print("\n" + "="*60)
    print("CODE REVIEW DEMO")
    print("="*60)
    
    api_key = get_api_key('openai')
    if not api_key:
        print("⚠️ OpenAI API key not found")
        return
    
    assistant = CodeAssistant(api_key)
    
    # Sample code with potential issues
    problematic_code = """
def process_data(data):
    result = []
    for i in range(len(data)):
        if data[i] != None:
            result.append(data[i] * 2)
    return result

def calculate_average(numbers):
    total = 0
    for num in numbers:
        total = total + num
    return total / len(numbers)
    """
    
    print("🔍 Reviewing this code:")
    print(problematic_code)
    
    result = assistant.review_code(problematic_code, "python")
    
    if result["status"] == "success":
        print("\n📋 Code Review:")
        print(result["review"])
    else:
        print(f"❌ Error: {result['error']}")


def demo_debugging():
    """Demonstrate debugging capabilities."""
    print("\n" + "="*60)
    print("DEBUGGING DEMO")
    print("="*60)
    
    api_key = get_api_key('openai')
    if not api_key:
        print("⚠️ OpenAI API key not found")
        return
    
    assistant = CodeAssistant(api_key)
    
    # Buggy code
    buggy_code = """
def divide_numbers(a, b):
    return a / b

def get_user_input():
    num1 = input("Enter first number: ")
    num2 = input("Enter second number: ")
    result = divide_numbers(num1, num2)
    print(f"Result: {result}")
    """
    
    error_message = "TypeError: unsupported operand type(s) for /: 'str' and 'str'"
    
    print("🐛 Debugging this code:")
    print(buggy_code)
    print(f"\nError: {error_message}")
    
    result = assistant.debug_code(buggy_code, error_message, "python")
    
    if result["status"] == "success":
        print("\n🔧 Debug Analysis:")
        print(result["debug_analysis"])
    else:
        print(f"❌ Error: {result['error']}")


def demo_complexity_analysis():
    """Demonstrate code complexity analysis."""
    print("\n" + "="*60)
    print("COMPLEXITY ANALYSIS DEMO")
    print("="*60)
    
    api_key = get_api_key('openai')
    if not api_key:
        print("⚠️ OpenAI API key not found, showing local analysis only")
    
    assistant = CodeAssistant(api_key or "dummy")
    
    # Sample code for analysis
    complex_code = """
class DataProcessor:
    def __init__(self, data):
        self.data = data
        self.processed = False
    
    def validate_data(self):
        if not self.data:
            return False
        for item in self.data:
            if not isinstance(item, (int, float)):
                return False
        return True
    
    def process(self):
        if not self.validate_data():
            raise ValueError("Invalid data")
        
        result = []
        for item in self.data:
            if item > 0:
                result.append(item * 2)
            elif item < 0:
                result.append(abs(item))
            else:
                result.append(1)
        
        self.processed = True
        return result
    
    def get_statistics(self):
        if not self.processed:
            self.process()
        
        total = sum(self.data)
        count = len(self.data)
        average = total / count if count > 0 else 0
        
        return {
            'total': total,
            'count': count,
            'average': average
        }
    """
    
    print("📊 Analyzing complexity of this code:")
    print(complex_code[:200] + "...")
    
    result = assistant.analyze_complexity(complex_code, "python")
    
    if result["status"] == "success":
        metrics = result["metrics"]
        print(f"\n📈 Complexity Metrics:")
        print(f"Lines of Code: {metrics['lines_of_code']}")
        print(f"Functions: {metrics['functions']}")
        print(f"Classes: {metrics['classes']}")
        print(f"Loops: {metrics['loops']}")
        print(f"Conditions: {metrics['conditions']}")
        print(f"Complexity Score: {metrics['complexity_score']}")
        print(f"Complexity Level: {metrics['complexity_level']}")
    else:
        print(f"❌ Error: {result['error']}")


def main():
    """Main function demonstrating the code assistant."""
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("🚀 Starting Code Assistant Demonstration")
    
    print("🔧 AI-Powered Code Assistant")
    print("This tool helps with code generation, explanation, review, and debugging.")
    
    try:
        # Run all demonstrations
        demo_code_generation()
        demo_code_explanation()
        demo_code_review()
        demo_debugging()
        demo_complexity_analysis()
        
        print("\n" + "="*60)
        print("CODE ASSISTANT FEATURES SUMMARY")
        print("="*60)
        print("✅ Code Generation - Natural language to code")
        print("✅ Code Explanation - Understand existing code")
        print("✅ Code Review - Quality and best practices")
        print("✅ Debugging - Fix errors and issues")
        print("✅ Complexity Analysis - Measure code complexity")
        print("✅ Improvement Suggestions - Optimization tips")
        
        print("\n💡 Use Cases:")
        print("• Learning new programming concepts")
        print("• Code review automation")
        print("• Legacy code understanding")
        print("• Debugging assistance")
        print("• Code quality improvement")
        print("• Rapid prototyping")
        
        logger.info("✅ Code Assistant demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error occurred: {e}")
        logger.info("💡 Check your API keys and internet connection")


if __name__ == "__main__":
    main()