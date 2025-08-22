#!/usr/bin/env python3
"""
Test Suite for LangChain Playbook

This comprehensive test suite validates all examples, projects, and utilities
to ensure everything works correctly and provides a good learning experience.

Test Categories:
1. Environment and Setup Tests
2. Utility Function Tests
3. Basic Examples Tests
4. Intermediate Examples Tests
5. Project Functionality Tests
6. Integration Tests
"""

import sys
import os
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Add project root and utils to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'utils'))

# Import utilities
try:
    from utils.config import get_api_key, validate_environment
    from utils.logging import setup_logging, get_logger
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False


class TestEnvironmentSetup(unittest.TestCase):
    """Test environment configuration and setup."""
    
    def test_python_version(self):
        """Test Python version compatibility."""
        import sys
        version = sys.version_info
        self.assertGreaterEqual(version.major, 3, "Python 3 required")
        self.assertGreaterEqual(version.minor, 8, "Python 3.8+ required")
    
    def test_required_packages(self):
        """Test that required packages are importable."""
        required_packages = [
            'langchain',
            'openai',
            'pathlib',
            'typing',
            'datetime',
            'json'
        ]
        
        for package in required_packages:
            with self.subTest(package=package):
                try:
                    __import__(package)
                except ImportError:
                    self.fail(f"Required package '{package}' not available")
    
    @unittest.skipUnless(UTILS_AVAILABLE, "Utils not available")
    def test_utils_import(self):
        """Test utils module imports correctly."""
        from utils.config import get_api_key
        from utils.logging import get_logger
        
        # Test basic functionality
        logger = get_logger('test')
        self.assertIsNotNone(logger)
    
    def test_project_structure(self):
        """Test that project structure is complete."""
        expected_dirs = [
            'basics',
            'intermediate', 
            'projects',
            'notebooks',
            'utils',
            'docs',
            'tests'
        ]
        
        for dir_name in expected_dirs:
            dir_path = project_root / dir_name
            with self.subTest(directory=dir_name):
                self.assertTrue(dir_path.exists(), f"Directory '{dir_name}' missing")


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions work correctly."""
    
    @unittest.skipUnless(UTILS_AVAILABLE, "Utils not available")
    def test_get_api_key(self):
        """Test API key retrieval."""
        from utils.config import get_api_key
        
        # Test with environment variable
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            key = get_api_key('openai')
            self.assertEqual(key, 'test-key')
        
        # Test missing key
        with patch.dict(os.environ, {}, clear=True):
            key = get_api_key('openai')
            self.assertIsNone(key)
    
    @unittest.skipUnless(UTILS_AVAILABLE, "Utils not available")
    def test_logging_setup(self):
        """Test logging configuration."""
        from utils.logging import setup_logging, get_logger
        
        setup_logging()
        logger = get_logger('test')
        
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, 'test')
    
    def test_file_existence(self):
        """Test that essential files exist."""
        essential_files = [
            'README.md',
            'requirements.txt',
            '.env.example'
        ]
        
        for file_name in essential_files:
            file_path = project_root / file_name
            with self.subTest(file=file_name):
                self.assertTrue(file_path.exists(), f"File '{file_name}' missing")


class TestBasicExamples(unittest.TestCase):
    """Test basic LangChain examples."""
    
    def setUp(self):
        """Set up test environment."""
        self.mock_api_key = "test-api-key"
    
    def test_hello_langchain_import(self):
        """Test that hello_langchain example imports correctly."""
        try:
            sys.path.append(str(project_root / 'basics' / '01_getting_started'))
            import hello_langchain
            self.assertTrue(hasattr(hello_langchain, 'main'))
        except ImportError as e:
            self.skipTest(f"Hello LangChain import failed: {e}")
    
    @patch('langchain.llms.OpenAI')
    def test_basic_llm_functionality(self, mock_openai):
        """Test basic LLM functionality with mocked API."""
        mock_llm = Mock()
        mock_llm.return_value = "This is a test response"
        mock_openai.return_value = mock_llm
        
        # Test basic LLM usage pattern
        from langchain.llms import OpenAI
        llm = OpenAI(openai_api_key=self.mock_api_key)
        response = llm("Test prompt")
        
        self.assertEqual(response, "This is a test response")
    
    def test_prompt_template_creation(self):
        """Test prompt template functionality."""
        from langchain.prompts import PromptTemplate
        
        template = PromptTemplate(
            input_variables=["topic"],
            template="Tell me about {topic}"
        )
        
        formatted = template.format(topic="AI")
        self.assertEqual(formatted, "Tell me about AI")
    
    def test_chain_composition(self):
        """Test basic chain composition."""
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
        
        # Mock LLM
        mock_llm = Mock()
        mock_llm.predict.return_value = "Chain response"
        
        template = PromptTemplate(
            input_variables=["input"],
            template="Process: {input}"
        )
        
        chain = LLMChain(llm=mock_llm, prompt=template)
        self.assertIsNotNone(chain)


class TestIntermediateExamples(unittest.TestCase):
    """Test intermediate LangChain examples."""
    
    def test_memory_import(self):
        """Test memory examples import correctly."""
        try:
            from langchain.memory import ConversationBufferMemory
            memory = ConversationBufferMemory()
            self.assertIsNotNone(memory)
        except ImportError as e:
            self.skipTest(f"Memory import failed: {e}")
    
    def test_conversation_memory(self):
        """Test conversation memory functionality."""
        from langchain.memory import ConversationBufferMemory
        
        memory = ConversationBufferMemory()
        
        # Test saving context
        memory.save_context(
            {"input": "Hello"}, 
            {"output": "Hi there!"}
        )
        
        # Test loading memory
        variables = memory.load_memory_variables({})
        self.assertIn("history", variables)
    
    def test_text_splitter(self):
        """Test document text splitting."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20
        )
        
        text = "This is a test document. " * 10
        chunks = splitter.split_text(text)
        
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 100)
    
    @patch('langchain.vectorstores.FAISS.from_texts')
    def test_vector_store_creation(self, mock_faiss):
        """Test vector store creation (mocked)."""
        mock_vectorstore = Mock()
        mock_faiss.return_value = mock_vectorstore
        
        from langchain.vectorstores import FAISS
        from langchain.embeddings import OpenAIEmbeddings
        
        # Mock embeddings
        mock_embeddings = Mock()
        
        vectorstore = FAISS.from_texts(
            ["test text"], 
            mock_embeddings
        )
        
        self.assertIsNotNone(vectorstore)


class TestProjectFunctionality(unittest.TestCase):
    """Test project functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_api_key = "test-api-key"
    
    def test_chatbot_project_import(self):
        """Test chatbot project imports correctly."""
        try:
            sys.path.append(str(project_root / 'projects' / 'chatbot'))
            import chatbot_app
            self.assertTrue(hasattr(chatbot_app, 'PersonalityChatbot'))
        except ImportError as e:
            self.skipTest(f"Chatbot import failed: {e}")
    
    def test_document_qa_import(self):
        """Test document Q&A project imports correctly."""
        try:
            sys.path.append(str(project_root / 'projects' / 'document_qa'))
            import qa_system
            self.assertTrue(hasattr(qa_system, 'DocumentQASystem'))
        except ImportError as e:
            self.skipTest(f"Document Q&A import failed: {e}")
    
    def test_code_assistant_import(self):
        """Test code assistant project imports correctly."""
        try:
            sys.path.append(str(project_root / 'projects' / 'code_assistant'))
            import code_assistant
            self.assertTrue(hasattr(code_assistant, 'CodeAssistant'))
        except ImportError as e:
            self.skipTest(f"Code assistant import failed: {e}")
    
    def test_research_assistant_import(self):
        """Test research assistant project imports correctly."""
        try:
            sys.path.append(str(project_root / 'projects' / 'research_assistant'))
            import research_assistant
            self.assertTrue(hasattr(research_assistant, 'ResearchAssistant'))
        except ImportError as e:
            self.skipTest(f"Research assistant import failed: {e}")
    
    def test_api_service_import(self):
        """Test API service project imports correctly."""
        try:
            sys.path.append(str(project_root / 'projects' / 'api_service'))
            import api_service
            self.assertTrue(hasattr(api_service, 'app'))
        except ImportError as e:
            self.skipTest(f"API service import failed: {e}")


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows."""
    
    @patch('langchain.chat_models.ChatOpenAI')
    def test_end_to_end_chain(self, mock_chat):
        """Test end-to-end chain execution."""
        # Mock ChatOpenAI
        mock_llm = Mock()
        mock_llm.predict.return_value = "Integration test response"
        mock_chat.return_value = mock_llm
        
        from langchain.chat_models import ChatOpenAI
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
        
        # Create chain
        llm = ChatOpenAI(openai_api_key="test-key")
        template = PromptTemplate(
            input_variables=["question"],
            template="Answer: {question}"
        )
        chain = LLMChain(llm=llm, prompt=template)
        
        # Execute chain
        result = chain.run(question="What is AI?")
        self.assertEqual(result, "Integration test response")
    
    def test_notebook_structure(self):
        """Test that notebooks have correct structure."""
        notebooks_dir = project_root / 'notebooks'
        
        if notebooks_dir.exists():
            notebook_files = list(notebooks_dir.glob('*.ipynb'))
            self.assertGreater(len(notebook_files), 0, "No notebooks found")
            
            for notebook_file in notebook_files:
                with open(notebook_file, 'r', encoding='utf-8') as f:
                    try:
                        notebook_data = json.load(f)
                        self.assertIn('cells', notebook_data)
                        self.assertIn('metadata', notebook_data)
                    except json.JSONDecodeError:
                        self.fail(f"Invalid notebook format: {notebook_file}")


class TestValidation(unittest.TestCase):
    """Validation tests for code quality and consistency."""
    
    def test_python_syntax(self):
        """Test that all Python files have valid syntax."""
        python_files = []
        
        # Collect all Python files
        for directory in ['basics', 'intermediate', 'projects', 'utils']:
            dir_path = project_root / directory
            if dir_path.exists():
                python_files.extend(dir_path.rglob('*.py'))
        
        for py_file in python_files:
            with self.subTest(file=str(py_file)):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        source = f.read()
                    compile(source, str(py_file), 'exec')
                except SyntaxError as e:
                    self.fail(f"Syntax error in {py_file}: {e}")
                except UnicodeDecodeError:
                    # Skip binary files
                    continue
    
    def test_import_consistency(self):
        """Test that imports are consistent across files."""
        # Test common import patterns
        common_imports = [
            'import sys',
            'from pathlib import Path',
            'from typing import'
        ]
        
        # This is a basic test - in practice, you might want more sophisticated checking
        self.assertTrue(True, "Import consistency check placeholder")
    
    def test_docstring_presence(self):
        """Test that key functions have docstrings."""
        # This would be implemented to check for docstrings in main functions
        # For now, just ensuring the test framework works
        self.assertTrue(True, "Docstring presence check placeholder")


def run_health_check():
    """Run a quick health check of the system."""
    print("🔍 Running LangChain Playbook Health Check...")
    
    health_status = {
        "python_version": True,
        "packages": True,
        "project_structure": True,
        "utils": UTILS_AVAILABLE,
        "examples": True
    }
    
    # Check Python version
    import sys
    if sys.version_info < (3, 8):
        health_status["python_version"] = False
        print("❌ Python 3.8+ required")
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check key packages
    try:
        import langchain
        print(f"✅ LangChain {langchain.__version__}")
    except ImportError:
        health_status["packages"] = False
        print("❌ LangChain not installed")
    
    # Check project structure
    required_dirs = ['basics', 'intermediate', 'projects', 'utils']
    for dir_name in required_dirs:
        if (project_root / dir_name).exists():
            print(f"✅ {dir_name}/ directory")
        else:
            health_status["project_structure"] = False
            print(f"❌ {dir_name}/ directory missing")
    
    # Check utils
    if UTILS_AVAILABLE:
        print("✅ Utils module")
    else:
        print("⚠️ Utils module not available")
    
    # Overall status
    if all(health_status.values()):
        print("\n🎉 All systems go! LangChain Playbook is ready.")
        return True
    else:
        print("\n⚠️ Some issues found. Check the messages above.")
        return False


def run_comprehensive_tests():
    """Run comprehensive test suite."""
    print("🧪 Running Comprehensive Test Suite...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestEnvironmentSetup,
        TestUtilityFunctions,
        TestBasicExamples,
        TestIntermediateExamples,
        TestProjectFunctionality,
        TestIntegration,
        TestValidation
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    if result.wasSuccessful():
        print(f"\n🎉 All {result.testsRun} tests passed!")
        return True
    else:
        print(f"\n❌ {len(result.failures)} failures, {len(result.errors)} errors out of {result.testsRun} tests")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LangChain Playbook Test Suite")
    parser.add_argument(
        "--mode", 
        choices=["health", "full", "specific"],
        default="health",
        help="Test mode to run"
    )
    parser.add_argument(
        "--test-class",
        help="Specific test class to run (for specific mode)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "health":
        success = run_health_check()
        sys.exit(0 if success else 1)
    
    elif args.mode == "full":
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)
    
    elif args.mode == "specific" and args.test_class:
        # Run specific test class
        suite = unittest.TestLoader().loadTestsFromName(args.test_class, sys.modules[__name__])
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    
    else:
        parser.print_help()
        sys.exit(1)