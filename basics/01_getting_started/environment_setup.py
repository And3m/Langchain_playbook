#!/usr/bin/env python3
\"\"\"
Environment Setup Validation

This script validates your LangChain Playbook setup and helps diagnose issues.
Run this before starting with other examples to ensure everything is configured correctly.

Checks:
1. Python version
2. Required packages
3. API keys
4. Model connectivity
5. Basic functionality
\"\"\"

import sys
import importlib
from pathlib import Path

# Add utils to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import setup_logging, get_logger, validate_api_keys, load_config


def check_python_version():
    \"\"\"Check if Python version is compatible.\"\"\"
    logger = get_logger(__name__)
    
    major, minor = sys.version_info[:2]
    
    if major < 3 or (major == 3 and minor < 8):
        logger.error(f\"❌ Python {major}.{minor} is not supported. Need Python 3.8+\")
        return False
    else:
        logger.info(f\"✅ Python {major}.{minor} - Compatible\")
        return True


def check_required_packages():
    \"\"\"Check if required packages are installed.\"\"\"
    logger = get_logger(__name__)
    
    required_packages = [
        'langchain',
        'langchain_community', 
        'langchain_openai',
        'openai',
        'requests',
        'python_dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            logger.info(f\"✅ {package} - Installed\")
        except ImportError:
            logger.error(f\"❌ {package} - Missing\")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f\"Missing packages: {', '.join(missing_packages)}\")
        logger.info(\"💡 Run: pip install -r requirements.txt\")
        return False
    
    return True


def check_api_keys():
    \"\"\"Check API key configuration.\"\"\"
    logger = get_logger(__name__)
    
    providers = ['openai', 'anthropic', 'google']
    results = validate_api_keys(providers)
    
    has_at_least_one = False
    
    for provider, available in results.items():
        if available:
            logger.info(f\"✅ {provider.upper()} API key - Found\")
            has_at_least_one = True
        else:
            logger.warning(f\"⚠️ {provider.upper()} API key - Not found\")
    
    if not has_at_least_one:
        logger.error(\"❌ No API keys found. You need at least one to use LangChain.\")
        logger.info(\"💡 Check docs/api-keys.md for setup instructions\")
        return False
    
    return True


def test_basic_llm():
    \"\"\"Test basic LLM functionality.\"\"\"
    logger = get_logger(__name__)
    
    try:
        from langchain.llms import OpenAI
        from utils import get_api_key
        
        api_key = get_api_key('openai')
        if not api_key:
            logger.warning(\"⚠️ Skipping OpenAI test - no API key\")
            return True
        
        logger.info(\"🧪 Testing OpenAI LLM connection...\")
        
        llm = OpenAI(
            openai_api_key=api_key,
            max_tokens=10,
            temperature=0
        )
        
        response = llm(\"Hello\")
        
        if response and len(response.strip()) > 0:
            logger.info(\"✅ OpenAI LLM - Working\")
            return True
        else:
            logger.error(\"❌ OpenAI LLM - Empty response\")
            return False
            
    except Exception as e:
        logger.error(f\"❌ OpenAI LLM test failed: {e}\")
        return False


def test_chat_model():
    \"\"\"Test chat model functionality.\"\"\"
    logger = get_logger(__name__)
    
    try:
        from langchain.chat_models import ChatOpenAI
        from langchain.schema import HumanMessage
        from utils import get_api_key
        
        api_key = get_api_key('openai')
        if not api_key:
            logger.warning(\"⚠️ Skipping Chat model test - no API key\")
            return True
        
        logger.info(\"🧪 Testing Chat model connection...\")
        
        chat = ChatOpenAI(
            openai_api_key=api_key,
            max_tokens=10,
            temperature=0
        )
        
        message = HumanMessage(content=\"Hi\")
        response = chat([message])
        
        if response and hasattr(response, 'content') and len(response.content.strip()) > 0:
            logger.info(\"✅ Chat model - Working\")
            return True
        else:
            logger.error(\"❌ Chat model - Empty response\")
            return False
            
    except Exception as e:
        logger.error(f\"❌ Chat model test failed: {e}\")
        return False


def show_configuration():
    \"\"\"Show current configuration.\"\"\"
    logger = get_logger(__name__)
    
    config = load_config()
    
    print(\"\n\" + \"=\"*50)
    print(\"CONFIGURATION\")
    print(\"=\"*50)
    print(f\"Default Model: {config['default_model']}\")
    print(f\"Temperature: {config['default_temperature']}\")
    print(f\"Max Tokens: {config['default_max_tokens']}\")
    print(f\"Log Level: {config['log_level']}\")
    print(f\"Debug Mode: {config['debug']}\")
    print(\"=\"*50)


def main():
    \"\"\"Main validation function.\"\"\"
    setup_logging()
    logger = get_logger(__name__)
    
    print(\"🔍 LangChain Playbook Setup Validation\")
    print(\"=\" * 50)
    
    checks = [
        (\"Python Version\", check_python_version),
        (\"Required Packages\", check_required_packages),
        (\"API Keys\", check_api_keys),
        (\"Basic LLM\", test_basic_llm),
        (\"Chat Model\", test_chat_model)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        logger.info(f\"\n🔍 Checking {check_name}...\")
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            logger.error(f\"❌ {check_name} check failed: {e}\")
            results.append((check_name, False))
    
    # Summary
    print(\"\n\" + \"=\"*50)
    print(\"VALIDATION SUMMARY\")
    print(\"=\"*50)
    
    all_passed = True
    for check_name, passed in results:
        status = \"✅ PASS\" if passed else \"❌ FAIL\"
        print(f\"{check_name:20} {status}\")
        if not passed:
            all_passed = False
    
    print(\"=\"*50)
    
    if all_passed:
        logger.info(\"🎉 All checks passed! You're ready to start learning LangChain.\")
        show_configuration()
        print(\"\n🚀 Next steps:\")
        print(\"1. Try: python hello_langchain.py\")
        print(\"2. Explore: python basic_chat.py\")
        print(\"3. Continue to: ../02_models/\")
    else:
        logger.error(\"❌ Some checks failed. Please fix the issues above.\")
        print(\"\n💡 Common solutions:\")
        print(\"- Install packages: pip install -r requirements.txt\")
        print(\"- Set up API keys: see docs/api-keys.md\")
        print(\"- Check internet connection\")


if __name__ == \"__main__\":
    main()