# Troubleshooting Guide 🔧

Common issues and solutions for the LangChain Playbook. This guide helps you quickly resolve problems and get back to building amazing AI applications.

## 📋 Quick Diagnostics

### Health Check Commands
```bash
# Check Python version
python --version  # Should be 3.8+

# Verify LangChain installation
python -c "import langchain; print(langchain.__version__)"

# Test API connectivity
python -c "import os; print('API Key:', 'Found' if os.getenv('OPENAI_API_KEY') else 'Missing')"

# Check utils availability
python -c "import sys; sys.path.append('utils'); from config import get_api_key; print('Utils:', 'OK')"
```

---

## 🔑 API Key Issues

### Problem: "OpenAI API key not found"

**Symptoms:**
- Error messages about missing API keys
- Applications fail to start
- Demo mode activates unexpectedly

**Solutions:**

1. **Environment Variable Method:**
   ```bash
   # Windows (Command Prompt)
   set OPENAI_API_KEY=your_key_here
   
   # Windows (PowerShell)
   $env:OPENAI_API_KEY="your_key_here"
   
   # macOS/Linux
   export OPENAI_API_KEY=your_key_here
   ```

2. **Create .env File:**
   ```bash
   # In project root directory
   echo "OPENAI_API_KEY=your_key_here" > .env
   ```

3. **Python-dotenv Method:**
   ```python
   # Install python-dotenv
   pip install python-dotenv
   
   # In your script
   from dotenv import load_dotenv
   load_dotenv()
   ```

**Verification:**
```python
from utils.config import get_api_key
api_key = get_api_key('openai')
print("✅ API key loaded" if api_key else "❌ API key missing")
```

### Problem: "Invalid API key"

**Symptoms:**
- Authentication errors from OpenAI
- 401 Unauthorized responses

**Solutions:**
1. **Check Key Format:** Should start with `sk-`
2. **Verify Key Active:** Test in OpenAI playground
3. **Check Billing:** Ensure account has credits
4. **Regenerate Key:** Create new key in OpenAI dashboard

---

## 📦 Installation Issues

### Problem: "ModuleNotFoundError: No module named 'langchain'"

**Solutions:**
```bash
# Install LangChain
pip install langchain

# Or install all requirements
pip install -r requirements.txt

# Verify installation
python -c "import langchain; print('LangChain version:', langchain.__version__)"
```

### Problem: "No module named 'utils'"

**Symptoms:**
- Import errors when running examples
- Utils functions not found

**Solutions:**

1. **Check Working Directory:**
   ```bash
   # Make sure you're in the project root
   pwd  # Should show .../Langchain-Playbook
   ls   # Should show basics/, intermediate/, projects/, utils/
   ```

2. **Add Utils to Python Path:**
   ```python
   import sys
   from pathlib import Path
   sys.path.append(str(Path(__file__).parent.parent / 'utils'))
   ```

3. **Run from Correct Directory:**
   ```bash
   # Run from project root
   cd /path/to/Langchain-Playbook
   python basics/01_getting_started/hello_langchain.py
   ```

### Problem: "pip install fails"

**Solutions:**
```bash
# Update pip
python -m pip install --upgrade pip

# Use specific Python version
python3.11 -m pip install langchain

# Install with user flag
pip install --user langchain

# Clear pip cache
pip cache purge
```

---

## 🐍 Python Environment Issues

### Problem: "Wrong Python version"

**Symptoms:**
- Compatibility errors
- Syntax errors with modern Python features

**Solutions:**
```bash
# Check Python version
python --version  # Should be 3.8+

# Use specific Python version
python3.11 -m pip install langchain
python3.11 your_script.py

# Create virtual environment with specific version
python3.11 -m venv langchain_env
source langchain_env/bin/activate  # macOS/Linux
langchain_env\Scripts\activate     # Windows
```

### Problem: "Virtual environment issues"

**Solutions:**
```bash
# Create new virtual environment
python -m venv .venv

# Activate virtual environment
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# Verify activation
which python  # Should show venv path

# Install requirements in venv
pip install -r requirements.txt
```

---

## 🔗 LangChain Specific Issues

### Problem: "Rate limit exceeded"

**Symptoms:**
- 429 HTTP errors
- "Too many requests" messages

**Solutions:**
1. **Add Delays:**
   ```python
   import time
   
   for prompt in prompts:
       response = llm(prompt)
       time.sleep(1)  # Add 1-second delay
   ```

2. **Implement Retry Logic:**
   ```python
   from utils.decorators import retry_decorator
   
   @retry_decorator(max_attempts=3, delay=2.0)
   def safe_llm_call(prompt):
       return llm(prompt)
   ```

3. **Use Different Model:**
   ```python
   # Switch to less rate-limited model
   llm = ChatOpenAI(model_name="gpt-3.5-turbo")  # Instead of gpt-4
   ```

### Problem: "Context length exceeded"

**Symptoms:**
- Token limit errors
- Truncated responses

**Solutions:**
1. **Reduce Input Length:**
   ```python
   # Truncate long prompts
   max_tokens = 3000
   if len(prompt) > max_tokens:
       prompt = prompt[:max_tokens]
   ```

2. **Use Summarization:**
   ```python
   from langchain.memory import ConversationSummaryMemory
   
   memory = ConversationSummaryMemory(llm=llm)
   ```

3. **Chunk Processing:**
   ```python
   from langchain.text_splitter import RecursiveCharacterTextSplitter
   
   splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
   chunks = splitter.split_text(long_text)
   ```

### Problem: "Memory issues with conversations"

**Solutions:**
1. **Use Window Memory:**
   ```python
   from langchain.memory import ConversationBufferWindowMemory
   
   memory = ConversationBufferWindowMemory(k=5)  # Keep last 5 exchanges
   ```

2. **Clear Memory Periodically:**
   ```python
   if len(memory.chat_memory.messages) > 20:
       memory.clear()
   ```

---

## 🗄️ Vector Database Issues

### Problem: "FAISS installation fails"

**Solutions:**
```bash
# Install CPU version
pip install faiss-cpu

# Or GPU version (if CUDA available)
pip install faiss-gpu

# Alternative: Use ChromaDB
pip install chromadb
```

### Problem: "Vector store persistence"

**Solutions:**
```python
# Save FAISS index
vectorstore.save_local("faiss_index")

# Load FAISS index
from langchain.vectorstores import FAISS
vectorstore = FAISS.load_local("faiss_index", embeddings)

# Use persistent ChromaDB
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
```

---

## 🚀 Performance Issues

### Problem: "Slow response times"

**Solutions:**
1. **Optimize Model Selection:**
   ```python
   # Use faster models for simple tasks
   llm = ChatOpenAI(model_name="gpt-3.5-turbo")  # Faster than gpt-4
   ```

2. **Implement Caching:**
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=100)
   def cached_llm_call(prompt):
       return llm(prompt)
   ```

3. **Reduce Token Usage:**
   ```python
   llm = ChatOpenAI(
       model_name="gpt-3.5-turbo",
       max_tokens=150,  # Reduce for faster responses
       temperature=0.3  # Lower temperature for consistency
   )
   ```

### Problem: "Memory usage too high"

**Solutions:**
1. **Clear Unused Variables:**
   ```python
   import gc
   
   del large_variable
   gc.collect()
   ```

2. **Process in Batches:**
   ```python
   def process_documents_batch(documents, batch_size=10):
       for i in range(0, len(documents), batch_size):
           batch = documents[i:i+batch_size]
           process_batch(batch)
   ```

---

## 🔒 Security Issues

### Problem: "API key exposure"

**Prevention:**
```python
# ❌ Never do this
api_key = "sk-actual-key-here"

# ✅ Use environment variables
api_key = os.getenv('OPENAI_API_KEY')

# ✅ Use .env files (don't commit to git)
from dotenv import load_dotenv
load_dotenv()
```

**Git Safety:**
```bash
# Add to .gitignore
echo ".env" >> .gitignore
echo "*.key" >> .gitignore

# Remove committed secrets
git filter-branch --force --index-filter \
'git rm --cached --ignore-unmatch .env' --prune-empty --tag-name-filter cat -- --all
```

---

## 🐛 Common Error Messages

### "SSL: CERTIFICATE_VERIFY_FAILED"

**Solutions:**
```python
# Temporary fix (not recommended for production)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Better solution: Update certificates
# macOS: Update certificates in Keychain
# Windows: Update Windows and certificates
# Linux: Update ca-certificates package
```

### "Connection timeout"

**Solutions:**
```python
# Increase timeout
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    openai_api_key=api_key,
    request_timeout=60  # Increase timeout to 60 seconds
)

# Add retry logic
import time

def retry_with_backoff(func, max_retries=3):
    for i in range(max_retries):
        try:
            return func()
        except Exception as e:
            if i == max_retries - 1:
                raise e
            time.sleep(2 ** i)  # Exponential backoff
```

### "JSON decode error"

**Solutions:**
```python
# Handle malformed responses
try:
    result = json.loads(response)
except json.JSONDecodeError:
    # Clean response text
    cleaned = response.strip().replace('\n', ' ')
    result = json.loads(cleaned)
```

---

## 📱 Platform-Specific Issues

### Windows Issues

1. **Path Separators:**
   ```python
   # Use pathlib for cross-platform paths
   from pathlib import Path
   file_path = Path("data") / "documents" / "file.txt"
   ```

2. **PowerShell Execution Policy:**
   ```powershell
   # Allow script execution
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

### macOS Issues

1. **SSL Certificates:**
   ```bash
   # Install certificates
   /Applications/Python\ 3.x/Install\ Certificates.command
   ```

2. **Command Line Tools:**
   ```bash
   # Install Xcode command line tools
   xcode-select --install
   ```

### Linux Issues

1. **Missing System Dependencies:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install build-essential python3-dev
   
   # CentOS/RHEL
   sudo yum install gcc python3-devel
   ```

---

## 🧪 Testing Solutions

### Verify Fix Commands

```bash
# Test basic functionality
python -c "
import sys
sys.path.append('utils')
from config import get_api_key
from langchain.chat_models import ChatOpenAI

api_key = get_api_key('openai')
if api_key:
    llm = ChatOpenAI(openai_api_key=api_key)
    print('✅ Everything working!')
else:
    print('⚠️ API key missing but imports work')
"

# Test specific project
cd projects/chatbot
python chatbot_app.py

# Test notebook functionality
jupyter notebook --version
```

---

## 🆘 Getting Help

### Before Asking for Help

1. **Check This Guide:** Look for your specific error message
2. **Review Logs:** Check console output for detailed errors
3. **Verify Environment:** Ensure correct Python version and dependencies
4. **Test Minimal Example:** Isolate the issue with simple code

### Where to Get Help

1. **GitHub Issues:** Report bugs and feature requests
2. **Community Discussions:** Ask questions and share solutions
3. **LangChain Discord:** Real-time community support
4. **Stack Overflow:** Tag questions with `langchain`

### When Reporting Issues

Include:
- **Error Message:** Full traceback
- **Environment:** Python version, OS, LangChain version
- **Code Sample:** Minimal reproducible example
- **Steps Taken:** What you've already tried

**Template:**
```
**Environment:**
- Python version: 3.11.0
- LangChain version: 0.2.0
- OS: Windows 11

**Error:**
[Full error traceback]

**Code:**
[Minimal code that reproduces the issue]

**Steps Tried:**
- Reinstalled LangChain
- Checked API key
- Cleared Python cache
```

---

**Remember:** Most issues have simple solutions. Check the basics first: API keys, Python version, dependencies, and working directory. The LangChain community is helpful and responsive! 🚀