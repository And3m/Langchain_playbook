# LangChain Intermediate - Advanced Features and Patterns

Welcome to the intermediate section! This builds on the basics to introduce advanced LangChain features including memory, agents, RAG, monitoring, and asynchronous operations.

## 🎯 Learning Path Overview

```
memory → agents → retrieval → callbacks → async
   ↓       ↓        ↓          ↓         ↓
Context  Tools   Knowledge  Monitor  Performance
```

## 📚 Modules in this Section

### [01_memory](01_memory/) - Conversational Context
**💭 Add memory and context to your applications**

- **conversational_memory.py**: Buffer, summary, window, and custom memory types
- Different memory strategies for various use cases
- Memory integration with chains and conversations

**Key Concepts**: Context preservation, conversation history, memory management

### [02_agents](02_agents/) - Tool-Using AI Systems  
**🤖 Build AI that can use tools and make decisions**

- **basic_agents.py**: ReAct agents, custom tools, planning, safety
- Agent types and selection strategies
- Tool integration and execution patterns

**Key Concepts**: Reasoning and acting, tool usage, agent architectures

### [03_retrieval](03_retrieval/) - RAG and Knowledge Integration
**🔍 Enhance LLMs with external knowledge**

- **rag_basics.py**: Document processing, vector stores, retrieval strategies
- Text splitting and embedding techniques
- Conversational RAG with memory integration

**Key Concepts**: Vector embeddings, similarity search, knowledge augmentation

### [04_callbacks](04_callbacks/) - Monitoring and Observability
**📊 Monitor and debug your LLM applications**

- **callback_monitoring.py**: Token tracking, performance monitoring, streaming
- Custom callback implementations
- Production monitoring strategies

**Key Concepts**: Observability, cost tracking, performance optimization

### [05_async](05_async/) - High-Performance Operations
**⚡ Build scalable, concurrent applications**

- **async_operations.py**: Async LLMs, concurrent processing, batch optimization
- Error handling in async contexts
- Performance comparison and best practices

**Key Concepts**: Concurrency, asynchronous programming, performance optimization

## 🎓 What You'll Learn

After completing this section, you'll be able to:

1. **Build Conversational AI**: Applications that remember context
2. **Create Intelligent Agents**: AI that can use tools and make decisions  
3. **Implement RAG Systems**: Enhance LLMs with external knowledge
4. **Monitor Applications**: Track performance, costs, and behavior
5. **Optimize Performance**: Use async operations for high throughput
6. **Production Patterns**: Best practices for real-world applications

## 📋 Prerequisites

### Required Knowledge
- Completed [basics section](../basics/) 
- Understanding of LLMs, prompts, chains, and output parsers
- Basic Python async/await concepts (for async module)
- Familiarity with APIs and external services

### Technical Requirements
- API keys for LLM providers
- Understanding of vector databases (for RAG)
- Basic knowledge of monitoring concepts
- Python 3.8+ with asyncio support

## 🚀 Quick Start Guide

### Option 1: Follow the Learning Path
```bash
# Start with memory
cd intermediate/01_memory
python conversational_memory.py

# Continue through agents
cd ../02_agents  
python basic_agents.py

# And so on...
```

### Option 2: Focus on Specific Topics
```bash
# Interested in RAG?
cd intermediate/03_retrieval
python rag_basics.py

# Want to monitor your apps?
cd ../04_callbacks
python callback_monitoring.py
```

### Option 3: Performance-First Approach
```bash
# Start with async for high-performance apps
cd intermediate/05_async
python async_operations.py

# Then add monitoring
cd ../04_callbacks
python callback_monitoring.py
```

## 💡 Key Concepts by Module

### Memory Strategies
| Type | Use Case | Pros | Cons |
|------|----------|------|------|
| **Buffer** | Short conversations | Complete context | Can become long |
| **Summary** | Long conversations | Efficient | May lose details |
| **Window** | Real-time chat | Fixed size | Loses older context |
| **Custom** | Specialized needs | Tailored | Development time |

### Agent Architectures
| Type | Best For | Reasoning Style |
|------|----------|----------------|
| **ReAct** | General problem solving | Reason → Act → Observe |
| **Plan-Execute** | Complex tasks | Plan → Execute steps |
| **Conversational** | Multi-turn interactions | Context-aware reasoning |
| **Custom** | Domain-specific | Application-specific logic |

### RAG Components
```
Documents → Text Splitting → Embeddings → Vector Store
     ↓            ↓             ↓           ↓
  Loading     Chunking      Encoding    Storage
                                          ↓
Query → Retrieval → Context + Query → LLM → Response
```

### Monitoring Layers
- **Token Usage**: Cost tracking and optimization
- **Performance**: Latency and throughput monitoring
- **Errors**: Failure detection and recovery
- **Business Metrics**: User satisfaction and outcomes

## 🔍 Common Patterns

### 1. Memory-Enabled RAG Agent
```python
# Combine memory, RAG, and agents
memory = ConversationBufferMemory()
vectorstore = FAISS.from_documents(documents, embeddings)
tools = [retrieval_tool, calculator_tool]
agent = initialize_agent(tools, llm, memory=memory)
```

### 2. Monitored Async Pipeline
```python
# High-performance pipeline with monitoring
monitor = PerformanceMonitor()
llm = ChatOpenAI(callbacks=[monitor])

async def process_batch(items):
    tasks = [llm.agenerate([item]) for item in items]
    return await asyncio.gather(*tasks)
```

### 3. Resilient Conversational RAG
```python
# Robust system with error handling
try:
    context = retriever.get_relevant_documents(query)
    response = conversation_chain.run(
        input=query, 
        context=context
    )
except Exception as e:
    logger.error(f\"RAG error: {e}\")
    response = fallback_chain.run(input=query)
```

## 🛠️ Development Tips

### Memory Management
- Choose memory type based on conversation length
- Monitor memory size and clear when needed
- Test with realistic conversation flows
- Consider cost implications of memory strategies

### Agent Development
- Start with simple, safe tools
- Implement proper error handling
- Test agent reasoning with edge cases
- Use sandboxed environments for code execution

### RAG Optimization
- Experiment with chunk sizes and overlap
- Tune retrieval parameters (k, score thresholds)
- Use appropriate embedding models for your domain
- Monitor retrieval quality and user feedback

### Monitoring Setup
- Implement logging from day one
- Track key metrics (cost, latency, errors)
- Set up alerts for critical failures
- Monitor user satisfaction and outcomes

### Async Best Practices
- Use async for I/O-bound operations
- Implement proper error handling
- Consider rate limiting and backpressure
- Monitor resource usage and performance

## 🚨 Common Pitfalls

❌ **Memory Issues**:
- Not clearing memory in long conversations
- Using buffer memory for all scenarios
- Ignoring memory costs

❌ **Agent Problems**:
- Giving agents unrestricted access
- Poor error handling in tools
- Not validating agent outputs

❌ **RAG Mistakes**:
- Poor document chunking strategies
- Not tuning retrieval parameters
- Ignoring retrieval quality

❌ **Monitoring Gaps**:
- Not tracking costs and usage
- Missing error monitoring
- No performance baselines

❌ **Async Issues**:
- Using async for CPU-bound tasks
- Poor error handling
- Not managing concurrency limits

## 📈 Progress Tracking

Track your intermediate learning:

- [ ] 01_memory: Conversational context and state management
- [ ] 02_agents: Tool-using AI and decision making
- [ ] 03_retrieval: RAG and knowledge integration
- [ ] 04_callbacks: Monitoring and observability  
- [ ] 05_async: High-performance concurrent operations

## 🎯 What's Next?

After mastering intermediate concepts:

### Advanced Topics
- **[Advanced](../advanced/)**: Expert-level patterns and optimization
- **Custom Tools**: Build specialized agent capabilities
- **Multi-Agent Systems**: Coordinated AI systems
- **Evaluation**: Testing and benchmarking
- **Deployment**: Production-ready applications

### Real Projects  
- **[Projects](../projects/)**: Build real-world applications
- **Chatbot**: Conversational AI with memory and tools
- **Document Q&A**: RAG-powered knowledge assistant
- **Code Assistant**: AI-powered development helper

### Interactive Learning
- **[Notebooks](../notebooks/)**: Jupyter-based tutorials
- **Exploration**: Experiment with concepts
- **Tutorials**: Step-by-step guided learning

## 💬 Success Tips

1. **Build Incrementally**: Start simple, add complexity gradually
2. **Test Thoroughly**: Each component should work independently
3. **Monitor Everything**: Implement observability from the start
4. **Handle Errors**: Plan for failures and edge cases
5. **Optimize Gradually**: Get it working, then make it fast
6. **Learn by Building**: Apply concepts to real problems

---

**🎉 Ready for advanced LangChain features? Start with [01_memory](01_memory/) and build sophisticated AI applications! 🚀**