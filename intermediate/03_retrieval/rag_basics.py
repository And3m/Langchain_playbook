#!/usr/bin/env python3
\"\"\"
RAG (Retrieval Augmented Generation) - Enhancing LLMs with External Knowledge

This example demonstrates:
1. Document loading and preprocessing
2. Text splitting strategies
3. Vector embeddings and storage
4. Similarity search and retrieval
5. RAG chain implementation
6. Advanced retrieval techniques

Key concepts:
- Combining retrieval with generation
- Vector databases and embeddings
- Document processing pipelines
- Contextual question answering
\"\"\"

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import os

# Add utils to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import setup_logging, get_logger, get_api_key, timing_decorator
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory


# Sample documents for demonstration
SAMPLE_DOCUMENTS = {
    \"langchain_intro.txt\": \"\"\"
LangChain is a framework for developing applications powered by language models. 
It enables applications that are data-aware and agentic, allowing language models 
to connect with other sources of data and interact with their environment.

The main value propositions of LangChain are:
1. Components: Modular abstractions for working with language models
2. Chains: Structured assemblies of components for accomplishing specific tasks
3. Agents: Allow LLMs to interact with their environment via decision making
4. Memory: Maintain state between calls to an LLM or chain
5. Callbacks: Hook into any stage of the LLM application for logging, monitoring, streaming

LangChain provides integrations with many LLM providers including OpenAI, Anthropic, 
Google, and Hugging Face. It also supports various vector databases like Pinecone, 
Chroma, and FAISS for retrieval augmented generation.
\"\"\",
    \"machine_learning.txt\": \"\"\"
Machine Learning is a subset of artificial intelligence that enables computers to 
learn and improve from experience without being explicitly programmed. It focuses 
on developing algorithms that can analyze data and make predictions or decisions.

Types of Machine Learning:
1. Supervised Learning: Uses labeled training data to learn a mapping function
2. Unsupervised Learning: Finds hidden patterns in data without labeled examples
3. Reinforcement Learning: Learns through interaction with an environment

Common algorithms include:
- Linear Regression and Logistic Regression
- Decision Trees and Random Forests
- Support Vector Machines (SVM)
- Neural Networks and Deep Learning
- K-Means Clustering
- Principal Component Analysis (PCA)

Machine learning is used in many applications including image recognition, 
natural language processing, recommendation systems, fraud detection, and autonomous vehicles.
\"\"\",
    \"python_programming.txt\": \"\"\"
Python is a high-level, interpreted programming language known for its simplicity 
and readability. It was created by Guido van Rossum and first released in 1991.

Key features of Python:
1. Easy to learn and use
2. Interpreted language - no compilation needed
3. Object-oriented programming support
4. Extensive standard library
5. Cross-platform compatibility
6. Large community and ecosystem

Python is widely used for:
- Web development (Django, Flask)
- Data science and analytics (Pandas, NumPy, Matplotlib)
- Machine learning (Scikit-learn, TensorFlow, PyTorch)
- Automation and scripting
- Desktop applications
- Game development

Python's syntax emphasizes readability and simplicity, making it an excellent 
choice for beginners and experienced programmers alike.
\"\"\"
}


def create_sample_documents() -> List[Document]:
    \"\"\"Create sample documents for demonstration.\"\"\"
    documents = []
    
    for filename, content in SAMPLE_DOCUMENTS.items():
        doc = Document(
            page_content=content,
            metadata={\"source\": filename, \"type\": \"text\"}
        )
        documents.append(doc)
    
    return documents


def demonstrate_text_splitting():
    \"\"\"Demonstrate different text splitting strategies.\"\"\"
    logger = get_logger(__name__)
    logger.info(\"✂️ Text Splitting Strategies\")
    
    # Get sample text
    sample_text = SAMPLE_DOCUMENTS[\"langchain_intro.txt\"]
    
    print(\"\n\" + \"=\"*70)
    print(\"TEXT SPLITTING STRATEGIES\")
    print(\"=\"*70)
    print(f\"Original text length: {len(sample_text)} characters\n\")
    
    # Different splitters
    splitters = {
        \"Character Splitter\": CharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            separator=\"\n\"
        ),
        \"Recursive Character Splitter\": RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20
        ),
        \"Token Splitter\": TokenTextSplitter(
            chunk_size=50,
            chunk_overlap=10
        )
    }
    
    for splitter_name, splitter in splitters.items():
        chunks = splitter.split_text(sample_text)
        
        print(f\"📄 {splitter_name}:\")
        print(f\"   Number of chunks: {len(chunks)}\")
        print(f\"   Average chunk size: {sum(len(chunk) for chunk in chunks) // len(chunks)} chars\")
        print(f\"   First chunk: {chunks[0][:100]}...\")
        print(\"-\" * 50)
    
    print(\"\n💡 Splitting Strategy Guidelines:\")
    guidelines = [
        \"Recursive splitter is usually the best default choice\",
        \"Use semantic boundaries when possible (sentences, paragraphs)\",
        \"Consider token limits of your embedding model\",
        \"Balance chunk size with retrieval granularity\",
        \"Include overlap to maintain context continuity\"
    ]
    
    for guideline in guidelines:
        print(f\"• {guideline}\")
    
    print(\"=\"*70)


@timing_decorator
def demonstrate_vector_store_setup():
    \"\"\"Demonstrate vector store setup and document indexing.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping vector store demo\")
        return None
    
    logger.info(\"🗃️ Vector Store Setup\")
    
    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    # Prepare documents
    documents = create_sample_documents()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    texts = text_splitter.split_documents(documents)
    
    print(\"\n\" + \"=\"*70)
    print(\"VECTOR STORE SETUP AND INDEXING\")
    print(\"=\"*70)
    print(f\"Processing {len(documents)} documents into {len(texts)} chunks\n\")
    
    try:
        # Create vector store with FAISS (local, no external dependencies)
        print(\"📚 Creating FAISS vector store...\")
        vectorstore = FAISS.from_documents(texts, embeddings)
        print(f\"✅ Vector store created with {len(texts)} document chunks\")
        
        # Test similarity search
        print(\"\n🔍 Testing similarity search:\")
        query = \"What is LangChain?\"
        docs = vectorstore.similarity_search(query, k=2)
        
        print(f\"Query: {query}\")
        for i, doc in enumerate(docs, 1):
            print(f\"\nResult {i}:\")
            print(f\"Source: {doc.metadata.get('source', 'Unknown')}\")
            print(f\"Content: {doc.page_content[:200]}...\")
        
        print(\"\n💡 Vector stores enable semantic similarity search.\")
        print(\"Documents are converted to embeddings for efficient retrieval.\")
        
        return vectorstore
        
    except Exception as e:
        logger.error(f\"Error setting up vector store: {e}\")
        return None
    
    finally:
        print(\"=\"*70)


@timing_decorator
def demonstrate_basic_rag(vectorstore):
    \"\"\"Demonstrate basic RAG implementation.\"\"\"
    logger = get_logger(__name__)
    
    if not vectorstore:
        logger.warning(\"⚠️ No vector store available, skipping RAG demo\")
        return
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping RAG demo\")
        return
    
    logger.info(\"🔄 Basic RAG Implementation\")
    
    # Create LLM
    llm = ChatOpenAI(openai_api_key=api_key, temperature=0.3, max_tokens=200)
    
    # Create retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=\"stuff\",
        retriever=vectorstore.as_retriever(search_kwargs={\"k\": 3}),
        return_source_documents=True,
        verbose=True
    )
    
    print(\"\n\" + \"=\"*70)
    print(\"BASIC RAG (RETRIEVAL AUGMENTED GENERATION)\")
    print(\"=\"*70)
    print(\"RAG combines retrieval with generation for informed responses\n\")
    
    # Test queries
    queries = [
        \"What is LangChain and what are its main components?\",
        \"What are the different types of machine learning?\",
        \"Why is Python popular for programming?\",
        \"How do chains work in LangChain?\"
    ]
    
    for i, query in enumerate(queries, 1):
        try:
            print(f\"❓ Query {i}: {query}\")
            result = qa_chain({\"query\": query})
            
            print(f\"🤖 Answer: {result['result']}\")
            
            print(f\"📚 Sources used:\")
            for j, doc in enumerate(result['source_documents'], 1):
                source = doc.metadata.get('source', 'Unknown')
                print(f\"   {j}. {source}: {doc.page_content[:100]}...\")
            
            print(\"-\" * 50)
            
        except Exception as e:
            logger.error(f\"Error processing query {i}: {e}\")
    
    print(\"\n💡 RAG provides factual, grounded responses by combining\")
    print(\"retrieval of relevant documents with language generation.\")
    print(\"=\"*70)


@timing_decorator
def demonstrate_conversational_rag(vectorstore):
    \"\"\"Demonstrate conversational RAG with memory.\"\"\"
    logger = get_logger(__name__)
    
    if not vectorstore:
        logger.warning(\"⚠️ No vector store available, skipping conversational RAG demo\")
        return
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping conversational RAG demo\")
        return
    
    logger.info(\"💬 Conversational RAG with Memory\")
    
    # Create LLM
    llm = ChatOpenAI(openai_api_key=api_key, temperature=0.3, max_tokens=200)
    
    # Create memory
    memory = ConversationBufferMemory(
        memory_key=\"chat_history\",
        return_messages=True
    )
    
    # Create conversational retrieval chain
    conv_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={\"k\": 2}),
        memory=memory,
        return_source_documents=True,
        verbose=True
    )
    
    print(\"\n\" + \"=\"*70)
    print(\"CONVERSATIONAL RAG WITH MEMORY\")
    print(\"=\"*70)
    print(\"Maintains conversation context while using retrieved information\n\")
    
    # Conversational flow
    conversation = [
        \"What is machine learning?\",
        \"What are some common algorithms?\",
        \"How is it different from traditional programming?\",
        \"Can you give me examples of applications?\"
    ]
    
    for i, question in enumerate(conversation, 1):
        try:
            print(f\"👤 Turn {i}: {question}\")
            result = conv_chain({\"question\": question})
            
            print(f\"🤖 Response: {result['answer']}\")
            
            if result.get('source_documents'):
                sources = [doc.metadata.get('source', 'Unknown') for doc in result['source_documents']]
                print(f\"📚 Sources: {', '.join(set(sources))}\")
            
            print(\"-\" * 50)
            
        except Exception as e:
            logger.error(f\"Error in conversation turn {i}: {e}\")
    
    print(\"\n💡 Conversational RAG maintains context across turns\")
    print(\"while still grounding responses in retrieved documents.\")
    print(\"=\"*70)


def demonstrate_custom_rag_prompt():
    \"\"\"Demonstrate custom RAG prompts and templates.\"\"\"
    logger = get_logger(__name__)
    logger.info(\"📝 Custom RAG Prompts\")
    
    print(\"\n\" + \"=\"*70)
    print(\"CUSTOM RAG PROMPTS AND TEMPLATES\")
    print(\"=\"*70)
    
    # Default RAG prompt
    default_prompt = \"\"\"
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:
\"\"\"
    
    # Custom prompts for different use cases
    custom_prompts = {
        \"Technical Documentation\": \"\"\"
You are a technical documentation assistant. Use the provided context to give 
accurate, detailed answers. Include code examples when relevant.

Context: {context}

Question: {question}

Detailed Answer:
\"\"\",
        \"Educational Tutor\": \"\"\"
You are a helpful tutor. Use the context to explain concepts clearly and simply.
Break down complex topics into easy-to-understand parts.

Context: {context}

Student Question: {question}

Explanation:
\"\"\",
        \"Research Assistant\": \"\"\"
You are a research assistant. Analyze the provided context and give a comprehensive 
answer. Cite specific information from the sources when possible.

Source Material: {context}

Research Question: {question}

Analysis:
\"\"\"
    }
    
    print(\"📋 RAG Prompt Templates:\n\")
    
    print(\"🔧 Default RAG Prompt:\")
    print(default_prompt)
    print(\"-\" * 50)
    
    for prompt_name, prompt_template in custom_prompts.items():
        print(f\"\n🎯 {prompt_name}:\")
        print(prompt_template)
        print(\"-\" * 50)
    
    print(\"\n💡 Custom Prompts Guidelines:\")
    guidelines = [
        \"Define the assistant's role clearly\",
        \"Specify how to handle unknown information\",
        \"Include formatting instructions for responses\",
        \"Adapt tone and style to your use case\",
        \"Test prompts with various question types\",
        \"Include context usage instructions\"
    ]
    
    for guideline in guidelines:
        print(f\"• {guideline}\")
    
    print(\"=\"*70)


def demonstrate_retrieval_strategies():
    \"\"\"Demonstrate different retrieval strategies and optimization.\"\"\"
    logger = get_logger(__name__)
    logger.info(\"🎯 Retrieval Strategies and Optimization\")
    
    print(\"\n\" + \"=\"*70)
    print(\"RETRIEVAL STRATEGIES AND OPTIMIZATION\")
    print(\"=\"*70)
    
    strategies = {
        \"Similarity Search\": {
            \"description\": \"Basic cosine similarity between query and document embeddings\",
            \"parameters\": \"k (number of documents to retrieve)\",
            \"best_for\": \"General purpose retrieval\",
            \"example\": \"vectorstore.similarity_search(query, k=4)\"
        },
        \"MMR (Maximal Marginal Relevance)\": {
            \"description\": \"Balances relevance with diversity to avoid redundant results\",
            \"parameters\": \"k, fetch_k, lambda_mult (diversity parameter)\",
            \"best_for\": \"Avoiding redundant information\",
            \"example\": \"vectorstore.max_marginal_relevance_search(query, k=4, lambda_mult=0.5)\"
        },
        \"Similarity with Score\": {
            \"description\": \"Returns documents with similarity scores for filtering\",
            \"parameters\": \"k, score_threshold\",
            \"best_for\": \"Quality filtering based on relevance scores\",
            \"example\": \"vectorstore.similarity_search_with_score(query, k=4)\"
        },
        \"Metadata Filtering\": {
            \"description\": \"Filters documents based on metadata before similarity search\",
            \"parameters\": \"filter dictionary, k\",
            \"best_for\": \"Domain-specific or filtered retrieval\",
            \"example\": \"retriever.get_relevant_documents(query, filter={'type': 'technical'})\"
        }
    }
    
    for strategy_name, details in strategies.items():
        print(f\"\n🔍 {strategy_name}:\")
        print(f\"   Description: {details['description']}\")
        print(f\"   Parameters: {details['parameters']}\")
        print(f\"   Best for: {details['best_for']}\")
        print(f\"   Example: {details['example']}\")
        print(\"-\" * 50)
    
    print(\"\n⚡ Optimization Tips:\")
    optimization_tips = [
        \"Experiment with different chunk sizes and overlap\",
        \"Use appropriate embedding models for your domain\",
        \"Consider metadata for filtering and organization\",
        \"Tune retrieval parameters (k, score thresholds)\",
        \"Monitor retrieval quality and user feedback\",
        \"Use hybrid search (semantic + keyword) when available\",
        \"Cache embeddings for frequently accessed documents\",
        \"Implement query expansion for better recall\"
    ]
    
    for tip in optimization_tips:
        print(f\"• {tip}\")
    
    print(\"\n📊 Performance Considerations:\")
    performance_notes = [
        \"Larger chunk sizes: Better context, slower retrieval\",
        \"More retrieved documents: Better recall, higher cost\",
        \"Higher embedding dimensions: Better quality, more storage\",
        \"Vector store choice impacts speed and scalability\",
        \"Consider approximate nearest neighbor for large datasets\"
    ]
    
    for note in performance_notes:
        print(f\"• {note}\")
    
    print(\"=\"*70)


def main():
    \"\"\"Main function demonstrating RAG concepts.\"\"\"
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info(\"🚀 Starting RAG (Retrieval Augmented Generation) Demonstration\")
    
    try:
        # Run all demonstrations
        demonstrate_text_splitting()
        vectorstore = demonstrate_vector_store_setup()
        demonstrate_basic_rag(vectorstore)
        demonstrate_conversational_rag(vectorstore)
        demonstrate_custom_rag_prompt()
        demonstrate_retrieval_strategies()
        
        print(\"\n🎯 RAG Key Takeaways:\")
        print(\"1. RAG combines retrieval with generation for accurate responses\")
        print(\"2. Document preprocessing and chunking are crucial\")
        print(\"3. Vector embeddings enable semantic similarity search\")
        print(\"4. Different retrieval strategies serve different needs\")
        print(\"5. Custom prompts adapt RAG to specific use cases\")
        print(\"6. Memory enables conversational RAG applications\")
        
        logger.info(\"✅ RAG demonstration completed successfully!\")
        
    except Exception as e:
        logger.error(f\"❌ Error occurred: {e}\")
        logger.info(\"💡 Check your API keys and dependencies\")


if __name__ == \"__main__\":
    main()