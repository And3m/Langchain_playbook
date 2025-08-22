#!/usr/bin/env python3
\"\"\"
Document Q&A System - RAG-Powered Knowledge Assistant

This project demonstrates building a complete document Q&A system with:
1. Document ingestion and preprocessing
2. Vector database setup and management  
3. Advanced retrieval strategies
4. Conversational Q&A with context
5. Source attribution and citation

Features:
- Support for multiple document formats
- Intelligent document chunking
- Vector similarity search with metadata
- Conversational Q&A with memory
- Source citation and confidence scoring
\"\"\"

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add utils to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import setup_logging, get_logger, get_api_key, timing_decorator
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory


# Sample documents for demonstration
SAMPLE_DOCUMENTS = {
    \"langchain_guide.txt\": \"\"\"
LangChain Framework Overview

LangChain is a framework for developing applications powered by language models. 
It enables applications that are data-aware and agentic, connecting language models 
with other sources of data and allowing them to interact with their environment.

Core Components:
1. Models: Various language model integrations including OpenAI, Anthropic, Google.
2. Prompts: Template management and prompt engineering utilities.
3. Chains: Sequences of calls to language models or other utilities.
4. Agents: Systems that use language models to choose which actions to take.
5. Memory: Ways to persist state between calls of a chain or agent.
6. Document Loaders: Utilities for loading various document formats.
7. Vector Stores: Integration with vector databases for similarity search.

Use Cases:
- Question answering over documents
- Chatbots and virtual assistants
- Content generation and summarization
- Code analysis and generation
\"\"\",
    \"ai_ethics.txt\": \"\"\"
Artificial Intelligence Ethics Guidelines

Core Ethical Principles:

1. Fairness and Non-discrimination
AI systems should treat all individuals fairly, avoiding bias and discrimination.

2. Transparency and Explainability
AI systems should be understandable and their decision-making processes explainable.

3. Privacy and Data Protection
Personal data should be handled with care, respecting privacy rights.

4. Human Agency and Oversight
AI systems should augment human capabilities rather than replace human judgment.

5. Robustness and Safety
AI systems should be reliable, secure, and safe in operation.
\"\"\",
    \"ml_basics.txt\": \"\"\"
Machine Learning Fundamentals

Types of Machine Learning:

1. Supervised Learning
Uses labeled training data to learn mapping from inputs to outputs.
Common algorithms: Linear Regression, Decision Trees, Neural Networks

2. Unsupervised Learning
Finds hidden patterns in data without labeled examples.
Key techniques: Clustering, Dimensionality Reduction, Anomaly Detection

3. Reinforcement Learning
Learns through interaction with environment using rewards and penalties.
Applications: Game playing, Robotics, Trading

Key Concepts:
- Training and Testing data splits
- Feature Engineering
- Model Evaluation metrics
- Overfitting and Underfitting
\"\"\"
}


class DocumentQASystem:
    \"\"\"Complete Document Q&A system with RAG.\"\"\"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_api_key('openai')
        self.logger = get_logger(self.__class__.__name__)
        
        if not self.api_key:
            raise ValueError(\"OpenAI API key is required\")
        
        self.vectorstore = None
        self.qa_chain = None
        self.conversational_chain = None
        
        self._setup_components()
    
    def _setup_components(self):
        \"\"\"Initialize LLM and embedding components.\"\"\"
        self.llm = ChatOpenAI(
            openai_api_key=self.api_key,
            temperature=0.2,
            max_tokens=300
        )
        
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        
        self.memory = ConversationBufferMemory(
            memory_key=\"chat_history\",
            return_messages=True,
            output_key=\"answer\"
        )
    
    @timing_decorator
    def setup_knowledge_base(self, documents: List[Document] = None):
        \"\"\"Set up the knowledge base with documents.\"\"\"
        if documents is None:
            documents = self._create_sample_documents()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        # Setup QA chains
        self._setup_qa_chains()
        
        self.logger.info(f\"Knowledge base setup with {len(chunks)} chunks\")
    
    def _create_sample_documents(self) -> List[Document]:
        \"\"\"Create sample documents.\"\"\"
        documents = []
        for filename, content in SAMPLE_DOCUMENTS.items():
            doc = Document(
                page_content=content,
                metadata={
                    \"source\": filename,
                    \"created_at\": datetime.now().isoformat()
                }
            )
            documents.append(doc)
        return documents
    
    def _setup_qa_chains(self):
        \"\"\"Setup Q&A chains.\"\"\"
        qa_prompt = PromptTemplate(
            template=\"\"\"Use the context to answer the question. 
            If you don't know the answer, say so. Cite sources when possible.
            
            Context: {context}
            Question: {question}
            
            Answer with source citation:\"\"\",
            input_variables=[\"context\", \"question\"]
        )
        
        # Basic QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=\"stuff\",
            retriever=self.vectorstore.as_retriever(search_kwargs={\"k\": 3}),
            return_source_documents=True,
            chain_type_kwargs={\"prompt\": qa_prompt}
        )
        
        # Conversational QA chain
        self.conversational_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={\"k\": 3}),
            memory=self.memory,
            return_source_documents=True
        )
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        \"\"\"Ask a question and get answer with sources.\"\"\"
        if not self.qa_chain:
            raise ValueError(\"Knowledge base not set up\")
        
        try:
            result = self.qa_chain({\"query\": question})
            
            response = {
                \"question\": question,
                \"answer\": result[\"result\"],
                \"sources\": [
                    {
                        \"source\": doc.metadata.get(\"source\", \"Unknown\"),
                        \"content_preview\": doc.page_content[:150] + \"...\"
                    }
                    for doc in result.get(\"source_documents\", [])
                ],
                \"timestamp\": datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f\"Error: {e}\")
            return {
                \"question\": question,
                \"answer\": f\"Error: {e}\",
                \"error\": True
            }
    
    def ask_conversational(self, question: str) -> Dict[str, Any]:
        \"\"\"Ask question in conversational context.\"\"\"
        if not self.conversational_chain:
            raise ValueError(\"Knowledge base not set up\")
        
        try:
            result = self.conversational_chain({\"question\": question})
            
            return {
                \"question\": question,
                \"answer\": result[\"answer\"],
                \"sources\": [
                    doc.metadata.get(\"source\", \"Unknown\")
                    for doc in result.get(\"source_documents\", [])
                ]
            }
            
        except Exception as e:
            return {\"question\": question, \"answer\": f\"Error: {e}\", \"error\": True}
    
    def search_documents(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        \"\"\"Search for relevant documents.\"\"\"
        if not self.vectorstore:
            return []
        
        docs = self.vectorstore.similarity_search_with_score(query, k=k)
        
        return [{
            \"content\": doc.page_content[:200] + \"...\",
            \"source\": doc.metadata.get(\"source\", \"Unknown\"),
            \"similarity_score\": float(score)
        } for doc, score in docs]


def demonstrate_basic_qa():
    \"\"\"Demonstrate basic document Q&A.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found\")
        return
    
    print(\"\n\" + \"=\"*60)
    print(\"DOCUMENT Q&A SYSTEM\")
    print(\"=\"*60)
    
    try:
        qa_system = DocumentQASystem()
        qa_system.setup_knowledge_base()
        
        questions = [
            \"What is LangChain?\",
            \"What are AI ethics principles?\",
            \"What is supervised learning?\"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f\"\nQ{i}: {question}\")
            response = qa_system.ask_question(question)
            print(f\"A{i}: {response['answer'][:150]}...\")
            if response.get('sources'):
                sources = [s['source'] for s in response['sources']]
                print(f\"Sources: {sources}\")
    
    except Exception as e:
        logger.error(f\"Error: {e}\")
    
    print(\"\n💡 System retrieves context and provides sourced answers.\")
    print(\"=\"*60)


def demonstrate_conversational_qa():
    \"\"\"Demonstrate conversational Q&A.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found\")
        return
    
    print(\"\n\" + \"=\"*60)
    print(\"CONVERSATIONAL Q&A\")
    print(\"=\"*60)
    
    try:
        qa_system = DocumentQASystem()
        qa_system.setup_knowledge_base()
        
        conversation = [
            \"What is machine learning?\",
            \"What are the types you mentioned?\",
            \"Tell me about supervised learning.\"
        ]
        
        for i, question in enumerate(conversation, 1):
            print(f\"\n👤 Turn {i}: {question}\")
            response = qa_system.ask_conversational(question)
            print(f\"🤖 Answer: {response['answer'][:200]}...\")
    
    except Exception as e:
        logger.error(f\"Error: {e}\")
    
    print(\"\n💡 Conversational Q&A maintains context across turns.\")
    print(\"=\"*60)


def main():
    \"\"\"Main function.\"\"\"
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info(\"🚀 Starting Document Q&A Project\")
    
    try:
        demonstrate_basic_qa()
        demonstrate_conversational_qa()
        
        print(\"\n🎯 Document Q&A Key Takeaways:\")
        print(\"1. RAG combines retrieval with generation for accurate answers\")
        print(\"2. Vector embeddings enable semantic document search\")
        print(\"3. Source attribution provides transparency and trust\")
        print(\"4. Conversational memory enables natural Q&A flows\")
        print(\"5. Proper chunking strategies improve retrieval quality\")
        
        logger.info(\"✅ Document Q&A project completed!\")
        
    except Exception as e:
        logger.error(f\"❌ Error: {e}\")


if __name__ == \"__main__\":
    main()