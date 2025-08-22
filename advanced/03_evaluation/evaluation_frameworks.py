#!/usr/bin/env python3
"""
LangChain Evaluation Frameworks

This module demonstrates:
1. LLM output evaluation methods
2. Custom evaluation metrics
3. A/B testing for LLM applications
4. Performance benchmarking
5. Quality assessment frameworks
6. Automated evaluation pipelines

Key concepts:
- Response quality evaluation
- Factual accuracy assessment
- Relevance and coherence metrics
- Custom scoring functions
- Evaluation dataset management
"""

import sys
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
import statistics

# Add utils to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import setup_logging, get_logger, get_api_key

# LangChain imports
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
from langchain.evaluation import load_evaluator, EvaluatorType

# Optional dependencies for advanced evaluation
try:
    import pandas as pd
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    ADVANCED_EVAL_AVAILABLE = True
except ImportError:
    ADVANCED_EVAL_AVAILABLE = False
    print("⚠️ Advanced evaluation dependencies not installed")
    print("Install with: pip install pandas numpy scikit-learn sentence-transformers")


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    score: float
    explanation: str
    metric_name: str
    timestamp: str
    metadata: Dict[str, Any] = None


@dataclass
class EvaluationDataset:
    """Container for evaluation datasets."""
    questions: List[str]
    expected_answers: List[str]
    contexts: Optional[List[str]] = None
    metadata: Optional[List[Dict[str, Any]]] = None


class BaseEvaluator:
    """Base class for LLM evaluators."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"Evaluator-{name}")
    
    def evaluate(self, prediction: str, reference: str = None, context: str = None) -> EvaluationResult:
        """Evaluate a prediction against reference."""
        raise NotImplementedError
    
    def batch_evaluate(self, predictions: List[str], references: List[str] = None, 
                      contexts: List[str] = None) -> List[EvaluationResult]:
        """Evaluate multiple predictions."""
        results = []
        for i, pred in enumerate(predictions):
            ref = references[i] if references and i < len(references) else None
            ctx = contexts[i] if contexts and i < len(contexts) else None
            results.append(self.evaluate(pred, ref, ctx))
        return results


class RelevanceEvaluator(BaseEvaluator):
    """Evaluates response relevance using LLM."""
    
    def __init__(self, llm_model: str = "gpt-3.5-turbo"):
        super().__init__("Relevance")
        api_key = get_api_key('openai')
        if api_key:
            self.llm = ChatOpenAI(openai_api_key=api_key, model_name=llm_model, temperature=0)
        else:
            self.llm = None
            self.logger.warning("No API key available, using mock evaluation")
        
        self.prompt = PromptTemplate(
            input_variables=["question", "answer"],
            template="""
You are an expert evaluator. Rate the relevance of the answer to the question on a scale of 1-10.

Question: {question}
Answer: {answer}

Evaluation Criteria:
- Does the answer directly address the question?
- Is the information provided relevant and useful?
- Are there any significant irrelevant tangents?

Provide your rating as a number from 1-10, followed by a brief explanation.
Format: SCORE: X
EXPLANATION: Your explanation here
"""
        )
    
    def evaluate(self, prediction: str, reference: str = None, context: str = None) -> EvaluationResult:
        """Evaluate relevance of prediction to the question (context)."""
        if not context:
            return EvaluationResult(
                score=0.0,
                explanation="No question context provided",
                metric_name=self.name,
                timestamp=datetime.now().isoformat()
            )
        
        if not self.llm:
            # Mock evaluation for demo
            score = 7.5
            explanation = "Mock evaluation - API key required for real evaluation"
        else:
            try:
                chain = LLMChain(llm=self.llm, prompt=self.prompt)
                result = chain.run(question=context, answer=prediction)
                
                # Parse the result
                lines = result.strip().split('\n')
                score_line = next((line for line in lines if line.startswith('SCORE:')), None)
                explanation_line = next((line for line in lines if line.startswith('EXPLANATION:')), None)
                
                if score_line:
                    score = float(score_line.split(':')[1].strip())
                else:
                    score = 5.0  # Default if parsing fails
                
                explanation = explanation_line.split(':', 1)[1].strip() if explanation_line else result
                
            except Exception as e:
                self.logger.error(f"Evaluation failed: {e}")
                score = 0.0
                explanation = f"Evaluation error: {str(e)}"
        
        return EvaluationResult(
            score=score,
            explanation=explanation,
            metric_name=self.name,
            timestamp=datetime.now().isoformat()
        )


class FactualAccuracyEvaluator(BaseEvaluator):
    """Evaluates factual accuracy of responses."""
    
    def __init__(self, llm_model: str = "gpt-3.5-turbo"):
        super().__init__("Factual-Accuracy")
        api_key = get_api_key('openai')
        if api_key:
            self.llm = ChatOpenAI(openai_api_key=api_key, model_name=llm_model, temperature=0)
        else:
            self.llm = None
        
        self.prompt = PromptTemplate(
            input_variables=["answer", "reference"],
            template="""
You are a fact-checking expert. Compare the given answer against the reference information and rate its factual accuracy.

Answer to evaluate: {answer}
Reference information: {reference}

Evaluation Criteria:
- Are the facts in the answer correct according to the reference?
- Are there any factual errors or inaccuracies?
- Is the information consistent with the reference?

Rate on a scale of 1-10 where:
- 10: Completely accurate, no factual errors
- 7-9: Mostly accurate with minor issues
- 4-6: Some accurate information but notable errors
- 1-3: Significant factual errors
- 0: Completely inaccurate

Format: SCORE: X
EXPLANATION: Your explanation here
"""
        )
    
    def evaluate(self, prediction: str, reference: str = None, context: str = None) -> EvaluationResult:
        """Evaluate factual accuracy against reference."""
        if not reference:
            return EvaluationResult(
                score=0.0,
                explanation="No reference information provided",
                metric_name=self.name,
                timestamp=datetime.now().isoformat()
            )
        
        if not self.llm:
            score = 6.5
            explanation = "Mock evaluation - API key required for real evaluation"
        else:
            try:
                chain = LLMChain(llm=self.llm, prompt=self.prompt)
                result = chain.run(answer=prediction, reference=reference)
                
                # Parse result
                lines = result.strip().split('\n')
                score_line = next((line for line in lines if line.startswith('SCORE:')), None)
                explanation_line = next((line for line in lines if line.startswith('EXPLANATION:')), None)
                
                score = float(score_line.split(':')[1].strip()) if score_line else 5.0
                explanation = explanation_line.split(':', 1)[1].strip() if explanation_line else result
                
            except Exception as e:
                self.logger.error(f"Factual accuracy evaluation failed: {e}")
                score = 0.0
                explanation = f"Evaluation error: {str(e)}"
        
        return EvaluationResult(
            score=score,
            explanation=explanation,
            metric_name=self.name,
            timestamp=datetime.now().isoformat()
        )


class SemanticSimilarityEvaluator(BaseEvaluator):
    """Evaluates semantic similarity using embeddings."""
    
    def __init__(self):
        super().__init__("Semantic-Similarity")
        if ADVANCED_EVAL_AVAILABLE:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.available = True
            except Exception as e:
                self.logger.warning(f"Could not load embedding model: {e}")
                self.available = False
        else:
            self.available = False
    
    def evaluate(self, prediction: str, reference: str = None, context: str = None) -> EvaluationResult:
        """Evaluate semantic similarity between prediction and reference."""
        if not reference:
            return EvaluationResult(
                score=0.0,
                explanation="No reference provided for similarity comparison",
                metric_name=self.name,
                timestamp=datetime.now().isoformat()
            )
        
        if not self.available:
            score = 0.75
            explanation = "Mock similarity score - sentence-transformers required for real evaluation"
        else:
            try:
                # Encode texts
                pred_embedding = self.model.encode([prediction])
                ref_embedding = self.model.encode([reference])
                
                # Calculate cosine similarity
                similarity = cosine_similarity(pred_embedding, ref_embedding)[0][0]
                score = float(similarity) * 10  # Scale to 1-10
                
                explanation = f"Cosine similarity: {similarity:.3f}"
                
            except Exception as e:
                self.logger.error(f"Similarity evaluation failed: {e}")
                score = 0.0
                explanation = f"Evaluation error: {str(e)}"
        
        return EvaluationResult(
            score=score,
            explanation=explanation,
            metric_name=self.name,
            timestamp=datetime.now().isoformat()
        )


class CompletenessEvaluator(BaseEvaluator):
    """Evaluates response completeness."""
    
    def __init__(self, llm_model: str = "gpt-3.5-turbo"):
        super().__init__("Completeness")
        api_key = get_api_key('openai')
        if api_key:
            self.llm = ChatOpenAI(openai_api_key=api_key, model_name=llm_model, temperature=0)
        else:
            self.llm = None
        
        self.prompt = PromptTemplate(
            input_variables=["question", "answer"],
            template="""
Evaluate how completely the answer addresses all aspects of the question.

Question: {question}
Answer: {answer}

Evaluation Criteria:
- Does the answer address all parts of the question?
- Are there missing important details or aspects?
- Is the response thorough and comprehensive?

Rate on a scale of 1-10 where:
- 10: Fully complete, addresses all aspects thoroughly
- 7-9: Mostly complete with minor gaps
- 4-6: Partially complete, some important aspects missing
- 1-3: Incomplete, major gaps in addressing the question

Format: SCORE: X
EXPLANATION: Your explanation here
"""
        )
    
    def evaluate(self, prediction: str, reference: str = None, context: str = None) -> EvaluationResult:
        """Evaluate completeness of response."""
        if not context:
            return EvaluationResult(
                score=0.0,
                explanation="No question context provided",
                metric_name=self.name,
                timestamp=datetime.now().isoformat()
            )
        
        if not self.llm:
            score = 7.0
            explanation = "Mock evaluation - API key required for real evaluation"
        else:
            try:
                chain = LLMChain(llm=self.llm, prompt=self.prompt)
                result = chain.run(question=context, answer=prediction)
                
                # Parse result
                lines = result.strip().split('\n')
                score_line = next((line for line in lines if line.startswith('SCORE:')), None)
                explanation_line = next((line for line in lines if line.startswith('EXPLANATION:')), None)
                
                score = float(score_line.split(':')[1].strip()) if score_line else 5.0
                explanation = explanation_line.split(':', 1)[1].strip() if explanation_line else result
                
            except Exception as e:
                self.logger.error(f"Completeness evaluation failed: {e}")
                score = 0.0
                explanation = f"Evaluation error: {str(e)}"
        
        return EvaluationResult(
            score=score,
            explanation=explanation,
            metric_name=self.name,
            timestamp=datetime.now().isoformat()
        )


class EvaluationSuite:
    """Comprehensive evaluation suite for LLM applications."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.evaluators = {
            'relevance': RelevanceEvaluator(),
            'factual_accuracy': FactualAccuracyEvaluator(),
            'semantic_similarity': SemanticSimilarityEvaluator(),
            'completeness': CompletenessEvaluator()
        }
        self.results_history = []
    
    def evaluate_response(self, prediction: str, question: str = None, 
                         reference: str = None, evaluators: List[str] = None) -> Dict[str, EvaluationResult]:
        """Evaluate a single response using specified evaluators."""
        if evaluators is None:
            evaluators = list(self.evaluators.keys())
        
        results = {}
        for evaluator_name in evaluators:
            if evaluator_name in self.evaluators:
                evaluator = self.evaluators[evaluator_name]
                result = evaluator.evaluate(prediction, reference, question)
                results[evaluator_name] = result
            else:
                self.logger.warning(f"Unknown evaluator: {evaluator_name}")
        
        # Store in history
        self.results_history.append({
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction[:100] + "..." if len(prediction) > 100 else prediction,
            'results': results
        })
        
        return results
    
    def evaluate_dataset(self, dataset: EvaluationDataset, 
                        llm_function: Callable[[str], str]) -> Dict[str, List[EvaluationResult]]:
        """Evaluate a complete dataset."""
        self.logger.info(f"Evaluating dataset with {len(dataset.questions)} examples")
        
        all_results = {name: [] for name in self.evaluators.keys()}
        
        for i, question in enumerate(dataset.questions):
            self.logger.info(f"Evaluating example {i+1}/{len(dataset.questions)}")
            
            # Generate prediction
            try:
                prediction = llm_function(question)
            except Exception as e:
                self.logger.error(f"LLM function failed for question {i}: {e}")
                prediction = "Error: Could not generate response"
            
            # Get reference if available
            reference = dataset.expected_answers[i] if i < len(dataset.expected_answers) else None
            
            # Evaluate
            results = self.evaluate_response(prediction, question, reference)
            
            # Collect results
            for evaluator_name, result in results.items():
                all_results[evaluator_name].append(result)
        
        return all_results
    
    def get_summary_statistics(self, results: Dict[str, List[EvaluationResult]]) -> Dict[str, Dict[str, float]]:
        """Calculate summary statistics for evaluation results."""
        summary = {}
        
        for evaluator_name, eval_results in results.items():
            scores = [r.score for r in eval_results]
            if scores:
                summary[evaluator_name] = {
                    'mean': statistics.mean(scores),
                    'median': statistics.median(scores),
                    'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    'min': min(scores),
                    'max': max(scores),
                    'count': len(scores)
                }
            else:
                summary[evaluator_name] = {
                    'mean': 0.0, 'median': 0.0, 'std_dev': 0.0,
                    'min': 0.0, 'max': 0.0, 'count': 0
                }
        
        return summary
    
    def compare_models(self, model_functions: Dict[str, Callable[[str], str]], 
                      dataset: EvaluationDataset) -> Dict[str, Dict[str, List[EvaluationResult]]]:
        """Compare multiple models on the same dataset."""
        self.logger.info(f"Comparing {len(model_functions)} models")
        
        model_results = {}
        for model_name, model_function in model_functions.items():
            self.logger.info(f"Evaluating model: {model_name}")
            model_results[model_name] = self.evaluate_dataset(dataset, model_function)
        
        return model_results
    
    def export_results(self, results: Dict[str, List[EvaluationResult]], filepath: str):
        """Export evaluation results to JSON file."""
        export_data = {}
        for evaluator_name, eval_results in results.items():
            export_data[evaluator_name] = [
                {
                    'score': r.score,
                    'explanation': r.explanation,
                    'timestamp': r.timestamp,
                    'metadata': r.metadata
                }
                for r in eval_results
            ]
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Results exported to {filepath}")


class PerformanceBenchmark:
    """Performance benchmarking for LLM applications."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    def benchmark_latency(self, llm_function: Callable[[str], str], 
                         queries: List[str], runs: int = 3) -> Dict[str, float]:
        """Benchmark response latency."""
        self.logger.info(f"Benchmarking latency with {len(queries)} queries, {runs} runs each")
        
        all_times = []
        for run in range(runs):
            run_times = []
            for query in queries:
                start_time = time.time()
                try:
                    _ = llm_function(query)
                    end_time = time.time()
                    run_times.append(end_time - start_time)
                except Exception as e:
                    self.logger.error(f"Query failed: {e}")
                    run_times.append(float('inf'))
            
            all_times.extend(run_times)
        
        # Calculate statistics
        valid_times = [t for t in all_times if t != float('inf')]
        if not valid_times:
            return {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std_dev': 0}
        
        return {
            'mean': statistics.mean(valid_times),
            'median': statistics.median(valid_times),
            'min': min(valid_times),
            'max': max(valid_times),
            'std_dev': statistics.stdev(valid_times) if len(valid_times) > 1 else 0,
            'success_rate': len(valid_times) / len(all_times)
        }
    
    def benchmark_throughput(self, llm_function: Callable[[str], str], 
                           query: str, duration_seconds: int = 60) -> Dict[str, float]:
        """Benchmark throughput (requests per second)."""
        self.logger.info(f"Benchmarking throughput for {duration_seconds} seconds")
        
        start_time = time.time()
        request_count = 0
        errors = 0
        
        while time.time() - start_time < duration_seconds:
            try:
                _ = llm_function(query)
                request_count += 1
            except Exception as e:
                errors += 1
                self.logger.error(f"Request failed: {e}")
        
        actual_duration = time.time() - start_time
        
        return {
            'requests_per_second': request_count / actual_duration,
            'total_requests': request_count,
            'total_errors': errors,
            'error_rate': errors / (request_count + errors) if (request_count + errors) > 0 else 0,
            'actual_duration': actual_duration
        }


def create_sample_dataset() -> EvaluationDataset:
    """Create a sample evaluation dataset."""
    questions = [
        "What is the capital of France?",
        "Explain the concept of machine learning.",
        "What are the benefits of renewable energy?",
        "How does photosynthesis work?",
        "What is the difference between AI and ML?"
    ]
    
    expected_answers = [
        "The capital of France is Paris.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
        "Renewable energy benefits include reduced carbon emissions, energy independence, job creation, and long-term cost savings.",
        "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen using chlorophyll.",
        "AI (Artificial Intelligence) is the broader concept of machines being able to carry out tasks in a smart way, while ML (Machine Learning) is a subset of AI that focuses on algorithms that can learn from data."
    ]
    
    return EvaluationDataset(questions=questions, expected_answers=expected_answers)


def demo_llm_function(question: str) -> str:
    """Demo LLM function for testing (uses actual LLM if API key available)."""
    api_key = get_api_key('openai')
    if api_key:
        llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-3.5-turbo", temperature=0.7)
        return llm.predict(question)
    else:
        # Mock responses for demo
        mock_responses = {
            "What is the capital of France?": "Paris is the capital of France.",
            "Explain the concept of machine learning.": "Machine learning is a type of AI that allows computers to learn from data.",
            "What are the benefits of renewable energy?": "Renewable energy reduces pollution and provides sustainable power.",
            "How does photosynthesis work?": "Plants use sunlight to convert CO2 and water into glucose.",
            "What is the difference between AI and ML?": "AI is broader, ML is a subset focused on learning from data."
        }
        return mock_responses.get(question, "This is a demo response since no API key is available.")


def main():
    """Demonstrate LLM evaluation frameworks."""
    setup_logging()
    logger = get_logger("Evaluation-Demo")
    
    print("🧪 LangChain Evaluation Frameworks Demo")
    print("=" * 50)
    
    # Initialize evaluation suite
    eval_suite = EvaluationSuite()
    
    # Create sample dataset
    dataset = create_sample_dataset()
    print(f"📊 Created dataset with {len(dataset.questions)} questions")
    
    # Single response evaluation
    print("\n🔍 Single Response Evaluation:")
    question = "What is machine learning?"
    response = demo_llm_function(question)
    reference = "Machine learning is a method of data analysis that automates analytical model building."
    
    results = eval_suite.evaluate_response(response, question, reference)
    
    for evaluator_name, result in results.items():
        print(f"  {evaluator_name}: {result.score:.2f}/10 - {result.explanation[:100]}...")
    
    # Dataset evaluation
    print(f"\n📈 Dataset Evaluation:")
    dataset_results = eval_suite.evaluate_dataset(dataset, demo_llm_function)
    
    # Summary statistics
    summary = eval_suite.get_summary_statistics(dataset_results)
    print("\n📊 Summary Statistics:")
    for evaluator_name, stats in summary.items():
        print(f"  {evaluator_name}:")
        print(f"    Mean: {stats['mean']:.2f}")
        print(f"    Median: {stats['median']:.2f}")
        print(f"    Std Dev: {stats['std_dev']:.2f}")
        print(f"    Range: {stats['min']:.2f} - {stats['max']:.2f}")
    
    # Performance benchmarking
    print(f"\n⚡ Performance Benchmarking:")
    benchmark = PerformanceBenchmark()
    
    # Latency benchmark
    latency_results = benchmark.benchmark_latency(demo_llm_function, dataset.questions[:3], runs=2)
    print(f"  Latency (avg): {latency_results['mean']:.3f}s")
    print(f"  Success rate: {latency_results['success_rate']:.2%}")
    
    # Export results
    output_dir = Path(__file__).parent
    results_file = output_dir / "evaluation_results.json"
    eval_suite.export_results(dataset_results, str(results_file))
    print(f"\n💾 Results exported to: {results_file}")
    
    print("\n✅ Evaluation framework demo completed!")
    print("\nKey takeaways:")
    print("1. Multiple evaluation metrics provide comprehensive assessment")
    print("2. Automated evaluation enables systematic testing")
    print("3. Performance benchmarking helps optimize applications")
    print("4. Custom evaluators can be created for specific domains")


if __name__ == "__main__":
    main()