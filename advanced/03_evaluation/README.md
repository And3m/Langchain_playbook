# LLM Evaluation Frameworks 🧪

Advanced evaluation techniques for LangChain applications, including automated assessment, performance benchmarking, and quality metrics.

## 📋 Overview

This module provides comprehensive evaluation tools for:
- **Response Quality Assessment**: Relevance, accuracy, completeness
- **Performance Benchmarking**: Latency, throughput, reliability
- **Comparative Analysis**: A/B testing and model comparison
- **Custom Metrics**: Domain-specific evaluation criteria

## 🎯 Key Concepts

### Evaluation Dimensions

1. **Relevance**: How well the response addresses the question
2. **Factual Accuracy**: Correctness against reference information
3. **Completeness**: Coverage of all question aspects
4. **Semantic Similarity**: Meaning alignment with expected answers

### Evaluation Types

- **Human-in-the-loop**: LLM-assisted evaluation
- **Automated Metrics**: Rule-based and ML-based scoring
- **Comparative**: Side-by-side model comparison
- **Performance**: Speed and reliability metrics

## 🚀 Quick Start

### Basic Response Evaluation

```python
from evaluation_frameworks import EvaluationSuite

# Initialize evaluation suite
eval_suite = EvaluationSuite()

# Evaluate a single response
question = "What is machine learning?"
response = "ML is a subset of AI that learns from data"
reference = "Machine learning enables computers to learn from data"

results = eval_suite.evaluate_response(
    prediction=response,
    question=question,
    reference=reference
)

# View results
for metric, result in results.items():
    print(f"{metric}: {result.score:.2f}/10")
```

### Dataset Evaluation

```python
from evaluation_frameworks import EvaluationDataset, demo_llm_function

# Create evaluation dataset
dataset = EvaluationDataset(
    questions=["What is AI?", "How does ML work?"],
    expected_answers=["AI description", "ML explanation"]
)

# Evaluate your LLM function
results = eval_suite.evaluate_dataset(dataset, your_llm_function)

# Get summary statistics
summary = eval_suite.get_summary_statistics(results)
print(f"Average relevance: {summary['relevance']['mean']:.2f}")
```

## 📊 Evaluation Metrics

### 1. Relevance Evaluator
**Purpose**: Assess how well responses address the question
**Method**: LLM-based evaluation with structured prompts
**Scale**: 1-10 (10 = perfectly relevant)

```python
relevance_eval = RelevanceEvaluator()
result = relevance_eval.evaluate(response, context=question)
```

### 2. Factual Accuracy Evaluator
**Purpose**: Verify correctness against reference information
**Method**: LLM comparison with fact-checking prompts
**Scale**: 1-10 (10 = completely accurate)

```python
accuracy_eval = FactualAccuracyEvaluator()
result = accuracy_eval.evaluate(response, reference=ground_truth)
```

### 3. Semantic Similarity Evaluator
**Purpose**: Measure meaning similarity using embeddings
**Method**: Cosine similarity of sentence embeddings
**Scale**: 0-10 (10 = identical meaning)

```python
similarity_eval = SemanticSimilarityEvaluator()
result = similarity_eval.evaluate(response, reference=expected)
```

### 4. Completeness Evaluator
**Purpose**: Assess whether all question aspects are addressed
**Method**: LLM evaluation of response thoroughness
**Scale**: 1-10 (10 = fully complete)

```python
completeness_eval = CompletenessEvaluator()
result = completeness_eval.evaluate(response, context=question)
```

## ⚡ Performance Benchmarking

### Latency Benchmarking

```python
from evaluation_frameworks import PerformanceBenchmark

benchmark = PerformanceBenchmark()

# Test response latency
queries = ["Question 1", "Question 2", "Question 3"]
latency_results = benchmark.benchmark_latency(
    llm_function=your_llm_function,
    queries=queries,
    runs=3
)

print(f"Average latency: {latency_results['mean']:.3f}s")
print(f"Success rate: {latency_results['success_rate']:.2%}")
```

### Throughput Benchmarking

```python
# Test requests per second
throughput_results = benchmark.benchmark_throughput(
    llm_function=your_llm_function,
    query="Test query",
    duration_seconds=60
)

print(f"Throughput: {throughput_results['requests_per_second']:.2f} req/s")
```

## 🔄 Model Comparison

### A/B Testing

```python
# Define model functions
def model_a(query):
    # Your model A implementation
    return "Response from model A"

def model_b(query):
    # Your model B implementation
    return "Response from model B"

# Compare models
model_functions = {
    'Model A': model_a,
    'Model B': model_b
}

comparison_results = eval_suite.compare_models(model_functions, dataset)

# Analyze results
for model_name, results in comparison_results.items():
    summary = eval_suite.get_summary_statistics(results)
    print(f"{model_name} average relevance: {summary['relevance']['mean']:.2f}")
```

## 🎛️ Custom Evaluators

### Creating Custom Metrics

```python
from evaluation_frameworks import BaseEvaluator, EvaluationResult

class CustomEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__("Custom-Metric")
    
    def evaluate(self, prediction, reference=None, context=None):
        # Your custom evaluation logic
        score = your_scoring_function(prediction)
        
        return EvaluationResult(
            score=score,
            explanation="Custom evaluation explanation",
            metric_name=self.name,
            timestamp=datetime.now().isoformat()
        )

# Use custom evaluator
custom_eval = CustomEvaluator()
eval_suite.evaluators['custom'] = custom_eval
```

## 📈 Advanced Features

### Evaluation Pipelines

```python
# Automated evaluation pipeline
def create_evaluation_pipeline(dataset, models, evaluators):
    results = {}
    
    for model_name, model_func in models.items():
        print(f"Evaluating {model_name}...")
        
        model_results = eval_suite.evaluate_dataset(dataset, model_func)
        results[model_name] = eval_suite.get_summary_statistics(model_results)
    
    return results

# Run pipeline
pipeline_results = create_evaluation_pipeline(
    dataset=your_dataset,
    models={'GPT-3.5': gpt35_func, 'GPT-4': gpt4_func},
    evaluators=['relevance', 'accuracy', 'completeness']
)
```

### Result Export and Analysis

```python
# Export detailed results
eval_suite.export_results(results, "evaluation_results.json")

# Statistical analysis
import pandas as pd

# Convert to DataFrame for analysis
def results_to_dataframe(results):
    data = []
    for metric, evals in results.items():
        for eval_result in evals:
            data.append({
                'metric': metric,
                'score': eval_result.score,
                'timestamp': eval_result.timestamp
            })
    return pd.DataFrame(data)

df = results_to_dataframe(results)
print(df.groupby('metric')['score'].describe())
```

## 📚 Best Practices

### 1. Evaluation Design
- Use multiple complementary metrics
- Include both automated and human evaluation
- Test on diverse, representative datasets
- Regular evaluation during development

### 2. Performance Optimization
- Cache evaluation results when possible
- Use batch evaluation for efficiency
- Monitor evaluation costs (LLM-based evaluators)
- Implement timeouts for reliability

### 3. Metric Selection
- Choose metrics aligned with use case goals
- Balance speed vs. accuracy in metric selection
- Consider domain-specific evaluation criteria
- Validate metrics against human judgment

### 4. Continuous Monitoring
- Implement evaluation in CI/CD pipelines
- Track metrics over time
- Set up alerting for quality degradation
- Regular model comparison and updates

## 🔧 Configuration

### Environment Setup

```bash
# Install dependencies
pip install sentence-transformers  # For semantic similarity
pip install pandas numpy scikit-learn  # For advanced analysis

# Set API keys for LLM-based evaluation
export OPENAI_API_KEY=your_key_here
```

### Evaluation Configuration

```python
# Configure evaluators
evaluators = {
    'relevance': RelevanceEvaluator(llm_model="gpt-4"),
    'accuracy': FactualAccuracyEvaluator(llm_model="gpt-3.5-turbo"),
    'similarity': SemanticSimilarityEvaluator(),
    'completeness': CompletenessEvaluator()
}

# Custom evaluation suite
custom_suite = EvaluationSuite()
custom_suite.evaluators = evaluators
```

## 📊 Evaluation Scenarios

### Chatbot Evaluation
```python
# Conversation quality assessment
conversation_eval = {
    'helpfulness': RelevanceEvaluator(),
    'accuracy': FactualAccuracyEvaluator(),
    'coherence': SemanticSimilarityEvaluator()
}
```

### RAG System Evaluation
```python
# Retrieval-augmented generation assessment
rag_eval = {
    'relevance': RelevanceEvaluator(),
    'faithfulness': FactualAccuracyEvaluator(),
    'completeness': CompletenessEvaluator()
}
```

### Code Generation Evaluation
```python
# Code quality assessment
code_eval = {
    'correctness': CustomCodeEvaluator(),
    'efficiency': CodePerformanceEvaluator(),
    'readability': CodeStyleEvaluator()
}
```

## 🚀 Usage Examples

See `evaluation_frameworks.py` for complete implementation and demo:

```bash
cd advanced/03_evaluation
python evaluation_frameworks.py
```

This will run a comprehensive evaluation demo showing:
- Single response evaluation
- Dataset evaluation with statistics
- Performance benchmarking
- Result export and analysis

## 🔗 Related Resources

- [LangChain Evaluation Guide](https://docs.langchain.com/docs/guides/evaluation/)
- [OpenAI Evals Framework](https://github.com/openai/evals)
- [Evaluation Metrics Papers](https://arxiv.org/list/cs.CL/recent)

---

**Build Robust LLM Applications with Comprehensive Evaluation! 📊**