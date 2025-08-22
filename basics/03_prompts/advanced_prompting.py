#!/usr/bin/env python3
\"\"\"
Advanced Prompt Engineering Techniques

This example demonstrates advanced prompting strategies:
1. Chain-of-thought prompting
2. Self-consistency prompting  
3. Prompt chaining
4. Dynamic prompt modification
5. Prompt optimization techniques
6. Error handling and fallback prompts

Key concepts:
- Reasoning and step-by-step thinking
- Multiple prompt attempts for consistency
- Breaking complex tasks into prompt chains
- Adaptive prompting based on context
\"\"\"

import sys
from pathlib import Path
from typing import List, Dict, Any
import random

# Add utils to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import setup_logging, get_logger, get_api_key, timing_decorator
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


def demonstrate_chain_of_thought():
    \"\"\"Demonstrate chain-of-thought prompting for reasoning.\"\"\"
    logger = get_logger(__name__)
    logger.info(\"🧠 Chain-of-Thought Prompting\")
    
    # Standard prompt vs Chain-of-thought prompt
    standard_template = PromptTemplate(
        input_variables=[\"problem\"],
        template=\"Solve this problem: {problem}\"
    )
    
    cot_template = PromptTemplate(
        input_variables=[\"problem\"],
        template=\"\"\"Solve this problem step by step. Think through each step carefully.
        
        Problem: {problem}
        
        Let me work through this step by step:
        Step 1:\"\"\"
    )
    
    # Test problem
    problem = \"A store sells apples at $2 per pound. If I buy 3.5 pounds and pay with a $10 bill, how much change will I receive?\"
    
    print(\"\n\" + \"=\"*70)
    print(\"CHAIN-OF-THOUGHT PROMPTING\")
    print(\"=\"*70)
    print(f\"Problem: {problem}\n\")
    
    # Show both approaches
    standard_prompt = standard_template.format(problem=problem)
    cot_prompt = cot_template.format(problem=problem)
    
    print(\"Standard Prompt:\")
    print(standard_prompt)
    print(\"\n\" + \"-\"*50 + \"\n\")
    
    print(\"Chain-of-Thought Prompt:\")
    print(cot_prompt)
    print(\"=\"*70)
    
    # Test with LLM if available
    api_key = get_api_key('openai')
    if api_key:
        try:
            llm = OpenAI(openai_api_key=api_key, temperature=0.3, max_tokens=200)
            
            print(\"\n🤖 LLM Responses:\")
            print(\"-\"*50)
            
            standard_response = llm(standard_prompt)
            print(\"Standard Response:\")
            print(standard_response.strip())
            
            print(\"\n\" + \"-\"*30 + \"\n\")
            
            cot_response = llm(cot_prompt)
            print(\"Chain-of-Thought Response:\")
            print(cot_response.strip())
            
        except Exception as e:
            logger.error(f\"LLM error: {e}\")


def demonstrate_self_consistency():
    \"\"\"Demonstrate self-consistency prompting.\"\"\"
    logger = get_logger(__name__)
    logger.info(\"🔄 Self-Consistency Prompting\")
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping demonstration\")
        return
    
    # Template for reasoning problem
    template = PromptTemplate(
        input_variables=[\"question\"],
        template=\"\"\"Answer this question by thinking step by step:
        
        {question}
        
        Let me think through this carefully:\"\"\"
    )
    
    question = \"If a train travels 120 miles in 2 hours, and then 180 miles in 3 hours, what is its average speed for the entire journey?\"
    
    print(\"\n\" + \"=\"*70)
    print(\"SELF-CONSISTENCY PROMPTING\")
    print(\"=\"*70)
    print(f\"Question: {question}\n\")
    
    try:
        llm = OpenAI(openai_api_key=api_key, temperature=0.7, max_tokens=150)
        prompt = template.format(question=question)
        
        # Generate multiple responses
        num_attempts = 3
        responses = []
        
        print(\"🔄 Generating multiple reasoning attempts:\")
        print(\"-\"*50)
        
        for i in range(num_attempts):
            response = llm(prompt)
            responses.append(response.strip())
            print(f\"\nAttempt {i+1}:\")
            print(response.strip())
            print(\"-\"*30)
        
        print(\"\n💡 Self-consistency helps identify the most reliable answer\")
        print(\"by comparing multiple reasoning paths for the same problem.\")
        
    except Exception as e:
        logger.error(f\"Self-consistency demo error: {e}\")


def demonstrate_prompt_chaining():
    \"\"\"Demonstrate breaking complex tasks into prompt chains.\"\"\"
    logger = get_logger(__name__)
    logger.info(\"⛓️ Prompt Chaining\")
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping demonstration\")
        return
    
    # Chain of prompts for content creation
    prompts = {
        \"brainstorm\": PromptTemplate(
            input_variables=[\"topic\"],
            template=\"Brainstorm 5 key points about {topic}. List them briefly:\"
        ),
        \"expand\": PromptTemplate(
            input_variables=[\"points\"],
            template=\"Take these key points and expand each into a detailed paragraph:\n{points}\"
        ),
        \"structure\": PromptTemplate(
            input_variables=[\"content\"],
            template=\"Organize this content into a well-structured article with introduction, main sections, and conclusion:\n{content}\"
        )
    }
    
    print(\"\n\" + \"=\"*70)
    print(\"PROMPT CHAINING\")
    print(\"=\"*70)
    
    topic = \"the benefits of renewable energy\"
    print(f\"Topic: {topic}\n\")
    
    try:
        llm = OpenAI(openai_api_key=api_key, temperature=0.7, max_tokens=200)
        
        # Step 1: Brainstorm
        print(\"🧠 Step 1: Brainstorming key points...\")
        brainstorm_prompt = prompts[\"brainstorm\"].format(topic=topic)
        key_points = llm(brainstorm_prompt)
        print(f\"Key Points:\n{key_points.strip()}\")
        print(\"-\"*50)
        
        # Step 2: Expand (using output from step 1)
        print(\"\n📝 Step 2: Expanding points...\")
        expand_prompt = prompts[\"expand\"].format(points=key_points.strip())
        expanded_content = llm(expand_prompt)
        print(f\"Expanded Content:\n{expanded_content.strip()}\")
        print(\"-\"*50)
        
        # Step 3: Structure (using output from step 2) 
        print(\"\n🏗️ Step 3: Structuring article...\")
        structure_prompt = prompts[\"structure\"].format(content=expanded_content.strip())
        final_article = llm(structure_prompt)
        print(f\"Final Article Structure:\n{final_article.strip()}\")
        
        print(\"\n💡 Prompt chaining breaks complex tasks into manageable steps,\")
        print(\"with each step building on the previous one's output.\")
        
    except Exception as e:
        logger.error(f\"Prompt chaining error: {e}\")


def demonstrate_dynamic_prompts():
    \"\"\"Demonstrate dynamic prompt modification based on context.\"\"\"
    logger = get_logger(__name__)
    logger.info(\"🎯 Dynamic Prompt Modification\")
    
    def create_adaptive_prompt(task_type: str, difficulty: str, context: str) -> str:
        \"\"\"Create a prompt adapted to the specific context.\"\"\"
        
        # Base templates for different task types
        base_templates = {
            \"explanation\": \"Explain {topic} in a {style} way.\",
            \"analysis\": \"Analyze {topic} focusing on {aspect}.\",
            \"creative\": \"Write a {format} about {topic} with {tone} tone.\"
        }
        
        # Difficulty modifiers
        difficulty_modifiers = {
            \"beginner\": \"Use simple language and basic concepts.\",
            \"intermediate\": \"Include some technical details and examples.\", 
            \"advanced\": \"Provide in-depth analysis with technical terminology.\"
        }
        
        # Context adaptations
        context_adaptations = {
            \"academic\": \"Use formal language and cite principles.\",
            \"business\": \"Focus on practical applications and ROI.\",
            \"casual\": \"Use conversational tone and relatable examples.\"
        }
        
        # Build adaptive prompt
        base = base_templates.get(task_type, \"Discuss {topic}.\")
        difficulty_mod = difficulty_modifiers.get(difficulty, \"\")
        context_mod = context_adaptations.get(context, \"\")
        
        return f\"{base} {difficulty_mod} {context_mod}\".strip()
    
    print(\"\n\" + \"=\"*70)
    print(\"DYNAMIC PROMPT MODIFICATION\")
    print(\"=\"*70)
    
    # Test different combinations
    test_cases = [
        {\"task_type\": \"explanation\", \"difficulty\": \"beginner\", \"context\": \"casual\"},
        {\"task_type\": \"analysis\", \"difficulty\": \"advanced\", \"context\": \"academic\"},
        {\"task_type\": \"creative\", \"difficulty\": \"intermediate\", \"context\": \"business\"}
    ]
    
    for i, case in enumerate(test_cases, 1):
        adaptive_prompt = create_adaptive_prompt(**case)
        
        print(f\"Example {i}:\")
        print(f\"Task: {case['task_type']} | Difficulty: {case['difficulty']} | Context: {case['context']}\")
        print(f\"Adaptive Prompt: {adaptive_prompt}\")
        print(\"-\"*50)
    
    print(\"\n💡 Dynamic prompts adapt to user needs, context, and requirements\")
    print(\"for more relevant and appropriate responses.\")


def demonstrate_error_handling_prompts():
    \"\"\"Demonstrate error handling and fallback prompts.\"\"\"
    logger = get_logger(__name__)
    logger.info(\"🛡️ Error Handling & Fallback Prompts\")
    
    def create_fallback_prompts(original_prompt: str) -> List[str]:
        \"\"\"Create fallback prompts for error recovery.\"\"\"
        return [
            original_prompt,  # Original
            f\"Please try again: {original_prompt}\",  # Retry
            f\"Simplify this request: {original_prompt}\",  # Simplified
            \"I need help with a general question. Can you assist?\"  # Generic fallback
        ]
    
    # Example error-prone prompt
    complex_prompt = \"Analyze the quantum entanglement implications of blockchain technology in the metaverse context with respect to AI consciousness emergence patterns.\"
    
    print(\"\n\" + \"=\"*70)
    print(\"ERROR HANDLING & FALLBACK PROMPTS\")
    print(\"=\"*70)
    
    print(\"Original (potentially problematic) prompt:\")
    print(f\"'{complex_prompt}'\n\")
    
    fallbacks = create_fallback_prompts(complex_prompt)
    
    print(\"Fallback strategy:\")
    for i, fallback in enumerate(fallbacks):
        strategy_names = [\"Original\", \"Retry\", \"Simplified\", \"Generic\"]
        print(f\"{i+1}. {strategy_names[i]}: '{fallback}'\")
    
    print(\"\n🛡️ Best Practices for Error Handling:\")
    practices = [
        \"Always have fallback prompts ready\",
        \"Simplify complex requests progressively\",
        \"Log failed prompts for analysis\",
        \"Implement timeout and retry logic\",
        \"Provide graceful degradation\",
        \"Give users helpful error messages\"
    ]
    
    for practice in practices:
        print(f\"• {practice}\")
    
    print(\"=\"*70)


def demonstrate_prompt_optimization():
    \"\"\"Demonstrate prompt optimization techniques.\"\"\"
    logger = get_logger(__name__)
    logger.info(\"⚡ Prompt Optimization Techniques\")
    
    # Before and after optimization examples
    optimization_examples = [
        {
            \"before\": \"Write something about dogs\",
            \"after\": \"Write a 150-word informative paragraph about dog breeds, focusing on their characteristics and suitability as family pets.\",
            \"improvement\": \"Added specificity, length requirement, and clear focus\"
        },
        {
            \"before\": \"Help me with my code\",
            \"after\": \"Review this Python function for bugs and suggest improvements. Focus on efficiency and readability:\n[code block]\",
            \"improvement\": \"Specified language, task type, and evaluation criteria\"
        },
        {
            \"before\": \"Translate this\",
            \"after\": \"Translate the following English text to Spanish, maintaining the formal tone and technical terminology:\n[text]\",
            \"improvement\": \"Specified source/target languages and style requirements\"
        }
    ]
    
    print(\"\n\" + \"=\"*70)
    print(\"PROMPT OPTIMIZATION TECHNIQUES\")
    print(\"=\"*70)
    
    for i, example in enumerate(optimization_examples, 1):
        print(f\"Example {i}:\")
        print(f\"❌ Before: {example['before']}\")
        print(f\"✅ After: {example['after']}\")
        print(f\"💡 Improvement: {example['improvement']}\")
        print(\"-\"*50)
    
    print(\"\n⚡ Optimization Principles:\")
    principles = [
        \"Be specific about requirements\",
        \"Include examples when helpful\",
        \"Specify output format and length\",
        \"Set clear context and constraints\",
        \"Use action verbs and clear instructions\",
        \"Test with edge cases\",
        \"Iterate based on results\",
        \"Measure and compare performance\"
    ]
    
    for principle in principles:
        print(f\"• {principle}\")
    
    print(\"=\"*70)


def main():
    \"\"\"Main function demonstrating advanced prompt engineering.\"\"\"
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info(\"🚀 Starting Advanced Prompt Engineering\")
    
    try:
        # Run all demonstrations
        demonstrate_chain_of_thought()
        demonstrate_self_consistency()
        demonstrate_prompt_chaining()
        demonstrate_dynamic_prompts()
        demonstrate_error_handling_prompts()
        demonstrate_prompt_optimization()
        
        print(\"\n🎯 Advanced Prompting Key Takeaways:\")
        print(\"1. Chain-of-thought improves reasoning tasks\")
        print(\"2. Self-consistency validates answers through multiple attempts\")
        print(\"3. Prompt chaining breaks complex tasks into steps\")
        print(\"4. Dynamic prompts adapt to context and requirements\")
        print(\"5. Error handling ensures robust applications\")
        print(\"6. Optimization improves accuracy and efficiency\")
        
        logger.info(\"✅ Advanced Prompt Engineering completed successfully!\")
        
    except Exception as e:
        logger.error(f\"❌ Error occurred: {e}\")
        logger.info(\"💡 Check your setup and try individual demonstrations\")


if __name__ == \"__main__\":
    main()