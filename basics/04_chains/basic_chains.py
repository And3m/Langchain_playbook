#!/usr/bin/env python3
\"\"\"
LangChain Chains - Building Workflow Pipelines

This example demonstrates:
1. Simple LLM chains
2. Sequential chains
3. Router chains
4. Transform chains
5. Conditional chains
6. Chain composition patterns

Key concepts:
- Connecting prompts to LLMs
- Chaining multiple operations
- Data flow between chain components
- Reusable chain patterns
\"\"\"

import sys
from pathlib import Path
from typing import Dict, Any, List

# Add utils to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import setup_logging, get_logger, get_api_key, timing_decorator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.schema import BaseOutputParser


class ListOutputParser(BaseOutputParser):
    \"\"\"Custom parser to extract lists from LLM output.\"\"\"
    
    def parse(self, text: str) -> List[str]:
        \"\"\"Parse text into a list of items.\"\"\"
        lines = text.strip().split('\n')
        items = []
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
                # Remove bullet points and numbers
                cleaned = line.lstrip('-•0123456789. ').strip()
                if cleaned:
                    items.append(cleaned)
        return items[:5]  # Limit to 5 items


@timing_decorator
def demonstrate_simple_llm_chain():
    \"\"\"Demonstrate basic LLM chain usage.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping demonstration\")
        return
    
    logger.info(\"🔗 Simple LLM Chain\")
    
    # Create LLM
    llm = OpenAI(openai_api_key=api_key, temperature=0.7, max_tokens=150)
    
    # Create prompt template
    prompt = PromptTemplate(
        input_variables=[\"product\"],
        template=\"\"\"You are a marketing expert. Create 3 compelling selling points for this product:
        
        Product: {product}
        
        Selling points:
        1.\"\"\"
    )
    
    # Create chain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    print(\"\n\" + \"=\"*60)
    print(\"SIMPLE LLM CHAIN\")
    print(\"=\"*60)
    
    # Test the chain
    test_products = [
        \"Wireless noise-canceling headphones\",
        \"Smart home security camera\",
        \"Eco-friendly water bottle\"
    ]
    
    for product in test_products:
        try:
            result = chain.run(product=product)
            print(f\"\n🛍️ Product: {product}\")
            print(f\"Selling Points:\n{result.strip()}\")
            print(\"-\" * 40)
        except Exception as e:
            logger.error(f\"Error processing {product}: {e}\")
    
    print(\"=\"*60)


@timing_decorator 
def demonstrate_sequential_chain():
    \"\"\"Demonstrate sequential chains with multiple steps.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping demonstration\")
        return
    
    logger.info(\"⛓️ Sequential Chain\")
    
    # Create LLM
    llm = OpenAI(openai_api_key=api_key, temperature=0.7, max_tokens=150)
    
    # Step 1: Brainstorm ideas
    brainstorm_prompt = PromptTemplate(
        input_variables=[\"topic\"],
        template=\"Brainstorm 5 creative ideas for: {topic}\n\nIdeas:\n1.\"
    )
    brainstorm_chain = LLMChain(llm=llm, prompt=brainstorm_prompt, output_key=\"ideas\")
    
    # Step 2: Evaluate ideas
    evaluate_prompt = PromptTemplate(
        input_variables=[\"ideas\"],
        template=\"\"\"Evaluate these ideas and pick the best one. Explain why it's the best:
        
        Ideas: {ideas}
        
        Best idea and reasoning:\"\"\"
    )
    evaluate_chain = LLMChain(llm=llm, prompt=evaluate_prompt, output_key=\"best_idea\")
    
    # Step 3: Create action plan
    plan_prompt = PromptTemplate(
        input_variables=[\"best_idea\"],
        template=\"\"\"Create a 3-step action plan to implement this idea:
        
        Idea: {best_idea}
        
        Action Plan:
        Step 1:\"\"\"
    )
    plan_chain = LLMChain(llm=llm, prompt=plan_prompt, output_key=\"action_plan\")
    
    # Combine into sequential chain
    overall_chain = SequentialChain(
        chains=[brainstorm_chain, evaluate_chain, plan_chain],
        input_variables=[\"topic\"],
        output_variables=[\"ideas\", \"best_idea\", \"action_plan\"],
        verbose=True
    )
    
    print(\"\n\" + \"=\"*60)
    print(\"SEQUENTIAL CHAIN\")
    print(\"=\"*60)
    
    topic = \"improving team productivity in remote work\"
    print(f\"Topic: {topic}\n\")
    
    try:
        result = overall_chain({\"topic\": topic})
        
        print(\"📋 Chain Results:\")
        print(f\"\n💡 Ideas Generated:\n{result['ideas']}\")
        print(f\"\n🎯 Best Idea:\n{result['best_idea']}\")
        print(f\"\n📝 Action Plan:\n{result['action_plan']}\")
        
    except Exception as e:
        logger.error(f\"Sequential chain error: {e}\")
    
    print(\"=\"*60)


@timing_decorator
def demonstrate_simple_sequential_chain():
    \"\"\"Demonstrate simple sequential chain (output to input).\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping demonstration\")
        return
    
    logger.info(\"🔄 Simple Sequential Chain\")
    
    # Create LLM
    llm = OpenAI(openai_api_key=api_key, temperature=0.7, max_tokens=100)
    
    # Chain 1: Generate a story idea
    story_prompt = PromptTemplate(
        input_variables=[\"theme\"],
        template=\"Create a brief story concept based on this theme: {theme}\n\nStory concept:\"
    )
    story_chain = LLMChain(llm=llm, prompt=story_prompt)
    
    # Chain 2: Create a title for the story
    title_prompt = PromptTemplate(
        input_variables=[\"story\"],
        template=\"Create a catchy title for this story concept:\n{story}\n\nTitle:\"
    )
    title_chain = LLMChain(llm=llm, prompt=title_prompt)
    
    # Combine chains (output of first becomes input of second)
    simple_chain = SimpleSequentialChain(
        chains=[story_chain, title_chain],
        verbose=True
    )
    
    print(\"\n\" + \"=\"*60)
    print(\"SIMPLE SEQUENTIAL CHAIN\")
    print(\"=\"*60)
    
    themes = [\"artificial intelligence and friendship\", \"time travel mystery\"]
    
    for theme in themes:
        try:
            result = simple_chain.run(theme)
            print(f\"\n🎭 Theme: {theme}\")
            print(f\"📖 Final Title: {result.strip()}\")
            print(\"-\" * 40)
        except Exception as e:
            logger.error(f\"Error with theme '{theme}': {e}\")
    
    print(\"=\"*60)


def demonstrate_transform_chain():
    \"\"\"Demonstrate data transformation in chains.\"\"\"
    logger = get_logger(__name__)
    logger.info(\"🔄 Transform Chain\")
    
    # Custom transformation function
    def transform_input(inputs: dict) -> dict:
        \"\"\"Transform input data before sending to LLM.\"\"\"
        text = inputs[\"text\"]
        
        # Clean and prepare text
        cleaned_text = text.strip().lower()
        word_count = len(cleaned_text.split())
        
        # Determine appropriate prompt based on text length
        if word_count < 10:
            prompt_type = \"brief\"
        elif word_count < 50:
            prompt_type = \"moderate\"
        else:
            prompt_type = \"detailed\"
        
        return {
            \"original_text\": text,
            \"cleaned_text\": cleaned_text,
            \"word_count\": word_count,
            \"prompt_type\": prompt_type
        }
    
    # Simulate transform chain
    test_inputs = [
        \"AI is amazing\",
        \"Machine learning algorithms can process vast amounts of data to identify patterns.\",
        \"\"\"Artificial intelligence represents one of the most significant technological 
        advancements of our time, with applications spanning from healthcare and finance 
        to transportation and entertainment, fundamentally changing how we interact with 
        technology and each other.\"\"\"
    ]
    
    print(\"\n\" + \"=\"*60)
    print(\"TRANSFORM CHAIN SIMULATION\")
    print(\"=\"*60)
    
    for i, text in enumerate(test_inputs, 1):
        result = transform_input({\"text\": text})
        print(f\"\nExample {i}:\")
        print(f\"Original: {result['original_text'][:50]}{'...' if len(result['original_text']) > 50 else ''}\")
        print(f\"Word Count: {result['word_count']}\")
        print(f\"Prompt Type: {result['prompt_type']}\")
        print(\"-\" * 40)
    
    print(\"\n💡 Transform chains preprocess data before LLM processing,\")
    print(\"enabling dynamic prompt selection and data cleanup.\")
    print(\"=\"*60)


def demonstrate_conditional_logic():
    \"\"\"Demonstrate conditional logic in chains.\"\"\"
    logger = get_logger(__name__)
    logger.info(\"🔀 Conditional Chain Logic\")
    
    def route_request(user_input: str) -> str:
        \"\"\"Route user requests to appropriate handlers.\"\"\"
        user_input = user_input.lower()
        
        if any(word in user_input for word in ['code', 'programming', 'function', 'debug']):
            return \"technical\"
        elif any(word in user_input for word in ['story', 'creative', 'write', 'poem']):
            return \"creative\"
        elif any(word in user_input for word in ['explain', 'what is', 'how does', 'definition']):
            return \"educational\"
        else:
            return \"general\"
    
    # Different prompt templates for each route
    prompt_templates = {
        \"technical\": \"You are a senior software engineer. Help with this technical request: {input}\",
        \"creative\": \"You are a creative writer. Assist with this creative request: {input}\",
        \"educational\": \"You are a knowledgeable teacher. Explain this clearly: {input}\",
        \"general\": \"You are a helpful assistant. Assist with: {input}\"
    }
    
    print(\"\n\" + \"=\"*60)
    print(\"CONDITIONAL CHAIN LOGIC\")
    print(\"=\"*60)
    
    test_requests = [
        \"Help me debug this Python function\",
        \"Write a short story about space exploration\", 
        \"What is quantum computing?\",
        \"I need help planning my vacation\"
    ]
    
    for request in test_requests:
        route = route_request(request)
        template = prompt_templates[route]
        
        print(f\"\n📝 Request: {request}\")
        print(f\"🎯 Route: {route}\")
        print(f\"🔧 Template: {template}\")
        print(\"-\" * 40)
    
    print(\"\n💡 Conditional logic enables smart routing based on input analysis,\")
    print(\"directing requests to specialized prompts and handlers.\")
    print(\"=\"*60)


def demonstrate_chain_composition():
    \"\"\"Demonstrate different chain composition patterns.\"\"\"
    logger = get_logger(__name__)
    logger.info(\"🏗️ Chain Composition Patterns\")
    
    patterns = {
        \"Linear\": \"Input → Chain1 → Chain2 → Chain3 → Output\",
        \"Parallel\": \"Input → [Chain1, Chain2, Chain3] → Combine → Output\",
        \"Conditional\": \"Input → Router → [Chain1 OR Chain2 OR Chain3] → Output\",
        \"Feedback Loop\": \"Input → Chain1 → Validator → [Chain2 OR retry Chain1] → Output\",
        \"Map-Reduce\": \"Input → Split → [Chain1, Chain2, ...] → Merge → Output\"
    }
    
    print(\"\n\" + \"=\"*60)
    print(\"CHAIN COMPOSITION PATTERNS\")
    print(\"=\"*60)
    
    for pattern_name, pattern_flow in patterns.items():
        print(f\"\n🔧 {pattern_name} Pattern:\")
        print(f\"   {pattern_flow}\")
        
        # Add use case examples
        use_cases = {
            \"Linear\": \"Content creation: brainstorm → write → edit → format\",
            \"Parallel\": \"Multi-aspect analysis: sentiment + topics + summary\",
            \"Conditional\": \"Smart routing: technical vs creative vs educational\",
            \"Feedback Loop\": \"Quality assurance: generate → validate → improve\",
            \"Map-Reduce\": \"Document processing: split → analyze each → combine results\"
        }
        
        print(f\"   Use case: {use_cases[pattern_name]}\")
        print(\"-\" * 50)
    
    print(\"\n💡 Choose composition patterns based on your workflow requirements:\")
    print(\"• Linear: Sequential processing with dependencies\")
    print(\"• Parallel: Independent processing for efficiency\")
    print(\"• Conditional: Smart routing based on input type\")
    print(\"• Feedback: Quality control and iterative improvement\")
    print(\"• Map-Reduce: Large-scale data processing\")
    print(\"=\"*60)


def main():
    \"\"\"Main function demonstrating chain concepts.\"\"\"
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info(\"🚀 Starting LangChain Chains Demonstration\")
    
    try:
        # Run all demonstrations
        demonstrate_simple_llm_chain()
        demonstrate_sequential_chain()
        demonstrate_simple_sequential_chain()
        demonstrate_transform_chain()
        demonstrate_conditional_logic()
        demonstrate_chain_composition()
        
        print(\"\n🎯 Chain Key Takeaways:\")
        print(\"1. Chains connect prompts and LLMs into workflows\")
        print(\"2. Sequential chains pass output from one step to the next\")
        print(\"3. Transform chains preprocess and modify data\")
        print(\"4. Conditional logic enables smart routing\")
        print(\"5. Different composition patterns serve different needs\")
        print(\"6. Chains make complex workflows reusable and maintainable\")
        
        logger.info(\"✅ LangChain Chains demonstration completed successfully!\")
        
    except Exception as e:
        logger.error(f\"❌ Error occurred: {e}\")
        logger.info(\"💡 Check your API keys and internet connection\")


if __name__ == \"__main__\":
    main()