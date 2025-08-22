#!/usr/bin/env python3
\"\"\"
Prompt Templates - Mastering Prompt Engineering

This example demonstrates:
1. Basic prompt templates
2. Variables and formatting
3. Few-shot prompting
4. Prompt composition
5. Chat prompt templates
6. Best practices for prompt engineering

Key concepts:
- Template variables and substitution
- Prompt reusability and modularity
- Different prompt types for different tasks
- Prompt optimization techniques
\"\"\"

import sys
from pathlib import Path
from typing import List, Dict

# Add utils to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import setup_logging, get_logger, get_api_key, timing_decorator
from langchain.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


def demonstrate_basic_prompt_template():
    \"\"\"Demonstrate basic prompt template usage.\"\"\"
    logger = get_logger(__name__)
    logger.info(\"📝 Basic Prompt Templates\")
    
    # Simple template with one variable
    simple_template = PromptTemplate(
        input_variables=[\"topic\"],
        template=\"Write a short paragraph about {topic}.\"
    )
    
    # Multiple variables
    detailed_template = PromptTemplate(
        input_variables=[\"subject\", \"audience\", \"length\"],
        template=\"\"\"Write a {length} explanation about {subject} for {audience}.
        Make it engaging and easy to understand.\"\"\"
    )
    
    print(\"\n\" + \"=\"*60)
    print(\"BASIC PROMPT TEMPLATES\")
    print(\"=\"*60)
    
    # Format simple template
    simple_prompt = simple_template.format(topic=\"artificial intelligence\")
    print(\"Simple Template:\")
    print(f\"Template: {simple_template.template}\")
    print(f\"Formatted: {simple_prompt}\")
    print(\"-\" * 40)
    
    # Format detailed template
    detailed_prompt = detailed_template.format(
        subject=\"machine learning\",
        audience=\"high school students\",
        length=\"brief\"
    )
    print(\"Detailed Template:\")
    print(f\"Template: {detailed_template.template}\")
    print(f\"Formatted: {detailed_prompt}\")
    print(\"=\"*60)


@timing_decorator
def demonstrate_prompt_with_llm():
    \"\"\"Demonstrate using prompt templates with LLMs.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping LLM demonstration\")
        return
    
    logger.info(\"🤖 Using Prompt Templates with LLMs\")
    
    # Create LLM
    llm = OpenAI(openai_api_key=api_key, temperature=0.7, max_tokens=150)
    
    # Create template
    template = PromptTemplate(
        input_variables=[\"product\", \"feature\"],
        template=\"\"\"You are a marketing copywriter. Write a compelling product description for:
        Product: {product}
        Key Feature: {feature}
        
        Make it persuasive and highlight the benefits.\"\"\"
    )
    
    # Test with different inputs
    test_cases = [
        {\"product\": \"Smart Water Bottle\", \"feature\": \"Temperature tracking\"},
        {\"product\": \"Wireless Earbuds\", \"feature\": \"Noise cancellation\"}
    ]
    
    print(\"\n\" + \"=\"*60)
    print(\"PROMPT TEMPLATES WITH LLM\")
    print(\"=\"*60)
    
    for case in test_cases:
        try:
            prompt = template.format(**case)
            response = llm(prompt)
            
            print(f\"Product: {case['product']}\")
            print(f\"Feature: {case['feature']}\")
            print(f\"Generated Description:\n{response.strip()}\")
            print(\"-\" * 40)
            
        except Exception as e:
            logger.error(f\"Error processing {case}: {e}\")


def demonstrate_few_shot_prompting():
    \"\"\"Demonstrate few-shot prompting techniques.\"\"\"
    logger = get_logger(__name__)
    logger.info(\"🎯 Few-Shot Prompting\")
    
    # Define examples for few-shot learning
    examples = [
        {
            \"word\": \"happy\",
            \"antonym\": \"sad\",
            \"sentence\": \"The happy child played in the sunny park, but later felt sad when it started raining.\"
        },
        {
            \"word\": \"hot\",
            \"antonym\": \"cold\", 
            \"sentence\": \"The hot summer day made everyone seek cold drinks to cool down.\"
        },
        {
            \"word\": \"fast\",
            \"antonym\": \"slow\",
            \"sentence\": \"The fast car overtook the slow bicycle on the highway.\"
        }
    ]
    
    # Example template
    example_template = PromptTemplate(
        input_variables=[\"word\", \"antonym\", \"sentence\"],
        template=\"Word: {word}\nAntonym: {antonym}\nSentence: {sentence}\"
    )
    
    # Create few-shot prompt template
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_template,
        prefix=\"Given a word, provide its antonym and create a sentence using both words:\",
        suffix=\"Word: {input_word}\nAntonym:\",
        input_variables=[\"input_word\"]
    )
    
    print(\"\n\" + \"=\"*60)
    print(\"FEW-SHOT PROMPTING\")
    print(\"=\"*60)
    
    # Format the prompt
    formatted_prompt = few_shot_prompt.format(input_word=\"bright\")
    print(\"Few-Shot Prompt:\")
    print(formatted_prompt)
    print(\"=\"*60)
    
    # Test with LLM if available
    api_key = get_api_key('openai')
    if api_key:
        try:
            llm = OpenAI(openai_api_key=api_key, temperature=0.3, max_tokens=100)
            response = llm(formatted_prompt)
            print(\"\nLLM Response:\")
            print(f\"Word: bright\nAntonym:{response.strip()}\")
        except Exception as e:
            logger.error(f\"LLM error: {e}\")


def demonstrate_chat_prompt_templates():
    \"\"\"Demonstrate chat-specific prompt templates.\"\"\"
    logger = get_logger(__name__)
    logger.info(\"💬 Chat Prompt Templates\")
    
    # Create chat prompt template
    chat_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            \"You are a helpful {role} assistant. Your expertise is in {domain}.\"
        ),
        HumanMessagePromptTemplate.from_template(
            \"I need help with: {request}\"
        )
    ])
    
    print(\"\n\" + \"=\"*60)
    print(\"CHAT PROMPT TEMPLATES\")
    print(\"=\"*60)
    
    # Test different scenarios
    scenarios = [
        {
            \"role\": \"cooking\", 
            \"domain\": \"Italian cuisine\",
            \"request\": \"making authentic pasta carbonara\"
        },
        {
            \"role\": \"fitness\",
            \"domain\": \"strength training\", 
            \"request\": \"creating a beginner workout routine\"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        messages = chat_template.format_prompt(**scenario).to_messages()
        
        print(f\"Scenario {i}:\")
        for msg in messages:
            msg_type = type(msg).__name__.replace('Message', '')
            print(f\"{msg_type}: {msg.content}\")
        print(\"-\" * 40)
    
    # Test with ChatGPT if available
    api_key = get_api_key('openai')
    if api_key:
        try:
            chat = ChatOpenAI(openai_api_key=api_key, temperature=0.7, max_tokens=150)
            
            # Use first scenario
            messages = chat_template.format_prompt(**scenarios[0]).to_messages()
            response = chat(messages)
            
            print(\"\nChat Response:\")
            print(f\"Assistant: {response.content}\")
            
        except Exception as e:
            logger.error(f\"Chat error: {e}\")


def demonstrate_prompt_composition():
    \"\"\"Demonstrate composing complex prompts from parts.\"\"\"
    logger = get_logger(__name__)
    logger.info(\"🔧 Prompt Composition\")
    
    # Define reusable prompt parts
    context_template = PromptTemplate(
        input_variables=[\"context\"],
        template=\"Context: {context}\n\"
    )
    
    task_template = PromptTemplate(
        input_variables=[\"task\", \"format\"],
        template=\"Task: {task}\nFormat: {format}\n\"
    )
    
    constraints_template = PromptTemplate(
        input_variables=[\"constraints\"],
        template=\"Constraints: {constraints}\n\"
    )
    
    # Compose full prompt
    full_template = PromptTemplate(
        input_variables=[\"context\", \"task\", \"format\", \"constraints\", \"input\"],
        template=\"\"\"{context}{task}{constraints}
Input: {input}

Output:\"\"\"
    )
    
    print(\"\n\" + \"=\"*60)
    print(\"PROMPT COMPOSITION\")
    print(\"=\"*60)
    
    # Example composition
    context = \"You are analyzing customer feedback for a software product.\"
    task = \"Classify the sentiment and extract key issues\"
    format_req = \"JSON with sentiment (positive/negative/neutral) and issues array\"
    constraints = \"Be objective and specific. Limit issues to top 3.\"
    user_input = \"The app crashes frequently and the UI is confusing, but I love the new features.\"
    
    composed_prompt = full_template.format(
        context=context_template.format(context=context),
        task=task_template.format(task=task, format=format_req),
        constraints=constraints_template.format(constraints=constraints),
        input=user_input
    )
    
    print(\"Composed Prompt:\")
    print(composed_prompt)
    print(\"=\"*60)


def prompt_engineering_tips():
    \"\"\"Share prompt engineering best practices.\"\"\"
    logger = get_logger(__name__)
    logger.info(\"💡 Prompt Engineering Tips\")
    
    tips = [
        \"Be specific and clear in your instructions\",
        \"Provide examples when possible (few-shot learning)\",
        \"Use consistent formatting and structure\",
        \"Set the context and role clearly\",
        \"Specify the desired output format\",
        \"Use constraints to guide behavior\",
        \"Test with different phrasings\",
        \"Iterate and refine based on results\",
        \"Consider the model's training and capabilities\",
        \"Use system messages for persistent instructions\"
    ]
    
    print(\"\n\" + \"=\"*60)
    print(\"PROMPT ENGINEERING BEST PRACTICES\")
    print(\"=\"*60)
    
    for i, tip in enumerate(tips, 1):
        print(f\"{i:2d}. {tip}\")
    
    print(\"\n💡 Template Design Patterns:\")
    patterns = {
        \"Task + Context + Examples\": \"Most reliable for specific tasks\",
        \"Role + Task + Constraints\": \"Good for creative tasks\", 
        \"Few-shot + Input\": \"Excellent for pattern recognition\",
        \"Chain of Thought\": \"Best for reasoning tasks\",
        \"System + User + Assistant\": \"Ideal for conversations\"
    }
    
    for pattern, usage in patterns.items():
        print(f\"• {pattern}: {usage}\")
    
    print(\"=\"*60)


def main():
    \"\"\"Main function demonstrating prompt engineering concepts.\"\"\"
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info(\"🚀 Starting Prompt Templates & Engineering\")
    
    try:
        # Run all demonstrations
        demonstrate_basic_prompt_template()
        demonstrate_prompt_with_llm()
        demonstrate_few_shot_prompting()
        demonstrate_chat_prompt_templates()
        demonstrate_prompt_composition()
        prompt_engineering_tips()
        
        print(\"\n🎯 Key Takeaways:\")
        print(\"1. Templates make prompts reusable and maintainable\")
        print(\"2. Variables allow dynamic content insertion\")
        print(\"3. Few-shot learning improves task performance\")
        print(\"4. Chat templates structure conversations properly\")
        print(\"5. Composition enables building complex prompts from parts\")
        print(\"6. Good prompt engineering is crucial for LLM success\")
        
        logger.info(\"✅ Prompt Templates & Engineering completed successfully!\")
        
    except Exception as e:
        logger.error(f\"❌ Error occurred: {e}\")
        logger.info(\"💡 Check your setup and API keys\")


if __name__ == \"__main__\":
    main()