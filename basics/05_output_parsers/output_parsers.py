#!/usr/bin/env python3
\"\"\"
Output Parsers - Structured Data from LLM Responses

This example demonstrates:
1. Built-in output parsers
2. Custom output parsers
3. Pydantic parsers for complex data
4. JSON output parsing
5. List and comma-separated parsers
6. Error handling and validation

Key concepts:
- Converting raw text to structured data
- Type validation and error handling
- Custom parsing logic
- Integration with data models
\"\"\"

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# Add utils to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import setup_logging, get_logger, get_api_key, timing_decorator
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser, OutputParserException
from langchain.output_parsers import (
    PydanticOutputParser,
    CommaSeparatedListOutputParser,
    DatetimeOutputParser,
    OutputFixingParser,
    RetryOutputParser
)
from pydantic import BaseModel, Field, validator


# Pydantic models for structured output
class ProductReview(BaseModel):
    \"\"\"Model for product review analysis.\"\"\"
    product_name: str = Field(description=\"Name of the product\")
    rating: int = Field(description=\"Rating from 1-5\", ge=1, le=5)
    sentiment: str = Field(description=\"Overall sentiment: positive, negative, or neutral\")
    key_points: List[str] = Field(description=\"Main points from the review\")
    recommended: bool = Field(description=\"Whether the product is recommended\")
    
    @validator('sentiment')
    def validate_sentiment(cls, v):
        if v.lower() not in ['positive', 'negative', 'neutral']:
            raise ValueError('Sentiment must be positive, negative, or neutral')
        return v.lower()


class PersonInfo(BaseModel):
    \"\"\"Model for person information extraction.\"\"\"
    name: str = Field(description=\"Full name of the person\")
    age: Optional[int] = Field(description=\"Age in years\", ge=0, le=150)
    occupation: str = Field(description=\"Job or profession\")
    location: str = Field(description=\"City or location\")
    interests: List[str] = Field(description=\"Hobbies or interests\")


class CustomJSONParser(BaseOutputParser):
    \"\"\"Custom parser for JSON-like output with error handling.\"\"\"
    
    def parse(self, text: str) -> Dict[str, Any]:
        \"\"\"Parse text into JSON, handling common formatting issues.\"\"\"
        # Clean the text
        text = text.strip()
        
        # Try to find JSON-like content
        start_idx = text.find('{')
        end_idx = text.rfind('}') + 1
        
        if start_idx != -1 and end_idx != 0:
            json_text = text[start_idx:end_idx]
        else:
            json_text = text
        
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            # Try to fix common issues
            fixed_text = self._fix_json_format(json_text)
            try:
                return json.loads(fixed_text)
            except json.JSONDecodeError as e:
                raise OutputParserException(f\"Failed to parse JSON: {e}\")
    
    def _fix_json_format(self, text: str) -> str:
        \"\"\"Attempt to fix common JSON formatting issues.\"\"\"
        # Add quotes around unquoted keys
        import re
        text = re.sub(r'(\\w+):', r'\"\\1\":', text)
        
        # Fix single quotes to double quotes
        text = text.replace(\"'\", '\"')
        
        # Remove trailing commas
        text = re.sub(r',\\s*}', '}', text)
        text = re.sub(r',\\s*]', ']', text)
        
        return text
    
    @property
    def _type(self) -> str:
        return \"custom_json\"


class EmailExtractorParser(BaseOutputParser):
    \"\"\"Custom parser to extract email addresses from text.\"\"\"
    
    def parse(self, text: str) -> List[str]:
        \"\"\"Extract email addresses from text.\"\"\"
        import re
        email_pattern = r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'
        emails = re.findall(email_pattern, text)
        return list(set(emails))  # Remove duplicates
    
    @property
    def _type(self) -> str:
        return \"email_extractor\"


def demonstrate_built_in_parsers():
    \"\"\"Demonstrate built-in output parsers.\"\"\"
    logger = get_logger(__name__)
    logger.info(\"🔧 Built-in Output Parsers\")
    
    print(\"\n\" + \"=\"*60)
    print(\"BUILT-IN OUTPUT PARSERS\")
    print(\"=\"*60)
    
    # 1. Comma Separated List Parser
    list_parser = CommaSeparatedListOutputParser()
    print(\"\n1. Comma Separated List Parser:\")
    print(f\"Format Instructions: {list_parser.get_format_instructions()}\")
    
    sample_list_output = \"apple, banana, orange, grape, strawberry\"
    parsed_list = list_parser.parse(sample_list_output)
    print(f\"Input: {sample_list_output}\")
    print(f\"Parsed: {parsed_list}\")
    print(f\"Type: {type(parsed_list)}\")
    
    # 2. Datetime Parser
    datetime_parser = DatetimeOutputParser()
    print(\"\n2. Datetime Parser:\")
    print(f\"Format Instructions: {datetime_parser.get_format_instructions()}\")
    
    sample_datetime = \"2024-03-15T14:30:00.000Z\"
    try:
        parsed_datetime = datetime_parser.parse(sample_datetime)
        print(f\"Input: {sample_datetime}\")
        print(f\"Parsed: {parsed_datetime}\")
        print(f\"Type: {type(parsed_datetime)}\")
    except Exception as e:
        print(f\"Error parsing datetime: {e}\")
    
    print(\"=\"*60)


@timing_decorator
def demonstrate_pydantic_parser():
    \"\"\"Demonstrate Pydantic output parser for structured data.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping demonstration\")
        return
    
    logger.info(\"📊 Pydantic Output Parser\")
    
    # Create parser for ProductReview model
    parser = PydanticOutputParser(pydantic_object=ProductReview)
    
    # Create prompt with format instructions
    prompt = PromptTemplate(
        template=\"\"\"Analyze this product review and extract structured information:
        
        Review: {review}
        
        {format_instructions}\"\"\",
        input_variables=[\"review\"],
        partial_variables={\"format_instructions\": parser.get_format_instructions()}
    )
    
    # Create chain
    llm = OpenAI(openai_api_key=api_key, temperature=0.3, max_tokens=300)
    chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)
    
    print(\"\n\" + \"=\"*60)
    print(\"PYDANTIC OUTPUT PARSER\")
    print(\"=\"*60)
    
    print(\"\nFormat Instructions:\")
    print(parser.get_format_instructions())
    
    # Test reviews
    reviews = [
        \"The XYZ Bluetooth Speaker is amazing! Great sound quality and battery life lasts all day. Definitely worth the money. I'd rate it 5 stars and recommend it to everyone.\",
        \"This Smart Watch is okay but not great. The interface is confusing and battery dies quickly. Some useful features though. Maybe 3 out of 5 stars. Not sure if I'd recommend it.\"
    ]
    
    for i, review in enumerate(reviews, 1):
        try:
            result = chain.run(review=review)
            print(f\"\n📝 Review {i}:\")
            print(f\"Input: {review[:100]}...\")
            print(f\"\n📊 Parsed Output:\")
            print(f\"Product: {result.product_name}\")
            print(f\"Rating: {result.rating}/5\")
            print(f\"Sentiment: {result.sentiment}\")
            print(f\"Key Points: {result.key_points}\")
            print(f\"Recommended: {result.recommended}\")
            print(\"-\" * 40)
            
        except Exception as e:
            logger.error(f\"Error parsing review {i}: {e}\")
    
    print(\"=\"*60)


def demonstrate_custom_parsers():
    \"\"\"Demonstrate custom output parsers.\"\"\"
    logger = get_logger(__name__)
    logger.info(\"🛠️ Custom Output Parsers\")
    
    print(\"\n\" + \"=\"*60)
    print(\"CUSTOM OUTPUT PARSERS\")
    print(\"=\"*60)
    
    # 1. Custom JSON Parser
    json_parser = CustomJSONParser()
    
    print(\"\n1. Custom JSON Parser:\")
    test_json_inputs = [
        '{\"name\": \"John\", \"age\": 30}',  # Valid JSON
        \"{name: 'Jane', age: 25}\",      # Invalid JSON (single quotes, no quotes on keys)
        '{ \"product\": \"laptop\", \"price\": 999, }',  # Trailing comma
    ]
    
    for i, json_input in enumerate(test_json_inputs, 1):
        try:
            result = json_parser.parse(json_input)
            print(f\"Input {i}: {json_input}\")
            print(f\"Parsed: {result}\")
            print(f\"Type: {type(result)}\")
        except OutputParserException as e:
            print(f\"Input {i}: Failed to parse - {e}\")
        print(\"-\" * 30)
    
    # 2. Email Extractor Parser
    email_parser = EmailExtractorParser()
    
    print(\"\n2. Email Extractor Parser:\")
    text_with_emails = \"\"\"Contact us at support@company.com or sales@business.org. 
    For urgent matters, reach out to admin@urgent.net. 
    Invalid emails like @invalid or missing@ should be ignored.\"\"\"
    
    extracted_emails = email_parser.parse(text_with_emails)
    print(f\"Input text: {text_with_emails}\")
    print(f\"Extracted emails: {extracted_emails}\")
    
    print(\"=\"*60)


def demonstrate_error_handling():
    \"\"\"Demonstrate error handling and fixing parsers.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, showing parser concepts only\")
    
    logger.info(\"🛡️ Error Handling & Output Fixing\")
    
    print(\"\n\" + \"=\"*60)
    print(\"ERROR HANDLING & OUTPUT FIXING\")
    print(\"=\"*60)
    
    # Create a parser that might fail
    base_parser = PydanticOutputParser(pydantic_object=PersonInfo)
    
    print(\"\n1. Base Parser (might fail):\")
    print(base_parser.get_format_instructions())
    
    # Malformed JSON that might come from LLM
    malformed_output = \"\"\"{
        name: \"Alice Smith\",
        age: 28,
        occupation: \"Software Engineer\",
        location: \"San Francisco\",
        interests: [\"coding\", \"hiking\", \"photography\"]
    }\"\"\"
    
    print(f\"\nMalformed Output:\n{malformed_output}\")
    
    try:
        result = base_parser.parse(malformed_output)
        print(f\"Base parser succeeded: {result}\")
    except Exception as e:
        print(f\"Base parser failed: {e}\")
        
        if api_key:
            # Try with OutputFixingParser
            print(\"\n2. Using OutputFixingParser to auto-fix:\")
            llm = OpenAI(openai_api_key=api_key, temperature=0)
            fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)
            
            try:
                fixed_result = fixing_parser.parse(malformed_output)
                print(f\"Fixing parser succeeded: {fixed_result}\")
            except Exception as fix_error:
                print(f\"Fixing parser also failed: {fix_error}\")
    
    print(\"\n💡 Error Handling Strategies:\")
    strategies = [
        \"Use OutputFixingParser to auto-correct malformed output\",
        \"Implement custom parsers with robust error handling\",
        \"Provide clear format instructions in prompts\",
        \"Use RetryOutputParser to re-attempt with better prompts\",
        \"Validate data types and ranges in Pydantic models\",
        \"Log parsing errors for analysis and improvement\"
    ]
    
    for strategy in strategies:
        print(f\"• {strategy}\")
    
    print(\"=\"*60)


def demonstrate_parser_integration():
    \"\"\"Demonstrate integrating parsers with chains.\"\"\"
    logger = get_logger(__name__)
    
    api_key = get_api_key('openai')
    if not api_key:
        logger.warning(\"⚠️ OpenAI API key not found, skipping integration demo\")
        return
    
    logger.info(\"🔗 Parser Integration with Chains\")
    
    print(\"\n\" + \"=\"*60)
    print(\"PARSER INTEGRATION WITH CHAINS\")
    print(\"=\"*60)
    
    # Create a chain that extracts structured data from text
    parser = PydanticOutputParser(pydantic_object=PersonInfo)
    
    prompt = PromptTemplate(
        template=\"\"\"Extract person information from this text:
        
        Text: {text}
        
        {format_instructions}\"\"\",
        input_variables=[\"text\"],
        partial_variables={\"format_instructions\": parser.get_format_instructions()}
    )
    
    llm = OpenAI(openai_api_key=api_key, temperature=0.2, max_tokens=300)
    chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)
    
    # Test texts
    test_texts = [
        \"Meet Sarah Johnson, a 32-year-old data scientist from Seattle. She loves rock climbing, cooking, and reading sci-fi novels.\",
        \"Dr. Michael Chen, age 45, works as a cardiologist in Boston. His hobbies include tennis, gardening, and classical music.\"
    ]
    
    for i, text in enumerate(test_texts, 1):
        try:
            result = chain.run(text=text)
            print(f\"\n👤 Person {i}:\")
            print(f\"Input: {text}\")
            print(f\"\n📊 Extracted Information:\")
            print(f\"Name: {result.name}\")
            print(f\"Age: {result.age}\")
            print(f\"Occupation: {result.occupation}\")
            print(f\"Location: {result.location}\")
            print(f\"Interests: {result.interests}\")
            print(\"-\" * 40)
        except Exception as e:
            logger.error(f\"Error processing text {i}: {e}\")
    
    print(\"\n💡 Integration Benefits:\")
    benefits = [
        \"Automatic type conversion and validation\",
        \"Consistent data structure across applications\",
        \"Easy integration with databases and APIs\",
        \"Reduced manual data processing\",
        \"Built-in error handling and recovery\"
    ]
    
    for benefit in benefits:
        print(f\"• {benefit}\")
    
    print(\"=\"*60)


def main():
    \"\"\"Main function demonstrating output parser concepts.\"\"\"
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info(\"🚀 Starting Output Parsers Demonstration\")
    
    try:
        # Run all demonstrations
        demonstrate_built_in_parsers()
        demonstrate_pydantic_parser()
        demonstrate_custom_parsers()
        demonstrate_error_handling()
        demonstrate_parser_integration()
        
        print(\"\n🎯 Output Parser Key Takeaways:\")
        print(\"1. Parsers convert raw text to structured data types\")
        print(\"2. Pydantic parsers provide robust validation and type safety\")
        print(\"3. Custom parsers handle specialized parsing needs\")
        print(\"4. Error handling ensures robust applications\")
        print(\"5. Parser integration with chains enables automated workflows\")
        print(\"6. Format instructions guide LLMs to produce parseable output\")
        
        logger.info(\"✅ Output Parsers demonstration completed successfully!\")
        
    except Exception as e:
        logger.error(f\"❌ Error occurred: {e}\")
        logger.info(\"💡 Check your setup and API keys\")


if __name__ == \"__main__\":
    main()