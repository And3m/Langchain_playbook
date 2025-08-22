# Output Parsers - Structured Data from LLM Responses

This section covers LangChain output parsers - essential tools for converting raw LLM text responses into structured, usable data formats.

## Files in this Section

### 1. `output_parsers.py`
**Complete guide to output parsing**

- Built-in parser types (list, datetime, JSON)
- Pydantic parsers for complex data models
- Custom parser implementations
- Error handling and validation
- Parser integration with chains
- Best practices and optimization

**Run it:**
```bash
python output_parsers.py
```

### 2. `validation_examples.py` (Coming soon)
**Advanced validation and error recovery**

- Data validation strategies
- Auto-fixing malformed output
- Retry mechanisms
- Fallback parsing logic
- Schema evolution handling

## What You'll Learn

1. **Parser Fundamentals**: Converting text to structured data
2. **Built-in Parsers**: Using LangChain's ready-made parsers
3. **Custom Parsers**: Building specialized parsing logic
4. **Validation**: Ensuring data quality and type safety
5. **Error Handling**: Robust parsing with fallback strategies
6. **Integration**: Seamless parser-chain workflows

## Key Concepts

### Why Output Parsers?

❌ **Without Parsers**:
```python
response = llm(\"List 5 colors\")
# Returns: \"1. Red\n2. Blue\n3. Green\n4. Yellow\n5. Purple\"
# Manual parsing required!
```

✅ **With Parsers**:
```python
parser = CommaSeparatedListOutputParser()
chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)
colors = chain.run(\"List 5 colors\")
# Returns: [\"Red\", \"Blue\", \"Green\", \"Yellow\", \"Purple\"]
```

### Parser Types

| Parser Type | Input | Output | Use Case |
|-------------|-------|--------|----------|
| **CommaSeparatedList** | \"a, b, c\" | `[\"a\", \"b\", \"c\"]` | Simple lists |
| **PydanticOutputParser** | JSON-like text | Pydantic model | Complex objects |
| **DatetimeOutputParser** | ISO datetime string | `datetime` object | Dates and times |
| **JSONOutputParser** | JSON string | `dict` | Structured data |
| **CustomParser** | Any format | Custom type | Specialized needs |

## Built-in Parsers

### 1. List Parser
```python
from langchain.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()
result = parser.parse(\"apple, banana, orange\")
# Output: [\"apple\", \"banana\", \"orange\"]
```

### 2. Datetime Parser
```python
from langchain.output_parsers import DatetimeOutputParser

parser = DatetimeOutputParser()
result = parser.parse(\"2024-03-15T14:30:00.000Z\")
# Output: datetime(2024, 3, 15, 14, 30)
```

### 3. Pydantic Parser
```python
from pydantic import BaseModel
from langchain.output_parsers import PydanticOutputParser

class Person(BaseModel):
    name: str
    age: int
    occupation: str

parser = PydanticOutputParser(pydantic_object=Person)
```

## Custom Parser Example

```python
class EmailExtractorParser(BaseOutputParser):
    def parse(self, text: str) -> List[str]:
        import re
        pattern = r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'
        return re.findall(pattern, text)
    
    @property
    def _type(self) -> str:
        return \"email_extractor\"
```

## Integration Patterns

### With Prompts
```python
parser = PydanticOutputParser(pydantic_object=ProductReview)

prompt = PromptTemplate(
    template=\"Analyze this review: {review}\n{format_instructions}\",
    input_variables=[\"review\"],
    partial_variables={\"format_instructions\": parser.get_format_instructions()}
)
```

### With Chains
```python
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    output_parser=parser  # Automatic parsing!
)

result = chain.run(review=\"Great product!\")
# Returns structured ProductReview object
```

## Error Handling Strategies

### 1. OutputFixingParser
Automatically fixes malformed output using an LLM:
```python
from langchain.output_parsers import OutputFixingParser

base_parser = PydanticOutputParser(pydantic_object=Person)
fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)

# Handles malformed JSON automatically
result = fixing_parser.parse(malformed_json)
```

### 2. RetryOutputParser
Retries with better prompts on failure:
```python
from langchain.output_parsers import RetryOutputParser

retry_parser = RetryOutputParser.from_llm(parser=base_parser, llm=llm)
```

### 3. Custom Error Handling
```python
class RobustParser(BaseOutputParser):
    def parse(self, text: str):
        try:
            return self._primary_parse(text)
        except Exception:
            return self._fallback_parse(text)
```

## Best Practices

### ✅ Do
- Always include format instructions in prompts
- Use Pydantic models for complex data structures
- Implement proper error handling
- Test parsers with various input formats
- Validate parsed data before use
- Log parsing failures for analysis

### ❌ Avoid
- Assuming LLM output will always be perfectly formatted
- Ignoring parsing errors
- Over-complicating parser logic
- Not providing clear format instructions
- Parsing without validation

## Performance Tips

### Optimization Strategies
1. **Simple Parsers First**: Start with built-in parsers
2. **Efficient Regex**: Optimize regular expressions
3. **Validation Levels**: Balance speed vs. thoroughness
4. **Caching**: Cache parsing results when appropriate
5. **Batch Processing**: Parse multiple outputs together

### Cost Considerations
- OutputFixingParser adds LLM calls (costs money)
- RetryOutputParser may retry multiple times
- Consider fallback to simpler parsing for cost control

## Common Use Cases

### Data Extraction
```python
# Extract structured information from unstructured text
class ContactInfo(BaseModel):
    name: str
    email: str
    phone: Optional[str]
    company: str
```

### Classification Tasks
```python
# Categorize content with confidence scores
class Classification(BaseModel):
    category: str
    confidence: float
    reasoning: str
```

### Content Analysis
```python
# Analyze sentiment and extract key points
class SentimentAnalysis(BaseModel):
    sentiment: str  # positive, negative, neutral
    confidence: float
    key_phrases: List[str]
    emotions: List[str]
```

## Prerequisites

- Understanding of chains (see `../04_chains/`)
- Basic Python data types and JSON
- Familiarity with Pydantic (helpful but not required)

## Next Steps

After mastering output parsers:
1. Explore intermediate concepts in `../../intermediate/`
2. Build memory-enabled applications in `../../intermediate/01_memory/`
3. Create intelligent agents in `../../intermediate/02_agents/`
4. Start real projects in `../../projects/`

---

**Ready to structure your LLM outputs? Start with `python output_parsers.py` 📊**