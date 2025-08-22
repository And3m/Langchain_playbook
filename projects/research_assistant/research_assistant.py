#!/usr/bin/env python3
"""
Research Assistant - AI-Powered Information Gathering and Synthesis

This project demonstrates:
1. Automated research and information gathering
2. Source verification and credibility assessment
3. Information synthesis and summarization
4. Multi-source data aggregation
5. Research report generation
6. Citation management

Key features:
- Web search integration
- Document analysis and extraction
- Information synthesis and summarization
- Source citation and references
- Research methodology suggestions
- Fact-checking and verification
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

# Add utils to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import setup_logging, get_logger, get_api_key, timing_decorator
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter


@dataclass
class ResearchSource:
    """Represents a research source with metadata."""
    title: str
    url: str
    content: str
    credibility_score: float
    publication_date: Optional[str] = None
    author: Optional[str] = None
    source_type: str = "web"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content[:500] + "..." if len(self.content) > 500 else self.content,
            "credibility_score": self.credibility_score,
            "publication_date": self.publication_date,
            "author": self.author,
            "source_type": self.source_type
        }


class ResearchAssistant:
    """AI-powered research assistant for information gathering and synthesis."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.logger = get_logger(self.__class__.__name__)
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name=model,
            temperature=0.3,
            max_tokens=2000
        )
        
        # Research context
        self.research_context = {
            "topic": "",
            "sources": [],
            "findings": [],
            "questions": [],
            "methodology": []
        }
        
        # Initialize specialized chains
        self._setup_chains()
        
    def _setup_chains(self):
        """Set up specialized chains for research tasks."""
        
        # Research planning chain
        self.research_plan_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a research methodology expert. Create comprehensive 
            research plans with clear objectives, questions, and methodologies."""),
            HumanMessage(content="""
            Research Topic: {topic}
            Research Goal: {goal}
            Available Resources: {resources}
            
            Please provide:
            1. Clear research objectives
            2. Key research questions to investigate
            3. Suggested research methodology
            4. Recommended sources and databases
            5. Timeline and milestones
            """)
        ])
        
        self.research_planning_chain = LLMChain(
            llm=self.llm,
            prompt=self.research_plan_prompt
        )
        
        # Information synthesis chain
        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert researcher skilled at synthesizing 
            information from multiple sources. Create coherent, well-structured summaries."""),
            HumanMessage(content="""
            Research Topic: {topic}
            Sources and Information:
            {sources_info}
            
            Please synthesize this information into:
            1. Executive summary (key findings)
            2. Main themes and patterns
            3. Supporting evidence from sources
            4. Conflicting viewpoints (if any)
            5. Gaps in current research
            6. Recommendations for further investigation
            """)
        ])
        
        self.synthesis_chain = LLMChain(
            llm=self.llm,
            prompt=self.synthesis_prompt
        )
        
        # Source credibility assessment chain
        self.credibility_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an information literacy expert. Assess the 
            credibility and reliability of sources based on various factors."""),
            HumanMessage(content="""
            Assess the credibility of this source:
            
            Title: {title}
            Author: {author}
            Source URL: {url}
            Publication Date: {publication_date}
            Source Type: {source_type}
            Content Sample: {content_sample}
            
            Evaluate based on:
            1. Author expertise and credentials
            2. Publication quality and reputation
            3. Date and relevance
            4. Bias and objectivity
            5. Citation and references
            
            Provide a credibility score (0-10) and explanation.
            """)
        ])
        
        self.credibility_chain = LLMChain(
            llm=self.llm,
            prompt=self.credibility_prompt
        )
        
        # Citation generation chain
        self.citation_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an academic citation expert. Generate proper 
            citations in various formats (APA, MLA, Chicago, etc.)."""),
            HumanMessage(content="""
            Generate citations for these sources in {citation_style} format:
            
            {sources_data}
            
            Provide:
            1. Complete citations for each source
            2. In-text citation examples
            3. Bibliography/References section
            """)
        ])
        
        self.citation_chain = LLMChain(
            llm=self.llm,
            prompt=self.citation_prompt
        )
    
    def create_research_plan(self, topic: str, goal: str = "comprehensive analysis", 
                           resources: str = "online sources, academic databases") -> Dict[str, Any]:
        """Create a structured research plan for a given topic."""
        try:
            self.logger.info(f"Creating research plan for: {topic}")
            
            result = self.research_planning_chain.run(
                topic=topic,
                goal=goal,
                resources=resources
            )
            
            # Update research context
            self.research_context["topic"] = topic
            
            return {
                "status": "success",
                "research_plan": result,
                "topic": topic,
                "goal": goal,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Research planning failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "topic": topic
            }
    
    def assess_source_credibility(self, source: ResearchSource) -> Dict[str, Any]:
        """Assess the credibility of a research source."""
        try:
            self.logger.info(f"Assessing credibility of: {source.title}")
            
            content_sample = source.content[:300] + "..." if len(source.content) > 300 else source.content
            
            result = self.credibility_chain.run(
                title=source.title,
                author=source.author or "Unknown",
                url=source.url,
                publication_date=source.publication_date or "Unknown",
                source_type=source.source_type,
                content_sample=content_sample
            )
            
            # Extract credibility score (simple regex approach)
            score_match = re.search(r'(\d+(?:\.\d+)?)/10', result)
            if score_match:
                score = float(score_match.group(1))
            else:
                # Look for alternative patterns
                score_match = re.search(r'score[:\s]*(\d+(?:\.\d+)?)', result, re.IGNORECASE)
                score = float(score_match.group(1)) if score_match else 5.0
            
            # Update source credibility
            source.credibility_score = score
            
            return {
                "status": "success",
                "credibility_assessment": result,
                "credibility_score": score,
                "source_title": source.title,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Credibility assessment failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "source_title": source.title
            }
    
    def add_source(self, title: str, url: str, content: str, 
                   author: str = None, publication_date: str = None, 
                   source_type: str = "web") -> ResearchSource:
        """Add a new research source and assess its credibility."""
        source = ResearchSource(
            title=title,
            url=url,
            content=content,
            credibility_score=0.0,  # Will be updated by assessment
            author=author,
            publication_date=publication_date,
            source_type=source_type
        )
        
        # Assess credibility
        assessment = self.assess_source_credibility(source)
        if assessment["status"] == "success":
            self.logger.info(f"Added source with credibility score: {source.credibility_score}")
        
        # Add to research context
        self.research_context["sources"].append(source)
        
        return source
    
    def synthesize_information(self, topic: str = None) -> Dict[str, Any]:
        """Synthesize information from all collected sources."""
        try:
            current_topic = topic or self.research_context["topic"]
            if not current_topic:
                return {
                    "status": "error",
                    "error": "No research topic specified"
                }
            
            if not self.research_context["sources"]:
                return {
                    "status": "error",
                    "error": "No sources available for synthesis"
                }
            
            self.logger.info(f"Synthesizing information for: {current_topic}")
            
            # Prepare sources information
            sources_info = []
            for i, source in enumerate(self.research_context["sources"], 1):
                source_text = f"""
                Source {i}: {source.title}
                Author: {source.author or 'Unknown'}
                Credibility Score: {source.credibility_score}/10
                Content: {source.content[:500]}...
                """
                sources_info.append(source_text)
            
            sources_text = "\n".join(sources_info)
            
            result = self.synthesis_chain.run(
                topic=current_topic,
                sources_info=sources_text
            )
            
            return {
                "status": "success",
                "synthesis": result,
                "topic": current_topic,
                "sources_count": len(self.research_context["sources"]),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Information synthesis failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def generate_citations(self, citation_style: str = "APA") -> Dict[str, Any]:
        """Generate citations for all research sources."""
        try:
            if not self.research_context["sources"]:
                return {
                    "status": "error",
                    "error": "No sources available for citation"
                }
            
            self.logger.info(f"Generating {citation_style} citations")
            
            # Prepare sources data for citation
            sources_data = []
            for source in self.research_context["sources"]:
                source_text = f"""
                Title: {source.title}
                Author: {source.author or 'Unknown Author'}
                URL: {source.url}
                Publication Date: {source.publication_date or 'No date'}
                Source Type: {source.source_type}
                """
                sources_data.append(source_text)
            
            sources_text = "\n---\n".join(sources_data)
            
            result = self.citation_chain.run(
                citation_style=citation_style,
                sources_data=sources_text
            )
            
            return {
                "status": "success",
                "citations": result,
                "citation_style": citation_style,
                "sources_count": len(self.research_context["sources"]),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Citation generation failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def generate_research_questions(self, topic: str, focus_areas: List[str] = None) -> Dict[str, Any]:
        """Generate relevant research questions for a topic."""
        try:
            self.logger.info(f"Generating research questions for: {topic}")
            
            focus_text = ""
            if focus_areas:
                focus_text = f"Focus areas: {', '.join(focus_areas)}"
            
            question_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are a research methodology expert. Generate 
                comprehensive, specific, and actionable research questions."""),
                HumanMessage(content=f"""
                Topic: {topic}
                {focus_text}
                
                Generate 10 research questions covering:
                1. Primary research questions (broad, fundamental)
                2. Secondary research questions (specific aspects)
                3. Comparative questions (relationships, differences)
                4. Exploratory questions (new insights)
                5. Applied questions (practical implications)
                
                Make questions specific, measurable, and researchable.
                """)
            ])
            
            question_chain = LLMChain(llm=self.llm, prompt=question_prompt)
            result = question_chain.run(topic=topic, focus_areas=focus_areas)
            
            return {
                "status": "success",
                "research_questions": result,
                "topic": topic,
                "focus_areas": focus_areas,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Research question generation failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def generate_report(self, format_type: str = "executive_summary") -> Dict[str, Any]:
        """Generate a comprehensive research report."""
        try:
            if not self.research_context["topic"] or not self.research_context["sources"]:
                return {
                    "status": "error",
                    "error": "Insufficient research data for report generation"
                }
            
            self.logger.info(f"Generating {format_type} research report")
            
            # First synthesize information
            synthesis_result = self.synthesize_information()
            if synthesis_result["status"] != "success":
                return synthesis_result
            
            # Generate citations
            citation_result = self.generate_citations()
            citations = citation_result.get("citations", "No citations available")
            
            # Create comprehensive report
            report_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are a professional research report writer. 
                Create well-structured, comprehensive reports with proper formatting."""),
                HumanMessage(content=f"""
                Create a {format_type} research report with the following information:
                
                Topic: {self.research_context["topic"]}
                Number of Sources: {len(self.research_context["sources"])}
                
                Research Synthesis:
                {synthesis_result["synthesis"]}
                
                Citations:
                {citations}
                
                Format the report with:
                1. Executive Summary
                2. Introduction and Background
                3. Methodology
                4. Key Findings
                5. Analysis and Discussion
                6. Conclusions and Recommendations
                7. References
                """)
            ])
            
            report_chain = LLMChain(llm=self.llm, prompt=report_prompt)
            result = report_chain.run()
            
            return {
                "status": "success",
                "report": result,
                "format_type": format_type,
                "topic": self.research_context["topic"],
                "sources_used": len(self.research_context["sources"]),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_research_summary(self) -> Dict[str, Any]:
        """Get a summary of the current research session."""
        sources_summary = []
        total_credibility = 0
        
        for source in self.research_context["sources"]:
            sources_summary.append({
                "title": source.title,
                "credibility_score": source.credibility_score,
                "source_type": source.source_type,
                "author": source.author
            })
            total_credibility += source.credibility_score
        
        avg_credibility = total_credibility / len(self.research_context["sources"]) if self.research_context["sources"] else 0
        
        return {
            "topic": self.research_context["topic"],
            "total_sources": len(self.research_context["sources"]),
            "average_credibility": round(avg_credibility, 2),
            "sources_summary": sources_summary,
            "research_status": "active" if self.research_context["sources"] else "not_started"
        }
    
    def clear_research(self):
        """Clear all research data for a new research session."""
        self.research_context = {
            "topic": "",
            "sources": [],
            "findings": [],
            "questions": [],
            "methodology": []
        }
        self.logger.info("Research context cleared")


def demo_research_planning():
    """Demonstrate research planning capabilities."""
    print("\n" + "="*60)
    print("RESEARCH PLANNING DEMO")
    print("="*60)
    
    api_key = get_api_key('openai')
    if not api_key:
        print("⚠️ OpenAI API key not found")
        return
    
    assistant = ResearchAssistant(api_key)
    
    # Create research plan
    topic = "Impact of Artificial Intelligence on Healthcare"
    goal = "Understand current applications, benefits, and challenges"
    
    print(f"📋 Creating research plan for: {topic}")
    result = assistant.create_research_plan(topic, goal)
    
    if result["status"] == "success":
        print("✅ Research Plan:")
        print(result["research_plan"])
    else:
        print(f"❌ Error: {result['error']}")


def demo_source_management():
    """Demonstrate source management and credibility assessment."""
    print("\n" + "="*60)
    print("SOURCE MANAGEMENT DEMO")
    print("="*60)
    
    api_key = get_api_key('openai')
    if not api_key:
        print("⚠️ OpenAI API key not found")
        return
    
    assistant = ResearchAssistant(api_key)
    assistant.research_context["topic"] = "AI in Healthcare"
    
    # Add sample sources
    sources_data = [
        {
            "title": "AI in Medical Diagnosis: A Systematic Review",
            "url": "https://pubmed.ncbi.nlm.nih.gov/example1",
            "content": "This systematic review examines the current state of AI applications in medical diagnosis. The study analyzed 150 peer-reviewed papers published between 2020-2024, focusing on machine learning algorithms used in radiology, pathology, and clinical decision support systems.",
            "author": "Dr. Sarah Johnson",
            "publication_date": "2024-01-15",
            "source_type": "academic_journal"
        },
        {
            "title": "Healthcare AI Market Report 2024",
            "url": "https://techcrunch.com/example2",
            "content": "The global healthcare AI market is expected to reach $102 billion by 2028. This report covers market trends, key players, and emerging technologies in healthcare artificial intelligence.",
            "author": "Tech Reporter",
            "publication_date": "2024-02-01",
            "source_type": "news_article"
        }
    ]
    
    print("📚 Adding and assessing sources:")
    
    for i, source_data in enumerate(sources_data, 1):
        print(f"\n🔍 Source {i}: {source_data['title']}")
        
        source = assistant.add_source(**source_data)
        print(f"   Credibility Score: {source.credibility_score}/10")
        print(f"   Source Type: {source.source_type}")
    
    # Show research summary
    summary = assistant.get_research_summary()
    print(f"\n📊 Research Summary:")
    print(f"   Topic: {summary['topic']}")
    print(f"   Total Sources: {summary['total_sources']}")
    print(f"   Average Credibility: {summary['average_credibility']}/10")


def demo_information_synthesis():
    """Demonstrate information synthesis."""
    print("\n" + "="*60)
    print("INFORMATION SYNTHESIS DEMO")
    print("="*60)
    
    api_key = get_api_key('openai')
    if not api_key:
        print("⚠️ OpenAI API key not found")
        return
    
    assistant = ResearchAssistant(api_key)
    assistant.research_context["topic"] = "AI in Healthcare"
    
    # Add sample sources for synthesis
    sample_sources = [
        ("AI Diagnosis Systems", "AI systems show 95% accuracy in detecting certain cancers", 8.5),
        ("Healthcare Cost Reduction", "AI implementation reduced diagnostic costs by 30% in pilot studies", 7.2),
        ("Ethical Considerations", "Privacy concerns and bias in AI algorithms remain significant challenges", 8.0)
    ]
    
    for title, content, score in sample_sources:
        source = ResearchSource(
            title=title,
            url=f"https://example.com/{title.lower().replace(' ', '-')}",
            content=content * 10,  # Expand content
            credibility_score=score
        )
        assistant.research_context["sources"].append(source)
    
    print("🔬 Synthesizing information from sources...")
    
    result = assistant.synthesize_information()
    
    if result["status"] == "success":
        print("✅ Information Synthesis:")
        print(result["synthesis"][:500] + "..." if len(result["synthesis"]) > 500 else result["synthesis"])
    else:
        print(f"❌ Error: {result['error']}")


def demo_citation_generation():
    """Demonstrate citation generation."""
    print("\n" + "="*60)
    print("CITATION GENERATION DEMO")
    print("="*60)
    
    api_key = get_api_key('openai')
    if not api_key:
        print("⚠️ OpenAI API key not found")
        return
    
    assistant = ResearchAssistant(api_key)
    
    # Add sample sources
    sources = [
        ResearchSource(
            title="Machine Learning in Healthcare: A Review",
            url="https://journal.example.com/ml-healthcare",
            content="Comprehensive review of ML applications",
            credibility_score=9.0,
            author="Dr. Jane Smith",
            publication_date="2024-01-15",
            source_type="academic_journal"
        ),
        ResearchSource(
            title="AI Ethics in Medical Practice",
            url="https://ethics.example.com/ai-medical",
            content="Discussion of ethical considerations",
            credibility_score=8.5,
            author="Prof. Michael Brown",
            publication_date="2023-12-20",
            source_type="academic_book"
        )
    ]
    
    assistant.research_context["sources"] = sources
    
    print("📝 Generating APA citations...")
    
    result = assistant.generate_citations("APA")
    
    if result["status"] == "success":
        print("✅ Generated Citations:")
        print(result["citations"])
    else:
        print(f"❌ Error: {result['error']}")


def demo_research_questions():
    """Demonstrate research question generation."""
    print("\n" + "="*60)
    print("RESEARCH QUESTIONS DEMO")
    print("="*60)
    
    api_key = get_api_key('openai')
    if not api_key:
        print("⚠️ OpenAI API key not found")
        return
    
    assistant = ResearchAssistant(api_key)
    
    topic = "Remote Work Impact on Employee Productivity"
    focus_areas = ["technology adoption", "work-life balance", "team collaboration"]
    
    print(f"❓ Generating research questions for: {topic}")
    print(f"Focus areas: {', '.join(focus_areas)}")
    
    result = assistant.generate_research_questions(topic, focus_areas)
    
    if result["status"] == "success":
        print("\n✅ Generated Research Questions:")
        print(result["research_questions"])
    else:
        print(f"❌ Error: {result['error']}")


def main():
    """Main function demonstrating the research assistant."""
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("🚀 Starting Research Assistant Demonstration")
    
    print("🔬 AI-Powered Research Assistant")
    print("This tool helps with research planning, source management, and information synthesis.")
    
    try:
        # Run all demonstrations
        demo_research_planning()
        demo_source_management()
        demo_information_synthesis()
        demo_citation_generation()
        demo_research_questions()
        
        print("\n" + "="*60)
        print("RESEARCH ASSISTANT FEATURES SUMMARY")
        print("="*60)
        print("✅ Research Planning - Methodology and objectives")
        print("✅ Source Management - Collection and credibility assessment")
        print("✅ Information Synthesis - Multi-source analysis")
        print("✅ Citation Generation - Academic formatting")
        print("✅ Research Questions - Comprehensive inquiry development")
        print("✅ Report Generation - Professional documentation")
        
        print("\n💡 Use Cases:")
        print("• Academic research projects")
        print("• Market research and analysis")
        print("• Literature reviews")
        print("• Policy research and analysis")
        print("• Competitive intelligence")
        print("• Due diligence investigations")
        
        logger.info("✅ Research Assistant demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error occurred: {e}")
        logger.info("💡 Check your API keys and internet connection")


if __name__ == "__main__":
    main()